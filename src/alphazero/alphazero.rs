
use indicatif::{MultiProgress};
use itertools::{multiunzip, Itertools};
use rand::{distributions::WeightedIndex, prelude::Distribution, seq::SliceRandom, thread_rng};

use std::{
    cmp::min,
    path::Path,
};
use tch::{
    nn::{self, Adam, Optimizer, OptimizerConfig},
    Tensor,
};


use super::nnet::ResNet;

use crate::{
    backgammon::{Backgammon, Actions},
    constants::{DEFAULT_TYPE, DEVICE},
    mcts::{
        alpha_mcts::{alpha_mcts},
    },
    MCTS_CONFIG,
};

#[derive(Debug)]
pub struct AlphaZeroConfig {
    pub temperature: f64,
    pub learn_iterations: usize,
    pub self_play_iterations: usize,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub model_path: Option<String>
}

pub struct AlphaZero {
    pub model: ResNet,
    pub(crate) optimizer: Optimizer,
    pub config: AlphaZeroConfig,
    pub(crate) pb: MultiProgress,
}
#[derive(Debug)]
pub struct MemoryFragment {
    pub outcome: i8,   // Outcome of game
    pub ps: Tensor,    // Probabilities
    pub state: Tensor, // Encoded game state
}

/*
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
*/


impl AlphaZero {
    pub fn new(config: AlphaZeroConfig) -> Self {
        let mut vs = nn::VarStore::new(*DEVICE);

        println!("Initializing AlphaZero...\nDevice: {:?}\n{:?}\n{:?}", *DEVICE, &config, MCTS_CONFIG);

        if let Some(m_path) = &config.model_path {
            match vs.load(m_path) {
                Ok(_) => println!("Successfully loaded model on path: {}", m_path),
                Err(e) => panic!("failed to load model: {}", e),
            }
        }

        let opt = Adam::default().wd(1e-4).build(&vs, 1e-4).unwrap();

        AlphaZero {
            model: ResNet::new(vs),
            optimizer: opt,
            config,
            pb: MultiProgress::new(),
        }
    }

    pub fn weighted_select_tensor_idx(pi: &Tensor) -> usize {
        let mut rng = thread_rng();
        let weights_iter = match pi.iter::<f64>() {
            Ok(iter) => iter,
            Err(err) => panic!("cannot convert pi to iterator, got error: {}", err),
        };
        let dist = WeightedIndex::new(weights_iter).unwrap();
        dist.sample(&mut rng)
    }

    pub fn get_next_move_for_state(&self, current_state: &Backgammon) -> Actions {
        let mut pi = match alpha_mcts(current_state, &self.model) {
            Some(pi) => pi,
            None => return vec![]
        };
        let temperatured_pi = pi.pow_(1.0 / self.config.temperature);
        let selected_action = AlphaZero::weighted_select_tensor_idx(&temperatured_pi);
        current_state.decode(selected_action as u32)
    }

    pub fn save_training_data(&self, data: &[MemoryFragment], path: &Path) {
        if !path.exists() {
            panic!("path: {} does not exist!", path.to_str().unwrap())
        }

        let (outcomes, ps_values, states): (Vec<i8>, Vec<Tensor>, Vec<Tensor>) =
            multiunzip(data.iter().map(|fragment| {
                (
                    fragment.outcome,
                    fragment.ps.shallow_clone(),
                    fragment.state.shallow_clone(),
                )
            }));
        // One channel for outcomes, one channel for ps, one channel for states,
        // Because mps is not supported we convert to cpu
        let ps = Tensor::stack(&ps_values, 0).to_device(tch::Device::Cpu).unsqueeze(2);
        let states = Tensor::stack(&states, 0).to_device(tch::Device::Cpu).squeeze();
        let outcomes = Tensor::from_slice(&outcomes);

        match (
            ps.save(path.join("ps")),
            states.save(path.join("states")),
            outcomes.save(path.join("outcomes")),
        ) {
            (Ok(_), Ok(_), Ok(_)) => (),
            _ => panic!("unable to save training data!"),
        }
    }

    /**
     Loads training data (ps, states, outcomes) given a folder path
     
     path: the Path to load the data from can end with:
     * lrn-x for all training data under the learn 
     */
    pub fn load_training_data(&self, path: &Path) -> Vec<MemoryFragment> {
        if !path.exists() {
            panic!("path: {} does not exist!", path.to_str().unwrap())
        }
        let ps = Tensor::load(path.join("ps")).unwrap().squeeze();
        let states = Tensor::load(path.join("states")).unwrap().squeeze();
        let outcomes = Tensor::load(path.join("outcomes")).unwrap().squeeze();

        let data_size = ps.size()[0];
        (0..data_size).map(|data_idx| {
            MemoryFragment {
                outcome: outcomes.get(data_idx).int64_value(&[]) as i8,
                ps: ps.get(data_idx).squeeze().shallow_clone(),
                state: states.get(data_idx).unsqueeze(2).shallow_clone(),
            }
        }).collect_vec()
    }

    pub fn train(&mut self, memory: &mut Vec<MemoryFragment>) {
        let mut rng = thread_rng();
        memory.shuffle(&mut rng);
        for batch_idx in (0..memory.len()).step_by(self.config.batch_size) {
            let sample = &memory[batch_idx..min(batch_idx + self.config.batch_size, memory.len())];
            let (outcomes, ps_values, states): (Vec<i8>, Vec<Tensor>, Vec<Tensor>) =
                multiunzip(sample.iter().map(|fragment| {
                    (
                        fragment.outcome,
                        fragment.ps.shallow_clone(),
                        fragment.state.shallow_clone(),
                    )
                }));

            // Format tensors to process
            let outcome_tensor = Tensor::from_slice(&outcomes).unsqueeze(1).to_device_(
                *DEVICE,
                DEFAULT_TYPE,
                true,
                false,
            );

            let ps_tensor = Tensor::stack(&ps_values, 0).to_device_(
                *DEVICE,
                DEFAULT_TYPE,
                true,
                false,
            );

            let state_tensor = Tensor::stack(&states, 0).squeeze().to_device_(
                *DEVICE,
                DEFAULT_TYPE,
                true,
                false,
            );

            let (out_policy, out_value) = self.model.forward_t(&state_tensor, true);
            let out_policy = out_policy.squeeze();

            // Calculate loss
            let policy_loss = out_policy.cross_entropy_loss::<Tensor>(
                &ps_tensor,
                None,
                tch::Reduction::Mean,
                -100,
                0.0,
            );
            let outcome_loss = out_value.mse_loss(&outcome_tensor, tch::Reduction::Mean);
            let loss = policy_loss + outcome_loss;

            self.optimizer.zero_grad();
            loss.backward();
            self.optimizer.step();
        }
    }

    pub fn save_current_model(&self, save_path: &Path) -> Result<(), tch::TchError> {
        self.model.vs.save(save_path)
    }
}