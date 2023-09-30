
use config::{Config};
use indicatif::MultiProgress;
use itertools::{multiunzip, Itertools};
use rand::{distributions::WeightedIndex, prelude::Distribution, seq::SliceRandom, thread_rng};

use std::{
    cmp::min,
    path::{Path, PathBuf},
};
use tch::{
    nn::{self, Adam, Optimizer, OptimizerConfig},
    Tensor,
};


use super::nnet::ResNet;

use crate::{
    constants::{DEFAULT_TYPE, DEVICE},
    mcts::alpha_mcts::alpha_mcts, MctsConfig, base::LearnableGame,
};

#[derive(Debug)]
pub struct AlphaZeroConfig {
    pub temperature: f64,
    pub learn_iterations: usize,
    pub self_play_iterations: usize,
    pub num_epochs: usize,
    pub training_batch_size: usize,
    pub num_self_play_batches: usize,
}

impl AlphaZeroConfig {
    pub fn from_config(conf: &config::Config) -> Result<Self, config::ConfigError> {
        Ok(AlphaZeroConfig {
            temperature: conf.get_float("temperature")?,
            learn_iterations: conf.get_int("learn_iterations")? as usize,
            self_play_iterations: conf.get_int("self_play_iterations")? as usize,
            num_epochs: conf.get_int("num_epochs")? as usize,
            training_batch_size: conf.get_int("training_batch_size")? as usize,
            num_self_play_batches: conf.get_int("num_self_play_batches")? as usize,
        })
    }
}

pub struct OptimizerParams {
    wd: f64, 
    lr: f64
}

impl OptimizerParams {
    pub fn from_config(conf: &Config) -> Result<Self, config::ConfigError> {
        Ok(OptimizerParams { 
            wd: conf.get_float("wd")?, 
            lr: conf.get_float("lr")?
        })
    }
}

pub struct AlphaZero {
    pub model: ResNet,
    pub(crate) optimizer: Optimizer,
    pub config: AlphaZeroConfig,
    pub mcts_config: MctsConfig,
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
    pub fn new(model_path: Option<PathBuf>, config: AlphaZeroConfig, mcts_config: MctsConfig, op: OptimizerParams) -> Self {
        let mut vs = nn::VarStore::new(*DEVICE);

        println!("Initializing AlphaZero...\nDevice: {:?}\n{:?}\n{:?}", *DEVICE, &config, mcts_config);

        match model_path {
            Some(m_path) => match vs.load(&m_path) {
                Ok(_) => println!("Successfully loaded model on path: {}", m_path.to_str().unwrap()),
                Err(e) => panic!("failed to load model: (might be a large error log) \n{}", e),
            },
            None if Path::new("./models/best_model.ot").exists() => {
                let bm_path = Path::new("./models/best_model.ot");
                match vs.load(bm_path) {
                    Ok(_) => println!("Successfully loaded best model"),
                    Err(e) => panic!("failed to load best model: {}", e),
                }
            },
            None => println!("No best model found, initialized from scratch")
        }

        let opt = Adam::default().wd(op.wd).build(&vs, op.lr).unwrap();

        AlphaZero {
            model: ResNet::new(vs),
            optimizer: opt,
            config,
            mcts_config,
            pb: MultiProgress::new(),
        }
    }

    pub fn from_config(model_path: Option<PathBuf>, config: &Config) -> Self {
        let az_config = match AlphaZeroConfig::from_config(config){
            Ok(config) => config,
            Err(e) => panic!("Unable to load AlphaZero config, {}", e),
        };
        let mcts_config = match MctsConfig::from_config(config) {
            Ok(config) => config,
            Err(e) => panic!("Unable to load MCTS config, {}", e),
        };
        let op = match OptimizerParams::from_config(config) {
            Ok(op) => op,
            Err(e) => panic!("Unable to load optimizer params, {}", e),
        };
        AlphaZero::new(model_path, az_config, mcts_config, op)
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

    pub fn get_next_move_for_state<T: LearnableGame>(&self, current_state: &T) -> T::Move {
        let mut pi = match alpha_mcts(current_state, &self.model, &self.mcts_config) {
            Some(pi) => pi,
            None => return T::EMPTY_MOVE
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
        let ps = Tensor::stack(&ps_values, 0).to_device(tch::Device::Cpu);
        let states = Tensor::concat(&states, 0).to_device(tch::Device::Cpu);
        let outcomes = Tensor::from_slice(&outcomes);

        match (
            ps.save(path.join("ps.ot")),
            states.save(path.join("states.ot")),
            outcomes.save(path.join("outcomes.ot")),
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
    pub fn load_training_data(path: &Path) -> Vec<MemoryFragment> {
        if !path.exists() {
            panic!("path: {} does not exist!", path.to_str().unwrap())
        }
        let ps = Tensor::load(path.join("ps.ot")).unwrap().squeeze();
        let states = Tensor::load(path.join("states.ot")).unwrap().squeeze();
        let outcomes = Tensor::load(path.join("outcomes.ot")).unwrap().squeeze();

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
        for batch_idx in (0..memory.len()).step_by(self.config.training_batch_size) {
            let sample = &memory[batch_idx..min(batch_idx + self.config.training_batch_size, memory.len())];
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