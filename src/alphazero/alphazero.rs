use itertools::multiunzip;
use rand::{distributions::WeightedIndex, prelude::Distribution, seq::SliceRandom, thread_rng};
use std::cmp::min;
use tch::{
    nn::{self, Optimizer, OptimizerConfig, Adam},
    Tensor,
};

use super::{encoding::decode, nnet::ResNet};

use crate::{backgammon::Backgammon, constants::DEVICE, mcts::alpha_mcts::alpha_mcts};

pub struct AlphaZeroConfig {
    pub temperature: f64,
    pub learn_iterations: usize,
    pub self_play_iterations: usize,
    pub num_epochs: usize,
    pub batch_size: usize,
}

pub struct AlphaZero {
    model: ResNet,
    optimizer: Optimizer,
    config: AlphaZeroConfig,
}
#[derive(Debug)]
pub struct MemoryFragment {
    outcome: i8,   // Outcome of game
    ps: Tensor,    // Probabilities
    state: Tensor, // Encoded game state
}

/*
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
*/

impl AlphaZero {
    pub fn new(config: AlphaZeroConfig) -> Self {
        let vs = nn::VarStore::new(*DEVICE);
        let opt = Adam::default()
            .wd(1e-4)
            .build(&vs, 1e-3).unwrap();
            
        AlphaZero {
            model: ResNet::new(vs),
            optimizer: opt,
            config,
        }
    }

    pub fn self_play(&self) -> Vec<MemoryFragment> {
        println!("Started self play");
        let mut bg = Backgammon::new();
        bg.roll_die();

        let mut player = -1;
        let mut memory: Vec<MemoryFragment> = vec![];
        // Is second play is a workaround for the case where doubles are rolled
        // Ex. when a player rolls 6,6 they can play 6 4-times
        // rather than the usual two that would be played on a normal role
        // So we feed doubles into the network as two back to back normal roles from the same player
        let mut is_second_play = false;
        let mut hard_lock = 0;
        loop {
            println!("Rolled die: {:?}", bg.roll);
            println!("Player: {}", player);
            bg.display_board();
            // Get probabilities from mcts
            let mut pi = match alpha_mcts(&bg, player, &self.model, is_second_play) {
                Some(pi) => pi,
                None => {
                    println!("No valid moves!");
                    is_second_play = false;
                    player *= -1;
                    bg.roll_die();
                    continue;
                }
            };

            // Apply temperature to pi
            let temperatured_pi = pi.pow_(1.0 / self.config.temperature);
            // Select an action from probabilities
            let selected_action = self.weighted_select_tensor_idx(&temperatured_pi);

            // Save results to memory
            memory.push(MemoryFragment {
                outcome: player,
                ps: pi,
                state: bg.as_tensor(player as i64, is_second_play),
            });

            // Decode and play selected action
            let decoded_action = decode(selected_action as u32, bg.roll, player);
            println!("Played action: {:?}\n\n", decoded_action);
            bg.apply_move(&decoded_action, player);

            if let Some(winner) = Backgammon::check_win_without_player(bg.board) {
                return memory
                    .iter()
                    .map(|mem| MemoryFragment {
                        outcome: if mem.outcome == winner { 1 } else { -1 },
                        ps: mem.ps.shallow_clone(),
                        state: mem.state.shallow_clone(),
                    })
                    .collect();
            } else if hard_lock == 10 {
                return memory
                    .iter()
                    .map(|mem| MemoryFragment {
                        outcome: if mem.outcome == 1 { 1 } else { -1 },
                        ps: mem.ps.shallow_clone(),
                        state: mem.state.shallow_clone(),
                    })
                    .collect();
            }
            hard_lock += 1;
            // If double roll
            if bg.roll.1 == bg.roll.0 && !is_second_play {
                is_second_play = true;
            } else {
                is_second_play = false;
                player *= -1;
                bg.roll_die();
            }
        }
    }

    fn train(&mut self, memory: &mut Vec<MemoryFragment>) {
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
                tch::Kind::Float,
                false,
                false,
            );

            let ps_tensor =
                Tensor::stack(&ps_values, 0).to_device_(*DEVICE, tch::Kind::Float, false, false);

            let state_tensor = Tensor::stack(&states, 0).squeeze().to_device_(
                *DEVICE,
                tch::Kind::Float,
                false,
                false,
            );

            let (out_policy, out_value) = self.model.forward_t(&state_tensor, true);
            let out_policy = out_policy.squeeze();

            println!("ps_tensor size: {:?}", ps_tensor.size());
            println!("out_policy size: {:?}", out_policy.size());
            println!("out_value size: {:?}", out_value.size());
            println!("outcome_tensor size: {:?}", outcome_tensor.size());

            // Calculate loss
            let policy_loss = out_policy.to_device(*DEVICE).cross_entropy_loss::<Tensor>(
                &ps_tensor,
                None,
                tch::Reduction::Mean,
                -100,
                0.0,
            );
            let outcome_loss = out_value
                .to_device(*DEVICE)
                .mse_loss(&outcome_tensor, tch::Reduction::Mean);
            let loss = policy_loss + outcome_loss;

            self.optimizer.zero_grad();
            loss.backward();
            self.optimizer.step();
        }
    }

    pub fn learn(&mut self) {
        for i in 0..self.config.learn_iterations {
            let mut memory: Vec<MemoryFragment> = vec![];
            for _ in 0..self.config.self_play_iterations {
                let mut res = self.self_play();
                memory.append(&mut res);
            }
            for _ in 0..self.config.num_epochs {
                self.train(&mut memory);
            }
            let model_save_path = format!("./models/model_{}.pt", i);
            match self.model.vs.save(&model_save_path) {
                Ok(_) => println!(
                    "Iteration {} saved successfully, path: {}",
                    i, &model_save_path
                ),
                Err(e) => println!(
                    "Unable to save model: {}, caught error: {}",
                    &model_save_path, e
                ),
            }
        }
    }

    fn weighted_select_tensor_idx(&self, pi: &Tensor) -> usize {
        let mut rng = thread_rng();
        let weights_iter = match pi.iter::<f64>() {
            Ok(iter) => iter,
            Err(err) => panic!("cannot convert pi to iterator, got error: {}", err),
        };
        let dist = WeightedIndex::new(weights_iter).unwrap();
        dist.sample(&mut rng)
    }
}
