use arrayvec::ArrayVec;
use rand::{distributions::WeightedIndex, prelude::Distribution, thread_rng};
use tch::{nn::Adam, Tensor};

use super::{encoding::decode, nnet::ResNet};

use crate::{
    backgammon::Backgammon,
    mcts::{alpha_mcts, ACTION_SPACE_SIZE},
};

pub struct AlphaZeroConfig {
    pub learn_iterations: usize,
    pub self_play_iterations: usize,
}

pub struct AlphaZero {
    model: ResNet,
    optimizer: Adam,
    config: AlphaZeroConfig,
}
#[derive(Debug)]
pub struct MemoryFragment {
    outcome: i8,   // Outcome of game
    ps: Tensor,    // Probabilities
    state: Tensor, // Encoded game state
}

impl AlphaZero {
    pub fn new(config: AlphaZeroConfig) -> Self {
        AlphaZero {
            model: ResNet::default(),
            optimizer: Adam::default(),
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

        loop {
            println!("Rolled die: {:?}", bg.roll);
            println!("Player: {}", player);
            bg.display_board();
            // Get probabilities from mcts
            let pi = match alpha_mcts(&bg, player, &self.model, is_second_play) {
                Some(pi) => pi,
                None => {
                    println!("No valid moves!");
                    is_second_play = false;
                    player *= -1;
                    bg.roll_die();
                    continue;
                }
            };

            // Select an action from probabilities
            let selected_action = self.weighted_select_tensor_idx(&pi);

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
            }
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
