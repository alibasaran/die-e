use arrayvec::ArrayVec;
use rand::{distributions::WeightedIndex, prelude::Distribution, thread_rng};
use tch::{nn::Adam, Tensor};

use super::{encoding::decode, nnet::ResNet};

use crate::{
    backgammon::Backgammon,
    mcts::{alpha_mcts, alpha_mcts_probs, ACTION_SPACE_SIZE},
};

struct AlphaZeroConfig {
    learn_iterations: usize,
    self_play_iterations: usize,
}

struct AlphaZero {
    model: ResNet,
    optimizer: Adam,
    config: AlphaZeroConfig,
}

struct MemoryFragment {
    outcome: i8,   // Outcome of game
    ps: Tensor,    // Probabilities
    state: Tensor, // Encoded game state
}

impl AlphaZero {
    fn new(config: AlphaZeroConfig) -> Self {
        AlphaZero {
            model: ResNet::default(),
            optimizer: Adam::default(),
            config,
        }
    }

    fn self_play(&self) -> Vec<MemoryFragment> {
        let mut bg = Backgammon::new();

        let mut player = -1;
        let mut memory: Vec<MemoryFragment> = vec![];
        let mut current_state = bg.board;
        loop {
            bg.roll_die();
            // Get probabilities from mcts
            let pi = match alpha_mcts_probs(&bg, player, &self.model) {
                Some(pi) => pi,
                None => continue,
            };

            // Select an action from probabilities
            let selected_action = self.weighted_select_tensor_idx(&pi);

            // Save results to memory
            memory.push(MemoryFragment {
                outcome: player,
                ps: pi,
                state: bg.as_tensor(player as i64),
            });

            // Decode and play selected action
            let decoded_action = decode(selected_action as u32, bg.roll, player);
            current_state = Backgammon::get_next_state(current_state, &decoded_action, player);
            bg.board = current_state;

            if let Some(winner) = Backgammon::check_win_without_player(current_state) {
                return memory
                    .iter()
                    .map(|mem| MemoryFragment {
                        outcome: if mem.outcome == winner {
                            winner
                        } else {
                            -winner
                        },
                        ps: mem.ps.shallow_clone(),
                        state: mem.state.shallow_clone(),
                    })
                    .collect();
            }
            player *= -1;
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
