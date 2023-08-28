pub use backgammon::Backgammon;

pub mod backgammon;
pub mod mcts;
pub mod alphazero;

use crate::alphazero::alphazero::{AlphaZero, AlphaZeroConfig};


fn main() {
    let config = AlphaZeroConfig {
        learn_iterations: 1,
        self_play_iterations: 4,
        batch_size: 2,
        num_epochs: 1,
    };
    let mut az = AlphaZero::new(config);
    az.learn();
}
