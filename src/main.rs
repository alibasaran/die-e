pub use backgammon::Backgammon;

pub mod backgammon;
pub mod mcts;
use mcts::mct_search;

use tch::Tensor;

fn main() {
    let bg = Backgammon::new();
    let actions = mct_search(bg, -1);
    println!("{:?}", actions);
}
