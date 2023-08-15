pub use backgammon::Backgammon;

pub mod backgammon;
pub mod mcts;
use mcts::mct_search;

use crate::mcts::roll_die;

fn main() {
    let bg = Backgammon::new();
    let roll = (3, 1); // roll_die();
    println!("Rolled die {} - {}", roll.0, roll.1);
    let actions = mct_search(bg, -1, roll);
    println!("{:?}", actions);
}
