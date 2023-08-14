pub use backgammon::Backgammon;

pub mod backgammon;
pub mod mcts;

use tch::Tensor;

fn main() {
    let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();
}

fn backgammon() {
    let bg = Backgammon::new();

    let normal_moves = Backgammon::get_normal_moves(&vec![6, 1], bg.board, 1);
    let all_sequences = Backgammon::extract_sequences_list(normal_moves);
    let sequences = all_sequences.clone();
    let unique_sequences = Backgammon::remove_duplicate_states(bg.board, sequences, 1);

    println!("All Sequences:");
    for (index, sequence) in all_sequences.iter().enumerate() {
        println!("Sequence {}: {:?}", index + 1, sequence);
    }

    println!("Unique Sequences:");
    for (index, sequence) in unique_sequences.iter().enumerate() {
        println!("Unique Sequence {}: {:?}", index + 1, sequence);
    }
}
