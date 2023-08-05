pub use backgammon::Backgammon;

pub mod backgammon;

fn main() {
    let mut bg = Backgammon::new();

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
