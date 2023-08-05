pub use backgammon::Backgammon;

pub mod backgammon;

fn main() {
    let mut bg = Backgammon::new();

    let entry_moves = Backgammon::get_normal_moves(&vec![6, 1], bg.board, 1);
    for tree in entry_moves {
        let sequences = Backgammon::extract_sequences(&tree);
        for (index, sequence) in sequences.iter().enumerate() {
            println!("Sequence {}: {:?}", index + 1, sequence);
        }
        println!("{}", tree)
    }
}
