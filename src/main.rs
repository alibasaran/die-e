pub use backgammon::Backgammon;

pub mod backgammon;

fn main() {
    let mut bg = Backgammon::new();

    bg.board.0 = [0; 24];
    bg.board.1.0 = 1;
    bg.board.0[18] = 0;

    let entry_moves = Backgammon::get_entry_moves(&vec![6], bg.board, -1);
    for tree in entry_moves {
        println!("{}", tree)
    }
}
