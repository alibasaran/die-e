use backgammon::Backgammon;

pub mod backgammon;

fn main() {
    let mut bg = Backgammon::new();

    bg.board.0 = [0; 24];
    bg.board.0[23] = 2;
    bg.board.0[0] = -1;

    let entry_moves = Backgammon::get_normal_moves(&vec![1, 1], bg.board, -1);
    for tree in entry_moves {
        println!("{}", tree)
    }
}
