pub use backgammon::Backgammon;

pub mod backgammon;
pub mod mcts;
use backgammon::Board;
use mcts::{mct_search, random_play};

use crate::mcts::roll_die;


struct Turn {
    player: i8,
    roll: (u8, u8),
    state: Board
}

fn main() {
    let mut bg = Backgammon::new();
    let mut turns: Vec<Turn> = vec![];
    // let roll = (3, 1); // roll_die();
    // println!("Rolled die {} - {}", roll.0, roll.1);
    // let actions = mct_search(bg, -1, roll);
    // println!("{:?}", actions);

    while Backgammon::check_win_without_player(bg.board).is_none() {
        let player_1_roll = roll_die();
        let player_1_action = mct_search(bg.clone(), -1, player_1_roll);

        let new_state = Backgammon::get_next_state(bg.board, 
            player_1_action.clone(), -1);

        println!("Player 1, roll: {:?}, action: {:?}", player_1_roll, player_1_action);
        Backgammon::display_board(&new_state);

        bg.board = new_state;

        let player_2_roll = roll_die();
        let player_2_action = random_play(bg.board, 1, player_2_roll);

        let new_state = Backgammon::get_next_state(bg.board, 
            player_2_action.clone(), 1);

        println!("Player 2, roll: {:?}, action: {:?}", player_2_roll, player_2_action);
        Backgammon::display_board(&new_state);

        bg.board = new_state;
    }

    let winner = Backgammon::check_win_without_player(bg.board).unwrap();
    println!("Winner is player {}", winner)
}
