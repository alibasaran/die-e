
use crate::{tictactoe::TicTacToe, base::LearnableGame};

#[cfg(test)]
mod initializes_as_expected {
    use super::*;
    #[test]
    fn board_should_be_empty() {
        let ttt = TicTacToe::new();
        assert_eq!(ttt.board.into_iter().sum::<i8>(), 0);
    }

    #[test]
    fn player_minus_1_should_start() {
        let ttt = TicTacToe::new();
        assert_eq!(ttt.get_player(), -1);
    }

    #[test]
    fn id_should_be_0_on_default() {
        let ttt = TicTacToe::new();
        assert_eq!(ttt.board.into_iter().sum::<i8>(), 0);
    }
}

#[cfg(test)]
mod game_logic {
    use super::*;

    mod apply_move {
        use super::*;
        
        #[test]
        fn should_switch_player() {
            let mut ttt = TicTacToe::new();
            let old_player = ttt.get_player();
            ttt.apply_move(&0);
            assert_eq!(ttt.get_player(), -old_player)
        }

        #[test]
        fn should_mut_board_as_expected() {
            let mut ttt = TicTacToe::new();
            ttt.apply_move(&5);
            assert_eq!(ttt.board[5], -1)
        }
    }

    mod get_valid_moves {
        use itertools::Itertools;

        use super::*;
        #[test]
        fn empty_board_should_return_every_move() {
            let ttt = TicTacToe::new();
            let moves = ttt.get_valid_moves();
            assert_eq!(moves.len(), 9);
            assert_eq!(moves, (0..ttt.board.len() as u8).collect_vec())
        }

        #[test]
        fn full_board_should_return_empty_move() {
            let mut ttt = TicTacToe::new();
            ttt.board = [1, 1, -1, -1, 1, 1, -1, -1, -1];
            assert!(ttt.get_valid_moves().is_empty());
        }

        #[test]
        fn normal_case() {
            let mut ttt = TicTacToe::new();
            ttt.board = [
                1, 1, -1, 
                -1, 1, 0, 
                -1, 0, 0
            ];
            assert!(ttt.get_valid_moves().eq(&vec![5, 7, 8]))
        }
    }

    mod check_winner {
        use super::*;
        
        #[test]
        fn is_none_when_no_winners() {
            let ttt = TicTacToe::new();
            let winner = ttt.check_winner();
            assert!(winner.is_none())
        }

        #[test]
        fn is_minus_1_when_minus_1_is_winner() {
            let mut ttt = TicTacToe::new();
            ttt.board = [
                1, 1, -1, 
                1, -1, -1, 
                -1, 0, 0
            ];
            let winner = ttt.check_winner();
            assert_eq!(winner.unwrap(), -1)
        }

        #[test]
        fn is_1_when_1_is_winner() {
            let mut ttt = TicTacToe::new();
            ttt.board = [
                -1, -1, 1, 
                -1, 1, -1, 
                1, 0, 0
            ];
            let winner = ttt.check_winner();
            assert_eq!(winner.unwrap(), 1)
        }
    }

}