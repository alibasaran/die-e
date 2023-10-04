mod tictactoe_test;

use itertools::Itertools;
use serde::{Serialize, Deserialize};

use crate::{base::LearnableGame, constants::DEFAULT_TYPE};

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct TicTacToe {
    player: i8,
    // 0 | 1 | 2 
    // 3 | 4 | 5
    // 6 | 7 | 8
    pub board: [i8; 9],
    id: usize
}

impl LearnableGame for TicTacToe {
    type Move = u8;

    const EMPTY_MOVE: Self::Move = 10;

    const ACTION_SPACE_SIZE: i64 = 9;
    const CONV_OUTPUT_SIZE: i64 = 9;
    const N_INPUT_CHANNELS: i64 = 3;
    const N_FILTERS: i64 = 64;
    const N_RES_BLOCKS: i64 = 4;

    const IS_DETERMINISTIC: bool = true;

    fn new() -> Self {
        TicTacToe { player: -1, board: [0; 9], id: 0 }
    }

    fn name() -> String {
        String::from("tictactoe")
    }

    fn get_valid_moves(&self) -> Vec<Self::Move> {
        let mut moves = vec![];
        for (i, &v) in self.board.iter().enumerate() {
            if v == 0 {
                moves.push(i.try_into().unwrap())
            }
        }
        moves
    }

    fn apply_move(&mut self, action: &Self::Move) {
        self.board[*action as usize] = self.player;
        self.player *= -1
    }

    fn skip_turn(&mut self) {
        self.player *= -1
    }

    fn get_player(&self) -> i8 {
        self.player
    }
    // Returns 0 if game is a draw
    fn check_winner(&self) -> Option<i8> {
        let winning_combinations: [[usize; 3]; 8] = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
            [0, 4, 8], [2, 4, 6],            // Diagonals
        ];

        for combination in winning_combinations.iter() {
            let a = self.board[combination[0]];
            let b = self.board[combination[1]];
            let c = self.board[combination[2]];

            if a != 0 && a == b && b == c {
                return Some(a);
            }
        }
        if self.board.iter().all(|&v| v != 0) {
            return Some(0)
        } 
        None
    }

    fn as_tensor(&self) -> tch::Tensor {
        let board_tensor = tch::Tensor::from_slice(&self.board)
            .view([3, 3]);
        tch::Tensor::stack(
            &[
                board_tensor.eq(-1),
                board_tensor.eq(0),
                board_tensor.eq(1),
            ], 0
        ).unsqueeze(0)
        .to_dtype(DEFAULT_TYPE, true, false)
    }

    fn decode(&self, action: u32) -> Self::Move {
        action.try_into().unwrap()
    }

    fn encode(&self, action: &Self::Move) -> u32 {
        *action as u32
    }

    fn get_id(&self) -> usize {
        self.id
    }

    fn set_id(&mut self, new_id: usize) {
        self.id = new_id
    }

    fn to_pretty_str(&self) -> String {
        format!("{}|{}|{}\n{}|{}|{}\n{}|{}|{}",
            self.board[0], self.board[1], self.board[2],
            self.board[3], self.board[4], self.board[5],
            self.board[6], self.board[7], self.board[8], 
        )
    }
}