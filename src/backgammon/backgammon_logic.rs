use itertools::Itertools;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fmt, vec};
use tch::Tensor;

use crate::{constants::DEFAULT_TYPE, base::LearnableGame};

// (the board itself, pieces_hit, pieces_collected)
pub type Board = ([i8; 24], (u8, u8), (u8, u8));
// (from, to) if to == -1 then it is collection, if from == -1 then it is putting a hit piece back
pub type Actions = Vec<(i8, i8)>;

// Player1 should always be -1 and Player2 should always be 1
#[repr(i8)]
pub enum Player {
    ONE = -1,
    TWO = 1,
}

#[derive(Debug)]
pub struct ActionNode {
    pub value: (i8, i8),
    pub children: Vec<ActionNode>,
}

impl fmt::Display for ActionNode {
    #[cfg(not(tarpaulin_include))]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format_tree(f, "", true)
    }
}

impl PartialEq for ActionNode {
    #[cfg(not(tarpaulin_include))]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.children == other.children
    }
}

impl ActionNode {
    #[cfg(not(tarpaulin_include))]
    fn format_tree(&self, f: &mut fmt::Formatter<'_>, prefix: &str, is_last: bool) -> fmt::Result {
        let node_prefix = if is_last { "└── " } else { "├── " };
        let child_prefix = if is_last { "    " } else { "│   " };

        // Print the current node's value
        writeln!(f, "{}{}{:?}", prefix, node_prefix, self.value)?;

        // Recursively print the children
        for (index, child) in self.children.iter().enumerate() {
            let is_last_child = index == self.children.len() - 1;
            let new_prefix = format!("{}{}", prefix, child_prefix);
            child.format_tree(f, &new_prefix, is_last_child)?;
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Backgammon {
    pub board: Board,
    pub roll: (u8, u8),
    pub player: i8,
    pub is_second_play: bool,
    pub id: usize // used in model vs model play
}

impl Default for Backgammon {
    fn default() -> Self {
        Self::new()
    }
}

impl LearnableGame for Backgammon {

    type Move = Actions;

    const EMPTY_MOVE: Self::Move = vec![];
    const IS_DETERMINISTIC: bool = false;
    const ACTION_SPACE_SIZE: i64 = 1352;
    const N_INPUT_CHANNELS: i64 = 6;
    const CONV_OUTPUT_SIZE: i64 = 24;

    fn new() -> Self {
        Backgammon {
            board: (
                [
                    2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2,
                ],
                (0, 0),
                (0, 0),
            ),
            roll: (0, 0),
            player: -1,
            is_second_play: false,
            id: 0,
        }
    }

    fn roll_die(&mut self) -> (u8, u8) {
        let mut rng = rand::thread_rng();
        self.roll = (rng.gen_range(1..=6), rng.gen_range(1..=6));
        self.roll
    }

    fn check_winner(&self) -> Option<i8> {
        Backgammon::check_win_without_player(self.board)
    }

    fn to_pretty_str(&self) -> String {
        let board = self.board.0;
        let board_len = board.len();

        let board_bot = (0..board_len/2).rev().map(|i| i.to_string()).collect_vec();
        let board_top = (board_len/2..board_len).map(|i| i.to_string()).collect_vec();
        let mut inner_board = vec![];
        
        for i in 1..=6 {
            let mut row = vec![];
            for pos in (0..board_len/2).rev() {
                let piece_info = board[pos];
                if i == 6 && i <= piece_info.abs() {
                    row.push(format!("+{}", piece_info.abs() - 5))
                } else if i <= piece_info.abs() {
                    let symbol = if piece_info < 0 { String::from("x") } else {String::from("o")};
                    row.push(symbol)
                } else {
                    row.push(String::from(" "))
                }
            }
            inner_board.push(row)
        }

        inner_board.push(vec![String::from(" "); inner_board[0].len()]);

        for i in (1..=6).rev() {
            let mut row = vec![];
            for piece_info in board.iter().take(board_len).skip(board_len/2) {
                if i == 6 && i <= piece_info.abs() {
                    row.push(format!("+{}", piece_info.abs() - 5))
                } else if i <= piece_info.abs() {
                    let symbol = if *piece_info < 0 { String::from("x") } else {String::from("o")};
                    row.push(symbol)
                } else {
                    row.push(String::from(" "))
                }
            }
            inner_board.push(row);
        }

        let mut final_board = vec![board_bot];
        final_board.extend(inner_board);
        final_board.push(board_top);
        final_board.iter_mut().for_each(|row| {
            row.insert(row.len()/2, String::from("|"));
            row.insert(0, String::from("|"));
            row.push(String::from("|"));

        });
        
        let final_board_str = final_board.iter().rev().map(|row| {
            row.join("\t")
        }).join("\n");

        let player_as_string = if self.player == -1 {String::from("Player 1")} else {String::from("Player 2")};
        let player1_info = format!("Player 1:\n\tBroken Pieces: {}\n\tPieces Collected:{}", &self.board.1.0, &self.board.2.0);
        let player2_info = format!("Player 2:\n\tBroken Pieces: {}\n\tPieces Collected:{}", &self.board.1.1, &self.board.2.1);
        
        let info_string = format!("Current turn: {}\tRoll: {:?}\n{}\n{}", player_as_string, &self.roll, player1_info, player2_info);

        let line_break = "=".repeat(110);

        format!("{}\n{}\n{}\n{}", info_string, line_break, final_board_str, line_break)
    }

    fn apply_move(&mut self, actions: &Actions) {
        let next_state = Self::get_next_state(self.board, actions, self.player);
        self.board = next_state;
        if self.roll.0 == self.roll.1 && !self.is_second_play {
            self.is_second_play = true;
        } else {
            self.is_second_play = false;
            self.player *= -1;
            self.roll_die();
        }
    }

    fn get_player(&self) -> i8 {
        self.player
    }
    
    fn skip_turn(&mut self) {
        self.is_second_play = false;
        self.player *= -1;
        self.roll_die();
    }
    
    fn as_tensor(&self) -> Tensor {
        assert!(self.roll != (0, 0), "die has not been rolled!");
    
        let board = self.board;
        let full_options = (DEFAULT_TYPE, tch::Device::Cpu);
    
        let board_tensor = Tensor::from_slice(&board.0)
            .view([4, 6, 1]);
        let player_tensor = Tensor::full(24, self.player as i64, full_options).view([4, 6, 1]);
        let hit_tensor = Tensor::cat(
            &[
                Tensor::full(12, board.1 .0 as i64, full_options),
                Tensor::full(12, board.1 .1 as i64, full_options),
            ],
            0,
        )
        .view([4, 6, 1]);
    
        let collect_tensor = Tensor::cat(
            &[
                Tensor::full(12, board.2 .0 as i64, full_options),
                Tensor::full(12, board.2 .1 as i64, full_options),
            ],
            0,
        )
        .view([4, 6, 1]);
    
        let roll_tensor = Tensor::cat(
            &[
                Tensor::full(12, self.roll.0 as i64, full_options),
                Tensor::full(12, self.roll.1 as i64, full_options),
            ],
            0,
        )
        .view([4, 6, 1]);
    
        let second_play_tensor = if self.is_second_play {
            Tensor::full(24, 1, full_options).view([4, 6, 1])
        } else {
            Tensor::full(24, 0, full_options).view([4, 6, 1])
        };
    
        Tensor::stack(
            &[
                board_tensor,
                player_tensor,
                hit_tensor,
                collect_tensor,
                roll_tensor,
                second_play_tensor,
            ],
            2,
        )
        .permute([3, 2, 0, 1])
    }

    fn get_id(&self) -> usize {
        self.id
    }

    fn set_id(&mut self, new_id: usize) {
        self.id = new_id
    }

    fn encode(&self, actions: &Actions) -> u32 {
        assert!(actions.len() <= 2, "encoding for actions > 2 is not implemented!");
    
        // return special case for when there are no actions
        if actions.is_empty() {
            return 1351;
        }
        
        // get the roll and the high and low roll values
        let roll = self.roll;
        let (_high_roll, low_roll) = if roll.0 > roll.1 { (roll.0, roll.1) } else { (roll.1, roll.0) };
        let mut low_roll_first_flag = false;
        let mut low_roll_second_flag = false;
    
        // get the minimum roll values required to be able to play the actions provided
        let mut minimum_rolls = actions.iter().map(|&(from, to)| {
            match (from, to) {
                (-1, t) if t < 6 => (t + 1) as u8,
                (-1, t) if t > 17 => (24 - t) as u8,
                (f, -1) if f < 6 => (f - (-1)) as u8,
                (f, -1) if f > 17 => (24 - f) as u8,
                (f, t) => (f - t).unsigned_abs(),
            }
        }).collect::<Vec<u8>>();
    
        // if only a single move is played, set the second minimum roll to be 0
        if minimum_rolls.len() == 1 {minimum_rolls.push(0);}
    
        /*
         * use base 26 encoding to encode the actions
         * the values 0-23 is reserved for 'normal' moves (i.e. moves that are not from the bar or are not collection moves)
         * and corresponds to the 'from' value of the action
         * 24 is reserved for moves from the bar
         * 25 is reserved for single moves (i.e. the value of the non-existent 'second move' of a single-move action)
         * add the corresponding value for the first move and the corresponding value for the second move times 26
         */ 
        let mut encode_sum = 0;
        for (i, &(from, to)) in actions.iter().enumerate() {
            match i {
                0 => {match (from, to) {
                    (-1, t) if t < 6 => {
                        encode_sum += 24;

                        // raise the low_roll_first_flag if the first move is certainly the low roll
                        let distance = (t - (-1)) as u8;
                        low_roll_first_flag = distance == low_roll;
                    },
                    (-1, t) if t > 17 => {
                        encode_sum += 24;

                        // raise the low_roll_first_flag if the first move is certainly the low roll
                        let distance = (24 - t) as u8;
                        low_roll_first_flag = distance == low_roll;
                    },
                    (f, -1) if f < 6 => {encode_sum += f as u32;},
                    (f, -1) if f > 17 => {encode_sum += f as u32;},
                    (f, _) => {
                        encode_sum += f as u32; 
                        // raise the low_roll_first_flag if the first move is certainly the low roll
                        low_roll_first_flag = minimum_rolls.first().unwrap() == &low_roll;
                    },
                }},
                1 => {match (from, to) {
                    (-1, t) if t < 6 => {
                        encode_sum += 26 * 24;

                        // raise the low_roll_second_flag if the first move is certainly the low roll
                        let distance = (t - (-1)) as u8;
                        low_roll_second_flag = distance == low_roll;
                    },
                    (-1, t) if t > 17 => {
                        encode_sum += 26 * 24;

                        // raise the low_roll_first_flag if the first move is certainly the low roll
                        let distance = (24 - t) as u8;
                        low_roll_second_flag = distance == low_roll;
                    },
                    (f, -1) if f < 6 => {encode_sum += 26 * (f as u32);},
                    (f, -1) if f > 17 => {encode_sum += 26 * (f as u32);},
                    (f, _) => {
                        encode_sum += 26 * (f as u32);
                        // raise the low_roll_second flag if the first move is certainly the low roll
                        low_roll_second_flag = minimum_rolls.get(1).unwrap() == &low_roll;
                    },
                }},
                _ => unreachable!(),
            }
        }
    
        // add 26 * 25 to encode_sum if the action has a single move and reset low_roll_first_flag
        if actions.get(1).is_none() {low_roll_first_flag = false; encode_sum += 26 * 25}
    
        // compute whether the high roll was played first
        let high_roll_first = if low_roll_first_flag {false} else if low_roll_second_flag {true} else if minimum_rolls[1] != 0 {minimum_rolls[0] >= minimum_rolls[1]} else {minimum_rolls[0] > low_roll};
    
        // add 676 to the final value if the high roll was played first
        if high_roll_first { encode_sum } else { encode_sum + 676 }
    }
    
    fn decode(&self, action: u32) -> Actions {
        // decoding for the special value (1351) for empty actions
        if action == 1351 {
            return vec![];
        }
    
        let roll = self.roll;
        let player = self.player;
        let high_roll_first = action < 676;

        /*
         * extract the from values of the first and second action
         * the from value '24' suggests a move from the bar
         * note that the from2 value will be 25 if the action has a single move
         */
        let (from1, from2) = (if high_roll_first { action } else { action - 676 } % 26, 
                                        if high_roll_first { action } else { action - 676 } / 26);
        let single_action = from2 == 25;
        let (high_roll, low_roll) = if roll.0 > roll.1 { (roll.0, roll.1) } else { (roll.1, roll.0) };
        let (mut from1_i8, mut from2_i8) = (from1 as i8, from2 as i8);
        let (low_roll_i8, high_roll_i8) = (low_roll as i8, high_roll as i8);
    
        // convert from values from 24 to -1 only if the player is the second player (helps with computation)
        if from1_i8 == 24 && player == 1 { from1_i8 = -1; }
        if from2_i8 == 24 && player == 1 { from2_i8 = -1; }
    
        // extract 'to' values
        let (mut to1, mut to2) = if high_roll_first {
            (from1_i8 + high_roll_i8 * player, from2_i8 + low_roll_i8 * player)
        } else {
            (from1_i8 + low_roll_i8 * player, from2_i8 + high_roll_i8 * player)
        };
    
        // convert the 'to' and 'from' values from -1
        if to1 >= 24 || to1 <= -1 { to1 = -1; }
        if to2 >= 24 || to2 <= -1 { to2 = -1; }
        if from1_i8 == 24 { from1_i8 = -1; }
        if from2_i8 == 24 { from2_i8 = -1; }
    
        if single_action { vec![(from1_i8, to1)] } else { vec![(from1_i8, to1), (from2_i8, to2)] }
    }

    fn get_valid_moves(&self) -> Vec<Actions> {
        assert!(self.roll != (0, 0), "die has not been rolled!");
    
        let all_moves: Vec<u8> = match self.roll {
            (r0, r1) if r0 > r1 => vec![r0, r1],
            (r0, r1) => vec![r1, r0],
        };
        let action_trees = Self::_get_action_trees(&all_moves, self.board, self.player);
        // parse trees into actions here
        let actions = Self::extract_sequences_list(action_trees);
        Self::remove_duplicate_states(self.board, actions, self.player)
    }
}

impl Backgammon {

    pub fn init_with_fields(board: Board, player: i8, is_second_play: bool) -> Self {
        Backgammon {
            board,
            roll: (0, 0),
            player,
            is_second_play,
            id: 0
        }
    }

    pub fn display_board(&self) {
        let (points, pieces_hit, pieces_collected) = self.board;
        let mut total_player_1 = 0;
        let mut total_player_2 = 0;

        // Display the main board
        for point in 13..=24 {
            if points[point as usize - 1] < 0 {
                total_player_1 -= points[point as usize - 1];
            } else if points[point as usize - 1] > 0 {
                total_player_2 += points[point as usize - 1];
            }
            if point == 19 {
                print!("|");
            }
            print!("{:3} ", points[point as usize - 1]);
        }

        println!("\n");
        println!("------------------------------------------------");
        println!();

        for point in (1..=12).rev() {
            if points[point as usize - 1] < 0 {
                total_player_1 -= points[point as usize - 1];
            } else if points[point as usize - 1] > 0 {
                total_player_2 += points[point as usize - 1];
            }
            if point == 6 {
                print!("|");
            }
            print!("{:3} ", points[point as usize - 1]);
        }
        println!("\n");

        // Display hit pieces and collected pieces
        println!(
            "Hit: ({}, {})   Collected: ({}, {})",
            pieces_hit.0, pieces_hit.1, pieces_collected.0, pieces_collected.1
        );

        total_player_1 += pieces_hit.0 as i8;
        total_player_1 += pieces_collected.0 as i8;
        total_player_2 += pieces_hit.1 as i8;
        total_player_2 += pieces_collected.1 as i8;
        println!("Total pieces: ({}, {})", total_player_1, total_player_2);
        println!("\n");
    }

    /**
     * Checks if current board is valid by asserting that 
     * both players have 15 pieces on the board including collected and barred ones.
     */
    pub fn is_valid(&mut self) {
        let (points, pieces_hit, pieces_collected) = self.board;
        let mut total_player_1 = 0;
        let mut total_player_2 = 0;

        for point in 0..=23 {
            if points[point as usize] < 0 {
                total_player_1 -= points[point as usize];
            } else if points[point as usize] > 0 {
                total_player_2 += points[point as usize];
            }
        }

        total_player_1 += pieces_hit.0 as i8;
        total_player_1 += pieces_collected.0 as i8;
        total_player_2 += pieces_hit.1 as i8;
        total_player_2 += pieces_collected.1 as i8;
        println!("Total pieces: ({}, {})", total_player_1, total_player_2);
        assert!(total_player_1 == 15 && total_player_2 == 15)
    }

    pub fn get_initial_state() -> Board {
        (
            [
                2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2,
            ],
            (0, 0),
            (0, 0),
        )
    }

    /**
     * Returns the next state of the board, given the actions and the player that is playing it.
     */
    pub fn get_next_state(mut state: Board, actions: &Actions, player: i8) -> Board {
        // loop over actions
        for &(from, to) in actions.iter() {
            if to == -1 {
                // Player is collecting a checker
                state.0[from as usize] -= player;

                if player == -1 {
                    state.2 .0 += 1;
                } else {
                    state.2 .1 += 1;
                }
                continue;
            }

            if from == -1 {
                // Playing from the bar
                if state.0[to as usize] == -player {
                    // Hitting the opponent's checker
                    state.0[to as usize] = player;
                    if player == -1 {
                        state.1 .1 += 1;
                        state.1 .0 -= 1;
                    } else {
                        state.1 .0 += 1;
                        state.1 .1 -= 1;
                    }
                } else if player == -1 {
                    state.0[to as usize] -= 1;
                    state.1 .0 -= 1;
                } else {
                    state.0[to as usize] += 1;
                    state.1 .1 -= 1;
                }
            } else if state.0[to as usize] == -player {
                // Hitting the opponent's checker
                state.0[to as usize] = player;
                state.0[from as usize] -= player;
                if player == -1 {
                    state.1 .1 += 1;
                } else {
                    state.1 .0 += 1;
                }
            } else {
                // Moving a checker from one position to another
                state.0[to as usize] += player;
                state.0[from as usize] -= player;
            }
        }
        state
    }

    pub fn check_win(state: Board, player: i8) -> bool {
        if player == -1 {
            state.2 .0 == 15
        } else {
            state.2 .1 == 15
        }
    }

    pub fn check_win_without_player(state: Board) -> Option<i8> {
        if state.2 .0 == 15 {
            return Some(-1);
        } else if state.2 .1 == 15 {
            return Some(1);
        }
        None
    }

    fn get_pieces_hit(state: Board, player: i8) -> u8 {
        if player == -1 {
            state.1 .0
        } else {
            state.1 .1
        }
    }

    // pub fn get_valid_moves(&self) -> Vec<Actions> {
    //     assert!(self.roll != (0, 0), "die has not been rolled!");

    //     let all_moves: Vec<u8> = match self.roll {
    //         (r0, r1) if r0 == r1 => vec![r0; 4],
    //         (r0, r1) if r0 > r1 => vec![r0, r1],
    //         (r0, r1) => vec![r1, r0],
    //     };
    //     let action_trees = Self::_get_action_trees(&all_moves, self.board, self.player);
    //     // parse trees into actions here
    //     let actions = Self::extract_sequences_list(action_trees);
    //     Self::remove_duplicate_states(self.board, actions, self.player)
    // }


    fn _get_action_trees(moves: &[u8], state: Board, player: i8) -> Vec<ActionNode> {
        let num_pieces_hit = Self::get_pieces_hit(state, player);
        // Return if there is a hit piece, no other move can be made without all pieces in-game
        if num_pieces_hit > 0 {
            Self::get_entry_moves(moves, state, player)
        } else {
            Self::get_normal_moves(moves, state, player)
        }
    }

    // gets all 'normal' moves, i.e. moves that are not from the bar (collection moves included)
    pub fn get_normal_moves(moves: &[u8], state: Board, player: i8) -> Vec<ActionNode> {
        let (board, _, _) = state;

        let mut trees: Vec<ActionNode> = vec![];
        let mut possible_actions: Vec<(i8, (i8, i8))> = vec![];

        // Players still can do normal moves even in collection so we do not only return collection moves
        if player == -1 && Self::is_collectible(state, player) {
            for m in moves.iter().map(|x| *x as i8) {
                let point = m - 1;
                let n_pieces_on_point = board[point as usize];
                if n_pieces_on_point < 0 {
                    possible_actions.push((m, (point, -1)))
                }
                // Add points that are smaller than the current move
                // Eg 6 can collect 5, if there is no pieces on the 6th point and there is a piece on 5
                for m_idx in (0..point).rev() {
                    let m_idx_usize = m_idx as usize;
                    let n_pieces_on_point = board[m_idx_usize];
                    let left_sum: i8 = board[m_idx_usize + 1..6].iter().sum();
                    if n_pieces_on_point < 0 && left_sum >= 0 {
                        possible_actions.push((m, (m_idx, -1)));
                        break;
                    }
                }
            }
        } else if player == 1 && Self::is_collectible(state, player) {
            for m in moves.iter().map(|x| *x as i8) {
                let point = 24 - m;
                let n_pieces_on_point = board[point as usize];
                if n_pieces_on_point > 0 {
                    possible_actions.push((m, (point, -1)))
                }
                for m_idx in point..=23 {
                    let m_idx_usize = m_idx as usize;
                    let n_pieces_on_point = board[m_idx_usize];
                    let left_sum: i8 = board[18..m_idx_usize].iter().sum();
                    if n_pieces_on_point > 0 && left_sum <= 0 {
                        possible_actions.push((m, (m_idx, -1)));
                        break;
                    }
                }
            }
        }

        for m in moves.iter().map(|x| *x as i8) {
            for (point, n_pieces) in board.iter().enumerate().map(|x| (x.0 as i8, x.1)) {
                // player == -1 && *n_pieces < 0, check if player -1 is going
                if player == -1
                    && *n_pieces <= player
                    && point - m >= 0
                    && board[(point - m) as usize] <= 1
                {
                    possible_actions.push((m, (point, (point - m))))
                } else if player == 1
                    && *n_pieces >= player
                    && point + m <= 23
                    && board[(point + m) as usize] >= -1
                {
                    possible_actions.push((m, (point, (point + m))))
                }
            }
        }
        // removes duplicate actions
        possible_actions.sort_unstable();
        possible_actions.dedup();

        for action in possible_actions {
            let current_node = ActionNode {
                value: action.1,
                children: Self::_get_children_of_node_action(
                    moves.to_vec(),
                    action.0 as u8,
                    action.1,
                    state,
                    player,
                ),
            };
            trees.push(current_node)
        }
        trees
    }

    pub fn is_collectible(board: Board, player: i8) -> bool {
        if player == -1 {
            if board.1 .0 != 0 {
                return false;
            }
            for i in 6..=23 {
                if board.0[i] < 0 {
                    return false;
                }
            }
        } else if player == 1 {
            if board.1 .1 != 0 {
                return false;
            }
            for i in 0..=17 {
                if board.0[i] > 0 {
                    return false;
                }
            }
        }
        true
    }

    // returns all 'entry' moves, i.e. moves from the bar
    pub fn get_entry_moves(moves: &[u8], state: Board, player: i8) -> Vec<ActionNode> {
        let (board, _, _) = state;

        let mut trees: Vec<ActionNode> = vec![];
        let mut possible_actions: Vec<(i8, (i8, i8))> = vec![];

        if player == -1 {
            for m in moves.iter().map(|x| *x as i8) {
                let point = 24 - m;
                if board[point as usize] < 2 {
                    possible_actions.push((m, (-1, point)));
                }
            }
        } else if player == 1 {
            for m in moves.iter().map(|x| *x as i8) {
                let point = m - 1;
                if board[point as usize] > -2 {
                    possible_actions.push((m, (-1, point)));
                }
            }
        }

        // removes duplicate actions
        possible_actions.sort_unstable();
        possible_actions.dedup();

        for action in possible_actions {
            let current_node = ActionNode {
                value: action.1,
                children: Self::_get_children_of_node_action(
                    moves.to_vec(),
                    action.0 as u8,
                    action.1,
                    state,
                    player,
                ),
            };
            trees.push(current_node)
        }

        trees
    }

    fn _get_children_of_node_action(
        moves: Vec<u8>,
        move_used: u8,
        action: (i8, i8),
        state: Board,
        player: i8,
    ) -> Vec<ActionNode> {
        let new_state = Self::get_next_state(state, &vec![action], player);
        // Remove the move played from all moves
        let mut new_moves = moves.to_vec();
        let index_to_remove = new_moves.iter().position(|&m| m == move_used).unwrap();
        new_moves.remove(index_to_remove);

        // return children
        Self::_get_action_trees(&new_moves, new_state, player)
    }

    pub fn extract_sequences_list(list: Vec<ActionNode>) -> Vec<Actions> {
        let mut sequences: Vec<Actions> = vec![];
        for action_node in list {
            sequences.extend(Self::extract_sequences_node(&action_node));
        }
        sequences
    }

    pub fn extract_sequences_node(node: &ActionNode) -> Vec<Actions> {
        Self::extract_sequences_helper(node, Vec::new())
    }

    fn extract_sequences_helper(node: &ActionNode, current_sequence: Actions) -> Vec<Actions> {
        let mut sequences = Vec::new();

        let mut new_sequence = current_sequence.clone();
        new_sequence.push(node.value);

        if node.children.is_empty() {
            sequences.push(new_sequence);
        } else {
            for child in &node.children {
                let child_sequences = Self::extract_sequences_helper(child, new_sequence.clone());
                sequences.extend(child_sequences);
            }
        }

        sequences
    }

    // applies the actions to the initial state and removes an action if an action before leads to the same state
    pub fn remove_duplicate_states(
        initial_state: Board,
        sequences: Vec<Actions>,
        player: i8,
    ) -> Vec<Actions> {
        let mut seen_states = HashSet::new();
        let mut unique_sequences = Vec::new();

        for sequence in sequences {
            let mut current_state = initial_state;

            for action in &sequence {
                current_state = Self::get_next_state(current_state, &vec![*action], player);
            }

            if seen_states.insert(current_state) {
                unique_sequences.push(sequence);
            }
        }

        unique_sequences
    }
}
