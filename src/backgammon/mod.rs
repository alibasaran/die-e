use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fmt, vec};
use tch::Tensor;

pub mod encoding;

use crate::constants::{DEVICE, DEFAULT_TYPE};

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

impl Backgammon {
    pub fn new() -> Self {
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

    pub fn init_with_fields(board: Board, player: i8, is_second_play: bool) -> Self {
        Backgammon {
            board,
            roll: (0, 0),
            player,
            is_second_play,
            id: 0
        }
    }

    pub fn roll_die(&mut self) -> (u8, u8) {
        let mut rng = rand::thread_rng();
        self.roll = (rng.gen_range(1..=6), rng.gen_range(1..=6));
        self.roll
    }

    pub fn apply_move(&mut self, actions: &Actions) {
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

    pub fn skip_turn(&mut self) {
        self.is_second_play = false;
        self.player *= -1;
        self.roll_die();
    }

    pub fn as_tensor(&self) -> Tensor {
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

    pub fn get_initial_state() -> Board {
        (
            [
                2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2,
            ],
            (0, 0),
            (0, 0),
        )
    }

    // for the player field, -1 or 1 is used to indicate which player's move it is
    pub fn get_next_state(mut state: Board, actions: &Actions, player: i8) -> Board {
        for &(from, to) in actions.iter() {
            if to == -1 {
                // Player is bearing off a checker
                state.0[from as usize] -= player;

                if player == -1 {
                    state.2 .0 += 1;
                } else {
                    state.2 .1 += 1;
                }
                continue;
            }

            if from == -1 {
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
    // get_next_state(state, action, player) // Sinan
    // get_valid_moves(die, state) // Ali
    // check_win // Ali
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

    pub fn get_valid_moves(&self) -> Vec<Actions> {
        assert!(self.roll != (0, 0), "die has not been rolled!");

        let all_moves: Vec<u8> = match self.roll {
            (r0, r1) if r0 == r1 => vec![r0; 4],
            (r0, r1) if r0 > r1 => vec![r0, r1],
            (r0, r1) => vec![r1, r0],
        };
        let action_trees = Self::_get_action_trees(&all_moves, self.board, self.player);
        // parse trees into actions here
        let actions = Self::extract_sequences_list(action_trees);
        Self::remove_duplicate_states(self.board, actions, self.player)
    }

    pub fn get_valid_moves_len_always_2(&self) -> Vec<Actions> {
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

    fn _get_action_trees(moves: &[u8], state: Board, player: i8) -> Vec<ActionNode> {
        let num_pieces_hit = Self::get_pieces_hit(state, player);
        // Return if there is a hit piece, no other move can be made without all pieces in-game
        if num_pieces_hit > 0 {
            Self::get_entry_moves(moves, state, player)
        } else {
            Self::get_normal_moves(moves, state, player)
        }
    }

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
