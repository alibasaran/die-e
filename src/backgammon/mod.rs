use std::{cmp, fmt, ops::Range, vec};
// (the board itself, pieces_hit, pieces_collected)
type Board = ([i8; 24], (u8, u8), (u8, u8));
// (from, to) if to == -1 then it is collection, if from == -1 then it is putting a hit piece back
type Actions = Vec<(i8, i8)>;

pub struct ActionNode {
    value: (i8, i8),
    children: Vec<ActionNode>,
}

impl fmt::Display for ActionNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format_tree(f, "", true)
    }
}

impl ActionNode {
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

pub struct Backgammon {
    // board:: 24 poz
    pub board: Board,
    // 15'er tas
    // kirik taslar [beyaz, siyah]
    // toplanmis taslar [beyaz, siyah]
    // hamle sirasi
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
        }
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
    pub fn get_next_state(mut state: Board, actions: Actions, player: i8) -> Board {
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
                if state.0[to as usize] == player * -1 {
                    // Hitting the opponent's checker
                    state.0[to as usize] = player;
                    if player == -1 {
                        state.1 .1 += 1;
                    } else {
                        state.1 .0 += 1;
                    }
                }
                // Moving a checker from the bar
                state.0[to as usize] = player;
                if player == -1 {
                    state.1 .0 -= 1;
                } else {
                    state.1 .1 -= 1;
                }
            } else if state.0[to as usize] == player * -1 {
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
    fn check_win(state: Board, action: Actions, player: i8) -> bool {
        let board: Board = Self::get_next_state(state, action, player);
        if player == -1 {
            return board.1 .0 == 15;
        } else {
            return board.1 .1 == 15;
        }
    }

    fn get_pieces_hit(state: Board, player: i8) -> u8 {
        if player == -1 {
            return state.1 .1;
        } else {
            return state.1 .0;
        }
    }

    fn get_valid_moves(roll: (u8, u8), state: Board, player: i8) -> Vec<Actions> {
        let mut all_moves: Vec<u8> = if roll.0 == roll.1 {
            vec![roll.0; 4]
        // To remove all dup combinations. Eg. 6-1 or 1-6
        } else if roll.0 > roll.1 {
            vec![roll.0, roll.1]
        } else {
            vec![roll.1, roll.0]
        };
        let action_trees = Self::_get_action_trees(&mut all_moves, state, player);
        // parse trees into actions here
        return vec![];
    }

    fn _get_action_trees(moves: &Vec<u8>, state: Board, player: i8) -> Vec<ActionNode> {
        let num_pieces_hit = Self::get_pieces_hit(state, player);
        // Return if there is a hit piece, no other move can be made without all pieces in-game
        if num_pieces_hit > 0 {
            return Self::get_entry_moves(moves, state, player);
        } else {
            return Self::get_normal_moves(moves, state, player);
        }
    }

    pub fn get_normal_moves(moves: &Vec<u8>, state: Board, player: i8) -> Vec<ActionNode> {
        let (board, _, _) = state;
        let num_pieces_hit = Self::get_pieces_hit(state, player);

        let mut trees: Vec<ActionNode> = vec![];
        let mut possible_actions: Vec<(i8, (i8, i8))> = vec![];

        // Players still can do normal moves even in collection so we do not only return collection moves
        if player == -1 && num_pieces_hit == 0 && board[6..18].iter().sum::<i8>() == 0 {
            for m in moves.iter().map(|x| *x as i8) {
                let point = (m - 1) as i8;
                let n_pieces_on_point = board[point as usize];
                if n_pieces_on_point < 0 {
                    possible_actions.push((m, (point, -1)))
                }
            }
        } else if player == 1 && num_pieces_hit == 0 && board[0..19].iter().sum::<i8>() == 0 {
            for m in moves.iter().map(|x| *x as i8) {
                let point = (24 - m) as i8;
                let n_pieces_on_point = board[point as usize];
                if n_pieces_on_point < 0 {
                    possible_actions.push((m, (point, -1)))
                }
            }
        }

        for m in moves.iter().map(|x| *x as i8) {
            for (point, n_pieces) in board.iter().enumerate().map(|x| (x.0 as i8, x.1)) {
                // player == -1 && *n_pieces < 0, check if player -1 is going
                if *n_pieces <= player && point - m >= 0 && board[(point - m) as usize] <= 1 {
                    possible_actions.push((m, (point as i8, (point - m) as i8)))
                } else if *n_pieces >= player
                    && point + m <= 23
                    && board[(point + m) as usize] >= -1
                {
                    possible_actions.push((m, (point as i8, (point + m) as i8)))
                }
            }
        }
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
        return trees;
    }


    pub fn get_entry_moves(moves: &Vec<u8>, state: Board, player: i8) -> Vec<ActionNode> {
        if moves.is_empty() {
            return vec![];
        }
        
        let (board, _, _) = state;

        let mut trees: Vec<ActionNode> = vec![];

        let mut entry_moves: Actions = vec![];
        if player == -1 {
            let range: Range<u8> = 0..6;
            for i in range {
                if board[23 - usize::from(i)] < 2 && (moves[0] == i || moves[1] == i) {
                    entry_moves.push((-1, 23 - i as i8))
                }
            }
        } else {
            let range: Range<u8> = 0..6;
            for i in range {
                if board[usize::from(i)] < 2 && (moves[0] == i || moves[1] == i) {
                    entry_moves.push((-1, i as i8))
                }
            }
        }

        for entry_move in entry_moves.iter() {
            let current_node = ActionNode {
                value: *entry_move,
                children: {
                    let new_state = Self::get_next_state(state, vec![*entry_move], player);

                    // Remove the move played from all moves
                    let mut new_moves = moves.to_vec();
                    let index_to_remove = new_moves
                        .iter()
                        .position(|&m| m == entry_move.1 as u8)
                        .unwrap();
                    new_moves.remove(index_to_remove);

                    // return children
                    Self::_get_action_trees(&mut new_moves, new_state, player)
                },
            };
            trees.push(current_node)
        }
        return trees;
    }

    fn _get_children_of_node_action(
        moves: Vec<u8>,
        move_used: u8,
        action: (i8, i8),
        state: Board,
        player: i8,
    ) -> Vec<ActionNode> {
        let new_state = Self::get_next_state(state, vec![action], player);

        // Remove the move played from all moves
        let mut new_moves = moves.to_vec();
        let index_to_remove = new_moves.iter().position(|&m| m == move_used).unwrap();
        new_moves.remove(index_to_remove);

        // return children
        return Self::_get_action_trees(&mut new_moves, new_state, player);
    }
}
