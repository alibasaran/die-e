type Board = ([i8; 24], (u8, u8), (u8, u8));
type Actions = Vec<(i8, i8)>;

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
            board: ([2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2], (0, 0), (0, 0))
        }
    }

    pub fn get_initial_state() -> Board {
        ([2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2], (0, 0), (0, 0))
    }

    // for the player field, -1 or 1 is used to indicate which player's move it is
    pub fn get_next_state(mut state: Board, actions: Actions, player: i8) -> Board {
        for &(from, to) in actions.iter() {
            if to == -1 {
                // Player is bearing off a checker
                state.0[from as usize] -= player;
                
                if player == -1 {
                    state.2.0 += 1;
                } else {
                    state.2.1 += 1;
                }
                continue;
            }

            if from == -1 {
                if state.0[to as usize] == player * -1 {
                    // Hitting the opponent's checker
                    state.0[to as usize] = player;
                    if player == -1 {
                        state.1.1 += 1;
                    } else {
                        state.1.0 += 1;
                    }
                }
                // Moving a checker from the bar
                state.0[to as usize] = player;
                if player == -1 {
                    state.1.0 -= 1;
                } else {
                    state.1.1 -= 1;
                }
            } else {
                if state.0[to as usize] == player * -1 {
                    // Hitting the opponent's checker
                    state.0[to as usize] = player;
                    state.0[from as usize] -= player;
                    if player == -1 {
                        state.1.1 += 1;
                    } else {
                        state.1.0 += 1;
                    }
                } else {
                    // Moving a checker from one position to another
                    state.0[to as usize] += player;
                    state.0[from as usize] -= player;
                }
            }
        }
        state
    }
    // get_next_state(state, action, player) // Sinan
    // get_valid_moves(die, state) // Ali
    // check_win // Ali
}

