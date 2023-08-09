use die_e::backgammon::Backgammon;
use std::vec;

#[cfg(test)]
mod get_initial_state {
    use super::*;
    #[test]
    fn get_initial_state_test() {
        let bg = Backgammon::new();
        let init = Backgammon::get_initial_state();
        let expected = (
            [
                2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2,
            ],
            (0, 0),
            (0, 0),
        );
        assert_eq!(bg.board, init);
        assert_eq!(init, expected);
    }
}

#[cfg(test)]
mod get_next_state {
    use super::*;
    #[test]
    fn it_should_not_change_state_when_actions_empty() {
        let state = Backgammon::get_initial_state();
        let actions: Vec<(i8, i8)> = vec![];
        let expected = (
            [
                2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2,
            ],
            (0, 0),
            (0, 0),
        );
        assert_eq!(Backgammon::get_next_state(state, actions, 1), expected);
    }

    #[test]
    fn it_should_change_state_on_normal_move1() {
        let state = Backgammon::get_initial_state();
        let actions: Vec<(i8, i8)> = vec![(23, 21), (23, 20)];
        let expected = (
            [
                2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, -1, -1, 0, 0,
            ],
            (0, 0),
            (0, 0),
        );
        assert_eq!(Backgammon::get_next_state(state, actions, -1), expected);
    }

    #[test]
    fn it_should_change_state_on_normal_move2() {
        let state = Backgammon::get_initial_state();
        let actions: Vec<(i8, i8)> = vec![(0, 3), (0, 3)];
        let expected = (
            [
                0, 0, 0, 2, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2,
            ],
            (0, 0),
            (0, 0),
        );
        assert_eq!(Backgammon::get_next_state(state, actions, 1), expected);
    }

    #[test]
    fn it_should_change_state_when_player1_hit() {
        let state = (
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, 0, 0, 1, 0, 1,
            ],
            (0, 0),
            (0, 0),
        );
        let actions: Vec<(i8, i8)> = vec![(18, 21), (18, 23)];
        let expected = (
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, -1, 0, -1,
            ],
            (0, 2),
            (0, 0),
        );
        assert_eq!(Backgammon::get_next_state(state, actions, -1), expected);
    }

    #[test]
    fn it_should_change_state_when_player2_hit() {
        let state = (
            [
                0, 0, 0, 0, 0, 5, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            (0, 0),
            (0, 0),
        );
        let actions: Vec<(i8, i8)> = vec![(5, 8), (5, 10)];
        let expected = (
            [
                0, 0, 0, 0, 0, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            (2, 0),
            (0, 0),
        );
        assert_eq!(Backgammon::get_next_state(state, actions, 1), expected);
    }

    #[test]
    fn it_should_change_state_when_collecting_player1() {
        let state = (
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2,
            ],
            (0, 0),
            (0, 0),
        );
        let actions: Vec<(i8, i8)> = vec![(23, -1), (23, -1)];
        let expected = (
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            (0, 0),
            (2, 0),
        );
        assert_eq!(Backgammon::get_next_state(state, actions, -1), expected);
    }

    #[test]
    fn it_should_change_state_when_collecting_player2() {
        let state = (
            [
                0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            (0, 0),
            (0, 0),
        );
        let actions: Vec<(i8, i8)> = vec![(5, -1), (5, -1)];
        let expected = (
            [
                0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            (0, 0),
            (0, 2),
        );
        assert_eq!(Backgammon::get_next_state(state, actions, 1), expected);
    }

    #[test]
    fn it_should_change_state_playing_from_bar_player1() {
        let state = (
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            (2, 0),
            (0, 0),
        );
        let actions: Vec<(i8, i8)> = vec![(-1, 23), (-1, 20)];
        let expected = (
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1,
            ],
            (0, 0),
            (0, 0),
        );
        assert_eq!(Backgammon::get_next_state(state, actions, -1), expected);
    }

    #[test]
    fn it_should_change_state_playing_from_bar_player2() {
        let state = (
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            (0, 2),
            (0, 0),
        );
        let actions: Vec<(i8, i8)> = vec![(-1, 0), (-1, 3)];
        let expected = (
            [
                1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            (0, 0),
            (0, 0),
        );
        assert_eq!(Backgammon::get_next_state(state, actions, 1), expected);
    }

    #[test]
    fn it_should_change_state_playing_from_bar_and_hitting_player1() {
        let state = (
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
            ],
            (2, 0),
            (0, 0),
        );
        let actions: Vec<(i8, i8)> = vec![(-1, 23), (-1, 20)];
        let expected = (
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1,
            ],
            (0, 2),
            (0, 0),
        );
        assert_eq!(Backgammon::get_next_state(state, actions, -1), expected);
    }

    #[test]
    fn it_should_change_state_playing_from_bar_and_hitting_player2() {
        let state = (
            [
                -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            (0, 2),
            (0, 0),
        );
        let actions: Vec<(i8, i8)> = vec![(-1, 0), (-1, 3)];
        let expected = (
            [
                1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            (2, 0),
            (0, 0),
        );
        assert_eq!(Backgammon::get_next_state(state, actions, 1), expected);
    }
}

mod get_normal_moves {
    use die_e::backgammon::ActionNode;

    use super::*;

    mod player1 {
        use super::*;

        #[test]
        fn it_should_work_for_single_move() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.0[20] = -1;
            let moves = Backgammon::get_normal_moves(&vec![1], state, -1);
            assert_eq!(moves[0].value, (20, 19));
        }

        #[test]
        fn it_should_return_empty_vec_if_there_are_no_moves() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.0[20] = -1;
            state.0[19] = 2;
            let moves = Backgammon::get_normal_moves(&vec![1], state, -1);
            assert!(moves.is_empty())
        }
        #[test]
        fn it_should_work_for_multiple_moves() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.0[20] = -1;
            let moves = Backgammon::get_normal_moves(&vec![1, 1], state, -1);
            let tree = &moves[0];
            assert_eq!(tree.value, (20, 19));
            assert_eq!(tree.children[0].value, (19, 18));
        }

        #[test]
        fn it_should_work_for_multiple_moves_multiple_possibilities() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.0[20] = -1;
            let moves = Backgammon::get_normal_moves(&vec![2, 1], state, -1);
            let tree1 = ActionNode {
                value: (20, 19),
                children: vec![ActionNode {
                    value: (19, 17),
                    children: vec![],
                }],
            };
            let tree2 = ActionNode {
                value: (20, 18),
                children: vec![ActionNode {
                    value: (18, 17),
                    children: vec![],
                }],
            };
            assert!(moves.len() == 2);
            assert!(moves.contains(&tree1));
            assert!(moves.contains(&tree2));
        }

        #[test]
        fn it_should_return_collect_moves_when_collectible_exact_move() {
            let mut state = Backgammon::get_initial_state();
            let mut board = [0; 24];
            board[0] = -1;
            state.0 = board;

            let moves = Backgammon::get_normal_moves(&vec![1], state, -1);
            let expected = ActionNode {
                value: (0, -1),
                children: vec![],
            };
            assert_eq!(moves[0], expected);
        }

        #[test]
        fn it_should_return_collect_moves_when_collectible_higher_move() {
            let mut state = Backgammon::get_initial_state();
            let mut board = [0; 24];
            board[0] = -1;
            state.0 = board;

            let moves = Backgammon::get_normal_moves(&vec![6], state, -1);
            let expected = ActionNode {
                value: (0, -1),
                children: vec![],
            };
            assert_eq!(moves[0], expected);
        }

        #[test]
        fn it_should_return_collect_moves_when_collectible_exact_moves() {
            let mut state = Backgammon::get_initial_state();
            let mut board = [0; 24];
            board[0] = -1;
            board[5] = -1;
            state.0 = board;

            let moves = Backgammon::get_normal_moves(&vec![6, 1], state, -1);
            let expected = ActionNode {
                value: (0, -1),
                children: vec![ActionNode {
                    value: (5, -1),
                    children: vec![],
                }],
            };
            // .contains because another possibility is the other way around
            assert!(moves.contains(&expected));
        }

        #[test]
        fn it_should_return_collect_moves_when_collectible_higher_moves() {
            let mut state = Backgammon::get_initial_state();
            let mut board = [0; 24];
            board[0] = -1;
            board[4] = -1;
            state.0 = board;

            let moves = Backgammon::get_normal_moves(&vec![6, 1], state, -1);
            let expected = ActionNode {
                value: (4, -1),
                children: vec![ActionNode {
                    value: (0, -1),
                    children: vec![],
                }],
            };
            // .contains because another possibility is the other way around
            assert!(moves.contains(&expected));
        }
    }

    mod player2 {
        use super::*;

        #[test]
        fn it_should_work_for_single_move() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.0[10] = 1;
            let moves = Backgammon::get_normal_moves(&vec![1], state, 1);
            assert_eq!(moves[0].value, (10, 11));
        }

        #[test]
        fn it_should_return_empty_vec_if_there_are_no_moves() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.0[10] = 1;
            state.0[11] = -2;
            let moves = Backgammon::get_normal_moves(&vec![1], state, 1);
            assert!(moves.is_empty())
        }
        #[test]
        fn it_should_work_for_multiple_moves() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.0[10] = 1;
            let moves = Backgammon::get_normal_moves(&vec![1, 1], state, 1);
            let tree = &moves[0];
            assert_eq!(tree.value, (10, 11));
            assert_eq!(tree.children[0].value, (11, 12));
        }

        #[test]
        fn it_should_work_for_multiple_moves_multiple_possibilities() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.0[10] = 1;
            let moves = Backgammon::get_normal_moves(&vec![2, 1], state, 1);
            let tree1 = ActionNode {
                value: (10, 11),
                children: vec![ActionNode {
                    value: (11, 13),
                    children: vec![],
                }],
            };
            let tree2 = ActionNode {
                value: (10, 12),
                children: vec![ActionNode {
                    value: (12, 13),
                    children: vec![],
                }],
            };
            assert!(moves.len() == 2);
            assert!(moves.contains(&tree1));
            assert!(moves.contains(&tree2));
        }

        #[test]
        fn it_should_return_collect_moves_when_collectible_exact_move() {
            let mut state = Backgammon::get_initial_state();
            let mut board = [0; 24];
            board[23] = 1;
            state.0 = board;

            let moves = Backgammon::get_normal_moves(&vec![1], state, 1);
            let expected = ActionNode {
                value: (23, -1),
                children: vec![],
            };
            assert_eq!(moves[0], expected);
        }

        #[test]
        fn it_should_return_collect_moves_when_collectible_higher_move() {
            let mut state = Backgammon::get_initial_state();
            let mut board = [0; 24];
            board[23] = 1;
            state.0 = board;

            let trees = Backgammon::get_normal_moves(&vec![6], state, 1);
            let expected = ActionNode {
                value: (23, -1),
                children: vec![],
            };
            assert!(!trees.is_empty());
            assert_eq!(trees[0], expected);
        }

        #[test]
        fn it_should_return_collect_moves_when_collectible_exact_moves() {
            let mut state = Backgammon::get_initial_state();
            let mut board = [0; 24];
            board[23] = 1;
            board[18] = 1;
            state.0 = board;

            let moves = Backgammon::get_normal_moves(&vec![6, 1], state, 1);
            let expected = ActionNode {
                value: (23, -1),
                children: vec![ActionNode {
                    value: (18, -1),
                    children: vec![],
                }],
            };
            // .contains because another possibility is the other way around
            assert!(moves.contains(&expected));
        }

        #[test]
        fn it_should_return_collect_moves_when_collectible_higher_moves() {
            let mut state = Backgammon::get_initial_state();
            let mut board = [0; 24];
            board[23] = 1;
            board[20] = 1;
            state.0 = board;

            let moves = Backgammon::get_normal_moves(&vec![6, 1], state, 1);
            let expected = ActionNode {
                value: (20, -1),
                children: vec![ActionNode {
                    value: (23, -1),
                    children: vec![],
                }],
            };
            // .contains because another possibility is the other way around
            assert!(moves.contains(&expected));
        }
    }
}

mod is_collectible {
    use super::*;

    #[test]
    fn returns_false_when_not_collectible() {
        let state = Backgammon::get_initial_state();
        let collectable_p1 = Backgammon::is_collectible(state, -1);
        let collectable_p2 = Backgammon::is_collectible(state, 1);
        assert!(!collectable_p1);
        assert!(!collectable_p2);
    }

    #[test]
    fn returns_true_when_collectable() {
        let mut state = Backgammon::get_initial_state();
        let mut board = [0; 24];
        board[0] = -1;
        board[23] = 1;
        state.0 = board;

        let collectable_p1 = Backgammon::is_collectible(state, -1);
        let collectable_p2 = Backgammon::is_collectible(state, 1);
        assert!(collectable_p1);
        assert!(collectable_p2);
    }

    #[test]
    fn returns_false_when_piece_on_bar() {
        let mut state = Backgammon::get_initial_state();
        let mut board = [0; 24];
        board[0] = -1;
        board[23] = 1;
        state.0 = board;
        state.1 = (1, 1);
        let collectable_p1 = Backgammon::is_collectible(state, -1);
        let collectable_p2 = Backgammon::is_collectible(state, 1);
        assert!(!collectable_p1);
        assert!(!collectable_p2);
    }
}

mod check_win {
    use super::*;

    #[test]
    fn it_should_return_false_when_nobody_won() {
        let bg = Backgammon::get_initial_state();
        let win_p1 = Backgammon::check_win(bg, -1);
        let win_p2 = Backgammon::check_win(bg, 1);
        assert!(!win_p1);
        assert!(!win_p2);
    }

    #[test]
    fn it_should_return_true_when_win() {
        let mut bg = Backgammon::get_initial_state();
        bg.0 = [0; 24];
        bg.2 .0 = 15; // Total num of pieces
        bg.2 .1 = 15; // Total num of pieces
        let win_p1 = Backgammon::check_win(bg, -1);
        let win_p2 = Backgammon::check_win(bg, 1);
        assert!(win_p1);
        assert!(win_p2);
    }
}
#[cfg(test)]
mod extract_sequences_node {
    use super::*;

    #[test]
    fn it_should_work_for_single_move() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[10] = 1;
        let moves = Backgammon::get_normal_moves(&vec![1], state, 1);
        let expected: Vec<Vec<(i8, i8)>> = vec![vec![(10, 11)]];
        assert_eq!(
            Backgammon::extract_sequences_node(moves.get(0).unwrap()),
            expected
        );
    }

    #[test]
    fn it_should_work_for_multiple_moves() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[20] = -1;
        let moves = Backgammon::get_normal_moves(&vec![1, 1], state, -1);
        let tree = &moves[0];
        let expected: Vec<Vec<(i8, i8)>> = vec![vec![(20, 19), (19, 18)]];
        assert_eq!(Backgammon::extract_sequences_node(tree), expected);
    }

    #[test]
    fn it_should_work_for_multiple_moves_multiple_possibilities() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[20] = -1;
        state.0[23] = -1;
        let moves = Backgammon::get_normal_moves(&vec![1, 1], state, -1);
        let expected: Vec<Vec<(i8, i8)>> = vec![vec![(20, 19), (19, 18)], vec![(20, 19), (23, 22)]];
        assert_eq!(
            Backgammon::extract_sequences_node(moves.get(0).unwrap()),
            expected
        );
    }
}

#[cfg(test)]
mod extract_sequences_list {
    use super::*;

    #[test]
    fn it_should_work_for_single_move() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[10] = 1;
        let moves = Backgammon::get_normal_moves(&vec![1], state, 1);
        let expected: Vec<Vec<(i8, i8)>> = vec![vec![(10, 11)]];
        assert_eq!(Backgammon::extract_sequences_list(moves), expected);
    }

    #[test]
    fn it_should_work_for_multiple_moves() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[20] = -1;
        let moves = Backgammon::get_normal_moves(&vec![1, 1], state, -1);
        let expected: Vec<Vec<(i8, i8)>> = vec![vec![(20, 19), (19, 18)]];
        assert_eq!(Backgammon::extract_sequences_list(moves), expected);
    }

    #[test]
    fn it_should_work_for_multiple_moves_multiple_possibilities() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[20] = -1;
        let moves = Backgammon::get_normal_moves(&vec![2, 1], state, -1);
        let expected: Vec<Vec<(i8, i8)>> = vec![vec![(20, 19), (19, 17)], vec![(20, 18), (18, 17)]];
        assert_eq!(Backgammon::extract_sequences_list(moves), expected);
    }

    #[test]
    fn it_should_work_for_multiple_action_nodes() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[20] = -1;
        state.0[19] = 2;
        state.0[16] = -1;
        let moves = Backgammon::get_normal_moves(&vec![2, 1], state, -1);
        let expected: Vec<Vec<(i8, i8)>> = vec![
            vec![(16, 15), (15, 13)],
            vec![(16, 15), (20, 18)],
            vec![(16, 14), (14, 13)],
            vec![(20, 18), (16, 15)],
            vec![(20, 18), (18, 17)],
        ];
        assert!(moves.len() > 1);
        assert_eq!(Backgammon::extract_sequences_list(moves), expected);
    }
}

#[cfg(test)]
mod remove_duplicate_states {
    use super::*;

    #[test]
    fn it_should_not_change_single_sequence_single_move() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[20] = -1;
        let sequences: Vec<Vec<(i8, i8)>> = vec![vec![(20, 19)]];
        let expected = sequences.clone();
        assert_eq!(
            Backgammon::remove_duplicate_states(state, sequences, -1),
            expected
        );
    }

    #[test]
    fn it_should_not_change_single_sequence_multiple_moves() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[20] = -1;
        let sequences: Vec<Vec<(i8, i8)>> = vec![vec![(20, 19), (19, 18)]];
        let expected = sequences.clone();
        assert_eq!(
            Backgammon::remove_duplicate_states(state, sequences, -1),
            expected
        );
    }

    #[test]
    fn it_should_remove_duplicate_move() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[20] = -1;
        let sequences: Vec<Vec<(i8, i8)>> =
            vec![vec![(20, 19), (19, 17)], vec![(20, 18), (18, 17)]];
        let expected = vec![vec![(20, 19), (19, 17)]];
        assert_eq!(
            Backgammon::remove_duplicate_states(state, sequences, -1),
            expected
        );
    }

    #[test]
    fn it_should_see_that_hitting_causes_unique_state() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[20] = -1;
        state.0[19] = 1;
        let sequences: Vec<Vec<(i8, i8)>> =
            vec![vec![(20, 19), (19, 17)], vec![(20, 18), (18, 17)]];
        let expected = sequences.clone();
        assert_eq!(
            Backgammon::remove_duplicate_states(state, sequences, -1),
            expected
        );
    }
}

#[cfg(test)]
mod get_entry_moves {
    use super::*;

    mod player1 {
        use super::*;
        #[test]
        fn is_should_return_empty_for_empty_moves() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.1 .0 = 1;
            let moves: &Vec<u8> = &vec![];
            assert!(Backgammon::get_entry_moves(moves, state, -1).is_empty());
        }

        #[test]
        fn it_should_return_empty_when_no_entry_is_possible() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.0[21] = 2;
            state.1 .0 = 1;
            let moves: &Vec<u8> = &vec![3];
            assert!(Backgammon::get_entry_moves(moves, state, -1).is_empty());
        }

        #[test]
        fn it_should_work_for_one_entry() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.1 .0 = 1;
            let moves: &Vec<u8> = &vec![3];
            assert!(Backgammon::get_entry_moves(moves, state, -1).len() == 1);
            assert_eq!(
                Backgammon::get_entry_moves(moves, state, -1)
                    .get(0)
                    .unwrap()
                    .value,
                (-1, 21)
            );
        }

        #[test]
        fn it_should_work_for_multiple_entries() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.0[19] = 2;
            state.1 .0 = 1;
            let moves: &Vec<u8> = &vec![3, 2];
            assert!(Backgammon::get_entry_moves(moves, state, -1).len() == 2);
            assert_eq!(
                Backgammon::get_entry_moves(moves, state, -1)
                    .get(0)
                    .unwrap()
                    .value,
                (-1, 22)
            );
            assert_eq!(
                Backgammon::get_entry_moves(moves, state, -1)
                    .get(1)
                    .unwrap()
                    .value,
                (-1, 21)
            );
        }
    }

    mod player2 {
        use super::*;
        #[test]
        fn is_should_return_empty_for_empty_moves() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.1 .1 = 1;
            let moves: &Vec<u8> = &vec![];
            assert!(Backgammon::get_entry_moves(moves, state, 1).is_empty());
        }

        #[test]
        fn it_should_return_empty_when_no_entry_is_possible() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.0[2] = -2;
            state.1 .1 = 1;
            let moves: &Vec<u8> = &vec![3];
            assert!(Backgammon::get_entry_moves(moves, state, 1).is_empty());
        }

        #[test]
        fn it_should_work_for_one_entry() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.1 .1 = 1;
            let moves: &Vec<u8> = &vec![3];
            assert!(Backgammon::get_entry_moves(moves, state, 1).len() == 1);
            assert_eq!(
                Backgammon::get_entry_moves(moves, state, 1)
                    .get(0)
                    .unwrap()
                    .value,
                (-1, 2)
            );
        }

        #[test]
        fn it_should_work_for_multiple_entries() {
            let mut state = Backgammon::get_initial_state();
            state.0 = [0; 24];
            state.0[4] = -2;
            state.1 .1 = 1;
            let moves: &Vec<u8> = &vec![3, 2];
            assert!(Backgammon::get_entry_moves(moves, state, 1).len() == 2);
            assert_eq!(
                Backgammon::get_entry_moves(moves, state, 1)
                    .get(0)
                    .unwrap()
                    .value,
                (-1, 1)
            );
            assert_eq!(
                Backgammon::get_entry_moves(moves, state, 1)
                    .get(1)
                    .unwrap()
                    .value,
                (-1, 2)
            );
        }
    }
}

#[cfg(test)]
mod get_valid_moves {
    use super::*;

    #[test]
    fn it_should_return_empty_for_no_possible_moves() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[20] = -1;
        state.0[19] = 2;
        state.0[18] = 2;
        let roll = (1, 2);
        assert!(Backgammon::get_valid_moves(roll, state, -1).is_empty());
    }

    #[test]
    fn it_should_return_same_for_inverted_roll() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[20] = -1;
        let roll_1 = (1, 2);
        let roll_2 = (2, 1);
        assert_eq!(
            Backgammon::get_valid_moves(roll_1, state, -1),
            Backgammon::get_valid_moves(roll_2, state, -1)
        );
    }

    #[test]
    fn it_should_work_for_normal_roll() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[20] = -1;
        let roll = (1, 2);
        let expected: Vec<Vec<(i8, i8)>> = vec![vec![(20, 19), (19, 17)]];
        assert_eq!(Backgammon::get_valid_moves(roll, state, -1), expected);
    }

    #[test]
    fn it_should_work_for_entry_move() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[21] = 2;
        state.1 .0 = 1;
        let roll = (1, 2);
        let expected: Vec<Vec<(i8, i8)>> = vec![vec![(-1, 23)], vec![(-1, 22)]];
        assert_eq!(Backgammon::get_valid_moves(roll, state, -1), expected);
    }

    #[test]
    fn it_should_work_for_double_roll() {
        let mut state = Backgammon::get_initial_state();
        state.0 = [0; 24];
        state.0[20] = -1;
        let roll = (1, 1);
        let expected: Vec<Vec<(i8, i8)>> = vec![vec![(20, 19), (19, 18), (18, 17), (17, 16)]];
        assert_eq!(Backgammon::get_valid_moves(roll, state, -1), expected);
    }
}
