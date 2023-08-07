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
