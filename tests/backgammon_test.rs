
use std::vec;

use die_e::backgammon::Backgammon;

#[test]
fn it_can_init() {
    let _bg = Backgammon::new();
    assert!(true)
}

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

#[test]
fn get_next_state_empty_actions() {
    let state = Backgammon::get_initial_state();
    let actions: Vec<(i8, i8)>= vec![];
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
fn get_next_state_normal_move_player1() {
    let state = Backgammon::get_initial_state();
    let actions: Vec<(i8, i8)>= vec![(23, 21), (23, 20)];
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
fn get_next_state_normal_move_player2() {
    let state = Backgammon::get_initial_state();
    let actions: Vec<(i8, i8)>= vec![(0, 3), (0, 3)];
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
fn get_next_state_hitting_player1() {
    let state = (
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, 0, 0, 1, 0, 1,
        ],
        (0, 0),
        (0, 0),
    );
    let actions: Vec<(i8, i8)>= vec![(18, 21), (18, 23)];
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
fn get_next_state_hitting_player2() {
    let state = (
        [
            0, 0, 0, 0, 0, 5, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        (0, 0),
        (0, 0),
    );
    let actions: Vec<(i8, i8)>= vec![(5, 8), (5, 10)];
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
fn get_next_state_collecting_player1() {
    let state = (
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2,
        ],
        (0, 0),
        (0, 0),
    );
    let actions: Vec<(i8, i8)>= vec![(23, -1), (23, -1)];
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
fn get_next_state_collecting_player2() {
    let state = (
        [
            0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        (0, 0),
        (0, 0),
    );
    let actions: Vec<(i8, i8)>= vec![(5, -1), (5, -1)];
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
fn get_next_state_from_bar_player1() {
    let state = (
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        (2, 0),
        (0, 0),
    );
    let actions: Vec<(i8, i8)>= vec![(-1, 23), (-1, 20)];
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
fn get_next_state_from_bar_player2() {
    let state = (
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        (0, 2),
        (0, 0),
    );
    let actions: Vec<(i8, i8)>= vec![(-1, 0), (-1, 3)];
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
fn get_next_state_from_bar_hitting_player1() {
    let state = (
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
        ],
        (2, 0),
        (0, 0),
    );
    let actions: Vec<(i8, i8)>= vec![(-1, 23), (-1, 20)];
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
fn get_next_state_from_bar_hitting_player2() {
    let state = (
        [
            -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        (0, 2),
        (0, 0),
    );
    let actions: Vec<(i8, i8)>= vec![(-1, 0), (-1, 3)];
    let expected = (
        [
            1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        (2, 0),
        (0, 0),
    );
    assert_eq!(Backgammon::get_next_state(state, actions, 1), expected);
}