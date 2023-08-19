extern crate proptest;

// use proptest::prelude::*;
use die_e::alphazero::encoding::*;

#[cfg(test)]
mod encoding_single_and_zero_moves {
    use super::*;

    #[test]
    fn original_should_be_same_as_decoded_empty_actions() {
        let player: i8 = -1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_normal_hrf_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(4, 2)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_normal_hrs_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(4, 3)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_bar_hrf_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(-1, 22)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_bar_hrs_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(-1, 23)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_hrf_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(1, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_hrs_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(0, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_hrf_player_1_2() {
        let player: i8 = -1;
        let roll: (u8, u8) = (6, 3);
        let actions: Vec<(i8, i8)> = vec![(1, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_hrs_player_1_2() {
        let player: i8 = -1;
        let roll: (u8, u8) = (6, 3);
        let actions: Vec<(i8, i8)> = vec![(2, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_normal_hrf_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(19, 21)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_normal_hrs_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(19, 20)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_bar_hrf_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(-1, 1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_bar_hrs_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(-1, 0)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_hrf_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(22, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_hrs_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(23, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_hrf_player_2_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (6, 3);
        let actions: Vec<(i8, i8)> = vec![(22, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_hrs_player_2_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (6, 3);
        let actions: Vec<(i8, i8)> = vec![(21, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }
}

#[cfg(test)]
mod encoding_double_moves {
    use super::*;

    #[test]
    fn original_should_be_same_as_decoded_normal_move_high_roll_first_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(23, 21), (5, 4)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_bar_move_high_roll_first_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(-1, 22), (-1, 23)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_move_high_roll_first_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(1, -1), (0, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_move_high_roll_first_player_1_2() {
        let player: i8 = -1;
        let roll: (u8, u8) = (4, 6);
        let actions: Vec<(i8, i8)> = vec![(1, -1), (0, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_normal_move_high_roll_second_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(5, 4), (23, 21)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_bar_move_high_roll_second_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(-1, 23), (-1, 22)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_move_high_roll_second_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(0, -1), (1, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_move_high_roll_second_player_1_2() {
        let player: i8 = -1;
        let roll: (u8, u8) = (4, 6);
        let actions: Vec<(i8, i8)> = vec![(0, -1), (1, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_normal_move_high_roll_first_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(1, 3), (21, 22)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_bar_move_high_roll_first_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(-1, 1), (-1, 0)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_move_high_roll_first_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(22, -1), (23, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_move_high_roll_first_player_2_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (4, 6);
        let actions: Vec<(i8, i8)> = vec![(22, -1), (23, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_normal_move_high_roll_second_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(4, 5), (21, 23)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_bar_move_high_roll_second_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(-1, 0), (-1, 1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_move_high_roll_second_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(23, -1), (22, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_move_high_roll_second_player_2_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (4, 6);
        let actions: Vec<(i8, i8)> = vec![(23, -1), (22, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }
}

#[cfg(test)]
mod encoding_exceptional_cases {
    use super::*;

    #[test]
    fn original_should_be_same_as_decoded_bar_and_normal_hrf_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (6, 1);
        let actions: Vec<(i8, i8)> = vec![(-1, 18), (18, 17)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_bar_and_normal_hrs_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (6, 1);
        let actions: Vec<(i8, i8)> = vec![(-1, 23), (23, 17)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_and_normal_hrf_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (6, 5);
        let actions: Vec<(i8, i8)> = vec![(6, 0), (3, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_and_normal_hrs_player_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (6, 5);
        let actions: Vec<(i8, i8)> = vec![(6, 1), (3, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_bar_and_normal_hrf_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (6, 1);
        let actions: Vec<(i8, i8)> = vec![(-1, 5), (5, 6)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_bar_and_normal_hrs_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (6, 1);
        let actions: Vec<(i8, i8)> = vec![(-1, 0), (0, 6)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_and_normal_hrf_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (6, 5);
        let actions: Vec<(i8, i8)> = vec![(17, 23), (20, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_collection_and_normal_hrs_player_2() {
        let player: i8 = 1;
        let roll: (u8, u8) = (6, 5);
        let actions: Vec<(i8, i8)> = vec![(17, 22), (20, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_exception_case_1() {
        let player: i8 = -1;
        let roll: (u8, u8) = (4, 5);
        let actions: Vec<(i8, i8)> = vec![(0, -1), (0, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(0, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }

    #[test]
    fn original_should_be_same_as_decoded_when_enc_is_0() {
        let player: i8 = -1;
        let roll: (u8, u8) = (2, 1);
        let actions: Vec<(i8, i8)> = vec![(0, -1), (0, -1)];
        let enc = encode(actions.clone(), roll, player);
        let dec = decode(enc, roll, player);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }
}

// #[cfg(test)]
// mod prop_tests {
//     use super::*;

//     fn roll_from_actions_normal(actions: &[(i8, i8)]) -> (u8, u8) {
//         let die_1 = (actions[0].0 - actions[0].1).unsigned_abs();
//         let die_2 = (actions[1].0 - actions[1].1).unsigned_abs();
//         (die_1, die_2)
//     }

//     proptest! {
//         #[test]
//         fn test_encode_decode_two_normal_moves_player_1(actions in proptest::collection::vec((0i8..=23, 0i8..=23), 2)) {
//             // Ensure that the difference between from and to values is not greater than 6
//             prop_assume!(actions.iter().all(|(from, to)| (from - to).abs() <= 6));
    
//             let roll = roll_from_actions_normal(&actions);
//             let player = -1;

//             // Ensure that from value is greater than to value for this test case
//             prop_assume!(actions[0].0 > actions[0].1);
//             prop_assume!(actions[1].0 > actions[1].1);
    
//             let encoded = encode(actions.clone(), roll, player);
//             let decoded_actions = decode(encoded, roll, player);
    
//             // The original and decoded actions should match
//             prop_assert_eq!(actions, decoded_actions);
//         }
//     }
// }
