use die_e::alphazero::encoding::*;

#[cfg(test)]
mod encoding {
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
}
