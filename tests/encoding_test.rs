extern crate proptest;

#[cfg(test)]
mod encoding_single_and_zero_moves {
    use die_e::backgammon::Backgammon;
    use test_case::test_case;

    #[test_case((2, 1), -1, vec![]; "original_should_be_same_as_decoded_empty_actions")]
    #[test_case((2, 1), -1, vec![(4, 2)]; "original_should_be_same_as_decoded_normal_hrf_player_1")]
    #[test_case((2, 1), -1, vec![(4, 3)]; "original_should_be_same_as_decoded_normal_hrs_player_1")]
    #[test_case((2, 1), -1, vec![(-1, 22)]; "original_should_be_same_as_decoded_bar_hrf_player_1")]
    #[test_case((2, 1), -1, vec![(-1, 23)]; "original_should_be_same_as_decoded_bar_hrs_player_1")]
    #[test_case((2, 1), -1, vec![(1, -1)]; "original_should_be_same_as_decoded_collection_hrf_player_1")]
    #[test_case((2, 1), -1, vec![(0, -1)]; "original_should_be_same_as_decoded_collection_hrs_player_1")]
    #[test_case((6, 3), -1, vec![(1, -1)]; "original_should_be_same_as_decoded_collection_hrf_player_1_2")]
    #[test_case((6, 3), -1, vec![(2, -1)]; "original_should_be_same_as_decoded_collection_hrs_player_1_2")]
    #[test_case((2, 1), 1, vec![(19, 21)]; "original_should_be_same_as_decoded_normal_hrf_player_2")]
    #[test_case((2, 1), 1, vec![(19, 20)]; "original_should_be_same_as_decoded_normal_hrs_player_2")]
    #[test_case((2, 1), 1, vec![(-1, 1)]; "original_should_be_same_as_decoded_bar_hrf_player_2")]
    #[test_case((2, 1), 1, vec![(-1, 0)]; "original_should_be_same_as_decoded_bar_hrs_player_2")]
    #[test_case((2, 1), 1, vec![(22, -1)]; "original_should_be_same_as_decoded_collection_hrf_player_2")]
    #[test_case((2, 1), 1, vec![(23, -1)]; "original_should_be_same_as_decoded_collection_hrs_player_2")]
    #[test_case((6, 3), 1, vec![(22, -1)]; "original_should_be_same_as_decoded_collection_hrf_player_2_2")]
    #[test_case((6, 3), 1, vec![(21, -1)]; "original_should_be_same_as_decoded_collection_hrs_player_2_2")]
    fn single_and_zero_move_tests(roll: (u8, u8), player: i8, actions: Vec<(i8, i8)>) {
        let mut bg = Backgammon::init_with_fields(([0; 24], (0, 0), (0, 0)), player, false);
        bg.roll = roll;
        let enc = bg.encode(&actions.clone());
        let dec = bg.decode(enc);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }
}

#[cfg(test)]
mod encoding_double_moves {
    use die_e::backgammon::Backgammon;
    use test_case::test_case;

    #[test_case((2, 1), -1, vec![(23, 21), (5, 4)]; "original_should_be_same_as_decoded_normal_move_high_roll_first_player_1")]
    #[test_case((2, 1), -1, vec![(-1, 22), (-1, 23)]; "original_should_be_same_as_decoded_bar_move_high_roll_first_player_1")]
    #[test_case((2, 1), -1, vec![(1, -1), (0, -1)]; "original_should_be_same_as_decoded_collection_move_high_roll_first_player_1")]
    #[test_case((4, 6), -1, vec![(1, -1), (0, -1)]; "original_should_be_same_as_decoded_collection_move_high_roll_first_player_1_2")]
    #[test_case((2, 1), -1, vec![(5, 4), (23, 21)]; "original_should_be_same_as_decoded_normal_move_high_roll_second_player_1")]
    #[test_case((2, 1), -1, vec![(-1, 23), (-1, 22)]; "original_should_be_same_as_decoded_bar_move_high_roll_second_player_1")]
    #[test_case((2, 1), -1, vec![(0, -1), (1, -1)]; "original_should_be_same_as_decoded_collection_move_high_roll_second_player_1")]
    #[test_case((4, 6), -1, vec![(0, -1), (1, -1)]; "original_should_be_same_as_decoded_collection_move_high_roll_second_player_1_2")]
    #[test_case((2, 1), 1, vec![(1, 3), (21, 22)]; "original_should_be_same_as_decoded_normal_move_high_roll_first_player_2")]
    #[test_case((2, 1), 1, vec![(-1, 1), (-1, 0)]; "original_should_be_same_as_decoded_bar_move_high_roll_first_player_2")]
    #[test_case((2, 1), 1, vec![(22, -1), (23, -1)]; "original_should_be_same_as_decoded_collection_move_high_roll_first_player_2")]
    #[test_case((4, 6), 1, vec![(22, -1), (23, -1)]; "original_should_be_same_as_decoded_collection_move_high_roll_first_player_2_2")]
    #[test_case((2, 1), 1, vec![(4, 5), (21, 23)]; "original_should_be_same_as_decoded_normal_move_high_roll_second_player_2")]
    #[test_case((2, 1), 1, vec![(-1, 0), (-1, 1)]; "original_should_be_same_as_decoded_bar_move_high_roll_second_player_2")]
    #[test_case((2, 1), 1, vec![(23, -1), (22, -1)]; "original_should_be_same_as_decoded_collection_move_high_roll_second_player_2")]
    #[test_case((4, 6), 1, vec![(23, -1), (22, -1)]; "original_should_be_same_as_decoded_collection_move_high_roll_second_player_2_2")]
    fn double_moves_test(roll: (u8, u8), player: i8, actions: Vec<(i8, i8)>) {
        let mut bg = Backgammon::init_with_fields(([0; 24], (0, 0), (0, 0)), player, false);
        bg.roll = roll;
        let enc = bg.encode(&actions.clone());
        let dec = bg.decode(enc);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }
}

#[cfg(test)]
mod encoding_exceptional_cases {
    use die_e::backgammon::Backgammon;
    use test_case::test_case;

    #[test_case((6, 1), -1, vec![(-1, 18), (18, 17)]; "original_should_be_same_as_decoded_bar_and_normal_hrf_player_1")]
    #[test_case((6, 1), -1, vec![(-1, 23), (23, 17)]; "original_should_be_same_as_decoded_bar_and_normal_hrs_player_1")]
    #[test_case((6, 5), -1, vec![(6, 0), (3, -1)]; "original_should_be_same_as_decoded_collection_and_normal_hrf_player_1")]
    #[test_case((6, 5), -1, vec![(6, 1), (3, -1)]; "original_should_be_same_as_decoded_collection_and_normal_hrs_player_1")]
    #[test_case((6, 1), 1, vec![(-1, 5), (5, 6)]; "original_should_be_same_as_decoded_bar_and_normal_hrf_player_2")]
    #[test_case((6, 1), 1, vec![(-1, 0), (0, 6)]; "original_should_be_same_as_decoded_bar_and_normal_hrs_player_2")]
    #[test_case((6, 5), 1, vec![(17, 23), (20, -1)]; "original_should_be_same_as_decoded_collection_and_normal_hrf_player_2")]
    #[test_case((6, 5), 1, vec![(17, 22), (20, -1)]; "original_should_be_same_as_decoded_collection_and_normal_hrs_player_2")]
    #[test_case((4, 5), -1, vec![(0, -1), (0, -1)]; "original_should_be_same_as_decoded_exception_case_1")]
    #[test_case((2, 1), -1, vec![(0, -1), (0, -1)]; "original_should_be_same_as_decoded_when_enc_is_0")]
    #[test_case((6, 1), 1, vec![(21, -1)]; "should_work_when_single_hrf_collection_while_low_roll_possible_without_collection")]
    fn exceptions_test(roll: (u8, u8), player: i8, actions: Vec<(i8, i8)>) {
        let mut bg = Backgammon::init_with_fields(([0; 24], (0, 0), (0, 0)), player, false);
        bg.roll = roll;
        let enc = bg.encode(&actions.clone());
        let dec = bg.decode(enc);
        println!("actions: {:?}, \nenc: {}, \ndec: {:?}", actions, enc, dec);
        assert_eq!(actions, dec);
    }
}
