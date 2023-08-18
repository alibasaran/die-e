use crate::backgammon::Actions;


pub fn encode(actions: Actions, roll: (u8, u8), _player: i8) -> u32 {
    assert!(actions.len() <= 2, "encoding for actions > 2 is not yet implemented!");

    let high_roll = if roll.0 > roll.1 {roll.0} else {roll.1};
    let mut high_roll_first = false;

    let mut encode_sum: u32 = 0;

    if let Some((from, to)) = actions.first() {
        if *from == -1 && *to < 6 {
            let dist = (to + 1) as u8;
            high_roll_first = dist == high_roll;
            encode_sum += 24;
        } else if *from == -1 && *to > 17 {
            let dist = (24 - to) as u8;
            high_roll_first = dist == high_roll;
            encode_sum += 24;
        } else {
            let dist: u8 = (from - to).abs().try_into().unwrap();
            high_roll_first = dist == high_roll;
            encode_sum += *from as u32;
        }
    }

    if let Some((from, _)) = actions.get(1) {
        if *from == -1 {
            encode_sum += 25 * 24
        } else {
            encode_sum += 25 * (*from as u32)
        }
    }

    if high_roll_first {
        encode_sum
    } else {
        encode_sum + 625
    }
    
}

pub fn decode(action: u32, roll: (u8, u8), player: i8) -> Actions {
    // Check if the higher roll is made first
    let high_roll_first = action < 625;

    // calculate from which position the rolls will be played
    let (from1, from2) = {
        let action_sub = if high_roll_first { action } else { action - 625 };
        (action_sub % 25, action_sub / 25)
    };

    // Retrive the high and low roll from the roll 
    let (high_roll, low_roll) = if roll.0 > roll.1 { (roll.0, roll.1) } else { (roll.1, roll.0) };
    let (mut from1_i8, mut from2_i8) = (from1 as i8, from2 as i8);
    let (low_roll_i8, high_roll_i8) = (low_roll as i8, high_roll as i8);

    if from1_i8 == 24 && player == 1 {
        from1_i8 = -1;
    }

    if from2_i8 == 24 && player == 1 {
        from2_i8 = -1;
    }

    // Calculate where the pieces will land
    let (mut to1, mut to2) = if high_roll_first {
        (from1_i8 + high_roll_i8 * player, from2_i8 + low_roll_i8 * player)
    } else {
        (from1_i8 + low_roll_i8 * player, from2_i8 + high_roll_i8 * player)
    };

    if to1 >= 24 || to1 <= -1 {
        to1 = -1;
    }

    if to2 >= 24 || to2 <= -1 {
        to2 = -1;
    }

    if from1_i8 == 24 {
        from1_i8 = -1;
    }

    if from2_i8 == 24 {
        from2_i8 = -1;
    }

    vec![(from1_i8, to1), (from2_i8, to2)]
}
