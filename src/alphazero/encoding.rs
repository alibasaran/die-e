use crate::backgammon::Actions;

pub fn encode(actions: Actions, roll: (u8, u8), _player: i8) -> u32 {
    assert!(actions.len() <= 2, "encoding for actions > 2 is not yet implemented!");

    if actions.is_empty() {
        return 676;
    }

    let (high_roll, low_roll) = if roll.0 > roll.1 { (roll.0, roll.1) } else { (roll.1, roll.0) };
    let mut low_roll_flag = false;

    let mut minimum_rolls = actions.iter().map(|&(from, to)| {
        match (from, to) {
            (-1, t) if t < 6 => (t + 1) as u8,
            (-1, t) if t > 17 => (24 - t) as u8,
            (f, -1) if f < 6 => (f - (-1)) as u8,
            (f, -1) if f > 17 => (24 - f) as u8,
            (f, t) => (f - t).unsigned_abs(),
        }
    }).collect::<Vec<u8>>();

    if minimum_rolls.len() == 1 {minimum_rolls.push(0);}

    let mut encode_sum = 0;

    for (i, &(from, to)) in actions.iter().enumerate() {
        match i {
            0 => {match (from, to) {
                (-1, t) if t < 6 => {encode_sum += 24;},
                (-1, t) if t > 17 => {encode_sum += 24;},
                (f, -1) if f < 6 => {encode_sum += f as u32;},
                (f, -1) if f > 17 => {encode_sum += f as u32;},
                (f, _) => {
                    encode_sum += f as u32; 
                    low_roll_flag = minimum_rolls.first().unwrap() == &low_roll;
                },
            }},
            1 => {match (from, to) {
                (-1, t) if t < 6 => {encode_sum += 26 * 24;},
                (-1, t) if t > 17 => {encode_sum += 26 * 24;},
                (f, -1) if f < 6 => {encode_sum += 26 * (f as u32);},
                (f, -1) if f > 17 => {encode_sum += 26 * (f as u32);},
                (f, _) => {encode_sum += 26 * (f as u32)},
            }},
            _ => unreachable!(),
        }
    }

    if actions.get(1).is_none() {encode_sum += 26 * 25}

    let high_roll_first = if low_roll_flag {false} else if minimum_rolls[1] != 0 {minimum_rolls[0] >= minimum_rolls[1]} else {minimum_rolls[0] == high_roll};

    if high_roll_first { encode_sum } else { encode_sum + 676 }
}

pub fn decode(action: u32, roll: (u8, u8), player: i8) -> Actions {
    if action == 676 {
        return vec![];
    }

    let high_roll_first = action < 676;
    let (from1, from2) = (if high_roll_first { action } else { action - 676 } % 26, 
                                    if high_roll_first { action } else { action - 676 } / 26);
    let single_action = from2 == 25;
    let (high_roll, low_roll) = if roll.0 > roll.1 { (roll.0, roll.1) } else { (roll.1, roll.0) };
    let (mut from1_i8, mut from2_i8) = (from1 as i8, from2 as i8);
    let (low_roll_i8, high_roll_i8) = (low_roll as i8, high_roll as i8);

    if from1_i8 == 24 && player == 1 { from1_i8 = -1; }
    if from2_i8 == 24 && player == 1 { from2_i8 = -1; }

    let (mut to1, mut to2) = if high_roll_first {
        (from1_i8 + high_roll_i8 * player, from2_i8 + low_roll_i8 * player)
    } else {
        (from1_i8 + low_roll_i8 * player, from2_i8 + high_roll_i8 * player)
    };

    if to1 >= 24 || to1 <= -1 { to1 = -1; }
    if to2 >= 24 || to2 <= -1 { to2 = -1; }
    if from1_i8 == 24 { from1_i8 = -1; }
    if from2_i8 == 24 { from2_i8 = -1; }

    if single_action { vec![(from1_i8, to1)] } else { vec![(from1_i8, to1), (from2_i8, to2)] }
}
