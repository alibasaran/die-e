use crate::backgammon::Actions;

use super::Backgammon;

impl Backgammon {
    pub fn encode(&self, actions: &Actions) -> u32 {
        assert!(actions.len() <= 2, "encoding for actions > 2 is not implemented!");
    
        // return special case for when there are no actions
        if actions.is_empty() {
            return 676;
        }
        
        // get the roll and the high and low roll values
        let roll = self.roll;
        let (high_roll, low_roll) = if roll.0 > roll.1 { (roll.0, roll.1) } else { (roll.1, roll.0) };
        let mut low_roll_flag = false;
    
        // get the minimum roll values required to be able to play the actions provided
        let mut minimum_rolls = actions.iter().map(|&(from, to)| {
            match (from, to) {
                (-1, t) if t < 6 => (t + 1) as u8,
                (-1, t) if t > 17 => (24 - t) as u8,
                (f, -1) if f < 6 => (f - (-1)) as u8,
                (f, -1) if f > 17 => (24 - f) as u8,
                (f, t) => (f - t).unsigned_abs(),
            }
        }).collect::<Vec<u8>>();
    
        // if only a single move is played, set the second minimum roll to be 0
        if minimum_rolls.len() == 1 {minimum_rolls.push(0);}
    
        /*
         * use base 26 encoding to encode the actions
         * the values 0-23 is reserved for 'normal' moves (i.e. moves that are not from the bar or are not collection moves)
         * and corresponds to the 'from' value of the action
         * 24 is reserved for moves from the bar
         * 25 is reserved for single moves (i.e. the value of the non-existent 'second move' of a single-move action)
         * add the corresponding value for the first move and the corresponding value for the second move times 26
         */ 
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
                        // raise the low_roll_flag if the first move is certainly the low roll
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
    
        // add 26 * 25 to encode_sum if the action has a single move
        if actions.get(1).is_none() {encode_sum += 26 * 25}
    
        // compute whether the high roll was played first
        let high_roll_first = if low_roll_flag {false} else if minimum_rolls[1] != 0 {minimum_rolls[0] >= minimum_rolls[1]} else {minimum_rolls[0] > low_roll};
    
        // add 676 to the final value if the high roll was played first
        if high_roll_first { encode_sum } else { encode_sum + 676 }
    }
    
    pub fn decode(&self, action: u32) -> Actions {
        // decoding for the special value (676) for empty actions
        if action == 676 {
            return vec![];
        }
    
        let roll = self.roll;
        let player = self.player;
        let high_roll_first = action < 676;

        /*
         * extract the from values of the first and second action
         * the from value '24' suggests a move from the bar
         * note that the from2 value will be 25 if the action has a single move
         */
        let (from1, from2) = (if high_roll_first { action } else { action - 676 } % 26, 
                                        if high_roll_first { action } else { action - 676 } / 26);
        let single_action = from2 == 25;
        let (high_roll, low_roll) = if roll.0 > roll.1 { (roll.0, roll.1) } else { (roll.1, roll.0) };
        let (mut from1_i8, mut from2_i8) = (from1 as i8, from2 as i8);
        let (low_roll_i8, high_roll_i8) = (low_roll as i8, high_roll as i8);
    
        // convert from values from 24 to -1 only if the player is the second player (helps with computation)
        if from1_i8 == 24 && player == 1 { from1_i8 = -1; }
        if from2_i8 == 24 && player == 1 { from2_i8 = -1; }
    
        // extract 'to' values
        let (mut to1, mut to2) = if high_roll_first {
            (from1_i8 + high_roll_i8 * player, from2_i8 + low_roll_i8 * player)
        } else {
            (from1_i8 + low_roll_i8 * player, from2_i8 + high_roll_i8 * player)
        };
    
        // convert the 'to' and 'from' values from -1
        if to1 >= 24 || to1 <= -1 { to1 = -1; }
        if to2 >= 24 || to2 <= -1 { to2 = -1; }
        if from1_i8 == 24 { from1_i8 = -1; }
        if from2_i8 == 24 { from2_i8 = -1; }
    
        if single_action { vec![(from1_i8, to1)] } else { vec![(from1_i8, to1), (from2_i8, to2)] }
    }
}

