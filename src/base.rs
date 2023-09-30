use std::fmt::Debug;

use tch::Tensor;


use serde::{Serialize, de::DeserializeOwned};

pub trait LearnableGame: Clone + Debug + DeserializeOwned + Serialize + Send + Sync + Copy {

    type Move: Clone + Debug + DeserializeOwned + Serialize + Send + Sync + PartialEq;
    const EMPTY_MOVE: Self::Move;
    const IS_DETERMINISTIC: bool;

    // Parameters used when creating and running models
    // Number of unique encodings in a game
    // Ex. TicTacToe is 9 as there are 9 different moves you can make
    const ACTION_SPACE_SIZE: i64;
    const N_INPUT_CHANNELS: i64;
    const CONV_OUTPUT_SIZE: i64; // The board size in backgammon there are 24 squares so 24

    fn new() -> Self;

    fn get_valid_moves(&self) -> Vec<Self::Move>;
    fn apply_move(&mut self, action: &Self::Move);
    fn roll_die(&mut self) -> (u8, u8) {
        if Self::IS_DETERMINISTIC {
            panic!("roll_die called on deterministic game!")
        }
        unimplemented!("You should implement roll_die for non-deterministic games!")
    }
    fn skip_turn(&mut self);
    fn get_player(&self) -> i8;

    fn check_winner(&self) -> Option<i8>;
    
    fn as_tensor(&self) -> Tensor;
    fn decode(&self, action: u32) -> Self::Move;
    fn encode(&self, action: &Self::Move) -> u32;

    fn get_id(&self) -> usize;
    fn set_id(&mut self, new_id: usize);

    fn to_pretty_str(&self) -> String;
}
