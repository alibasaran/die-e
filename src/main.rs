use std::time::{SystemTime, UNIX_EPOCH};

use die_e::{alphazero::{nnet::ResNet, alphazero::{AlphaZeroConfig, AlphaZero}}, mcts::{node_store::NodeStore, alpha_mcts::{alpha_mcts_parallel, TimeLogger}}, backgammon::Backgammon, constants::{N_SELF_PLAY_BATCHES, DEVICE}};
use itertools::Itertools;
use tch::{Tensor, Kind, Device};


fn main() {
    let config = AlphaZeroConfig {
        temperature: 1.,
        learn_iterations: 10,
        self_play_iterations: 10,
        batch_size: 2048,
        num_epochs: 2,
    };
    let mut az = AlphaZero::new(config);
    az.learn_parallel();
    // time_states_conv();
}

// Time state conversion using Backgammon::as_tensor
// Outcome: 1700ms when to_device mps was called on as_tensor vs 60ms as_tensor after stacking states 
fn time_states_conv() {
    let mut timer = TimeLogger::default();
    let states = (0..2048).map(|_| {
        let mut bg = Backgammon::new();
        bg.roll_die();
        bg
    }).collect_vec();
    timer.start();
    let states_vec = states.iter().map(|state| state.as_tensor()).collect_vec();
    let _ = Tensor::stack(
        &states_vec,
        0
    ).squeeze_dim(1).to_device(*DEVICE);
    timer.log("Inital states conversion");
}
