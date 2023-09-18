use std::{time::{SystemTime, UNIX_EPOCH}, path::Path};

use die_e::{
    alphazero::{
        alphazero::{AlphaZero, AlphaZeroConfig, MemoryFragment},
        nnet::ResNet,
    },
    backgammon::Backgammon,
    constants::{DEVICE, N_SELF_PLAY_BATCHES, DEFAULT_TYPE},
    mcts::{
        alpha_mcts::{alpha_mcts_parallel, TimeLogger},
        node_store::NodeStore,
    },
};
use itertools::Itertools;
use tch::{Device, Kind, Tensor, nn::VarStore};

fn main() {
    // let mut vs = VarStore::new(*DEVICE);
    // vs.load("./models/best_model.ot").unwrap();
    let config = AlphaZeroConfig {
        temperature: 1.,
        learn_iterations: 100,
        self_play_iterations: 16,
        batch_size: 2048,
        num_epochs: 2,
    };
    let mut az = AlphaZero::new(config);
    az.learn_parallel();
    // let mut timer = TimeLogger::default();
    // timer.start();
    // let a = Tensor::rand([2048, 1352], (Kind::Float, Device::Mps));
    // let max = a.argmax(1, false);
    // max.print();
    // timer.log("argmax done")
}

// Time state conversion using Backgammon::as_tensor
// Outcome: 1700ms when to_device mps was called on as_tensor vs 60ms as_tensor after stacking states
fn time_states_conv() {
    let mut timer = TimeLogger::default();
    let states = (0..2048)
        .map(|_| {
            let mut bg = Backgammon::new();
            bg.roll_die();
            bg
        })
        .collect_vec();
    timer.start();
    let states_vec = states.iter().map(|state| state.as_tensor()).collect_vec();
    let _ = Tensor::stack(&states_vec, 0)
        .squeeze_dim(1)
        .to_device(*DEVICE);
    timer.log("Inital states conversion");
}
