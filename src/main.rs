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
    }, versus::{Game, play_mcts_vs_model, save_game, load_game, print_game},
};
use itertools::Itertools;
use tch::{Device, Kind, Tensor, nn::VarStore};

fn main() {
    // let mut vs = VarStore::new(*DEVICE);
    // vs.load("./models/best_model.ot").unwrap();
    let model_path = Path::new("./models/trained_model.ot");
    let config = AlphaZeroConfig {
        temperature: 1.,
        learn_iterations: 100,
        self_play_iterations: 8,
        batch_size: 2048,
        num_epochs: 2,
        model_path: Some(model_path.to_str().unwrap().to_string())
    };
    let mut az = AlphaZero::new(config);

    let random_net = ResNet::default();

    let result = az.play_vs_model(&random_net);
    println!("Result: {:?}", result);

    // let data_path1 = Path::new("./data/run-0/lrn-0/sp-0");
    // let mut memory = az.load_training_data(data_path1);

    // let data_path2 = Path::new("./data/run-0/lrn-0/sp-1");
    // let mut memory2 = az.load_training_data(data_path2);

    // memory.append(&mut memory2);
    // az.train(&mut memory);

    // let _ = az.save_current_model(model_path);

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
