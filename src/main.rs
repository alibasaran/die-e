use die_e::{
    backgammon::backgammon_logic::Backgammon,
    mcts::alpha_mcts::TimeLogger,
    constants::DEVICE,
};
use itertools::Itertools;
use tch::Tensor;

fn main() {
    // Use 4 cores
    rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build_global()
        .unwrap();
    // let mut vs = VarStore::new(*DEVICE);
    // vs.load("./models/best_model.ot").unwrap();

    let mut bg = Backgammon::new();

    bg.board.0[12] = 10;
    
    println!("{}", bg.to_pretty_str())

    // let model_path = Path::new("./models/best_model.ot");
    // let config = AlphaZeroConfig {
    //     temperature: 1.,
    //     learn_iterations: 100,
    //     self_play_iterations: 4,
    //     batch_size: 2048,
    //     num_epochs: 2,
    //     model_path: Some(model_path.to_str().unwrap().to_string()),
    // };
    // let mut az = AlphaZero::new(config);
    // az.learn_parallel();
    // // 'outer_loop: for i in 0..10000 {
    // //     let mut state = Backgammon::new();
    // //     state.roll_die();
    // //     for i in 0..10000 {
    // //         let game_end = Backgammon::check_win_without_player(state.board);
    // //         println!("\nGame ended? => {:?}", game_end);
    // //         if game_end.is_some() {
    // //             state.display_board();
    // //             break;
    // //         }
    // //         println!("Roll {:?}", &state.roll);
    // //         let valid_moves = state.get_valid_moves_len_always_2();
    // //         println!("Valid moves: {:?}", valid_moves);
    // //         let encoded_valid_moves = valid_moves.iter().map(|m| state.encode(m)).collect_vec();
    // //         println!("Valid moves encoded: {:?}", encoded_valid_moves);
    // //         assert!(encoded_valid_moves.iter().all_unique());
    // //         if valid_moves.is_empty() {
    // //             state.display_board();
    // //             state.skip_turn();
    // //             continue
    // //         }
    // //         let next_move = az.get_next_move_for_state(&state);
    // //         println!("Result: {:?}", next_move);
    // //         state.apply_move(&next_move);
    // //         assert!(valid_moves.contains(&next_move));
    // //         state.is_valid();
    // //         if i >= 9999 {
    // //             println!("Game didn't end!");
    // //             state.display_board();
    // //             break 'outer_loop
    // //         }
    // //     }
    // // }

    // let player1 = Player {
    //     player_type: Agent::Model,
    //     model: Some(az),
    // };

    // let player2 = Player {
    //     player_type: Agent::Random,
    //     model: None,
    // };

    // let result = play(player1, player2);

    // println!("Wins P1: {:?}", result.wins_p1)
    // println!("Result: {:?}", result)

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
