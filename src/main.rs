use std::{
    collections::HashMap,
    time::{Duration},
};

use die_e::{
    alphazero::{
        alphazero::{AlphaZero},
    },
    backgammon::{Actions, Backgammon},
    constants::{DEVICE},
    mcts::{
        alpha_mcts::{alpha_mcts_parallel, TimeLogger},
        node_store::NodeStore,
        simple_mcts::mct_search,
        utils::get_prob_tensor_parallel,
    },
    versus::{
        Game, Agent, Turn,
    },
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use itertools::Itertools;
use rand::seq::SliceRandom;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use tch::{Tensor};

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

#[derive(Debug)]
enum PlayerType {
    Model,
    MCTS,
    Random,
}

struct Player {
    player_type: Agent,
    model: Option<AlphaZero>,
}

#[derive(Debug)]
struct PlayResult {
    player1: Agent,
    player2: Agent,
    wins_p1: usize,
    wins_p2: usize,
    n_games: usize,
    winrate: f64,
    games: Vec<Game> // TODO: games, list of all games played during the session
}

fn play(player1: Player, player2: Player) -> PlayResult {
    println!("\nStarting play!");
    let pb_play = MultiProgress::new();
    let sty = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
    )
    .unwrap();

    let num_games = 100;
    let round_limit = 400;
    let mut games: HashMap<usize, (Backgammon, Game)> = HashMap::from_iter((0..num_games).map(|idx| {
        let mut bg = Backgammon::new();
        let game = Game::new(player1.player_type.clone(), player2.player_type.clone(), bg);
        if idx >= num_games / 2 {
            bg.skip_turn();
        }
        bg.roll_die();
        bg.id = idx;
        (idx, (bg, game))
    }));

    let mut games_played: Vec<Game> = vec![];
    let mut wins_p1 = 0.;
    let player_p1 = -1;

    let pb_games =
        pb_play.add(ProgressBar::new(num_games.try_into().unwrap()).with_style(sty.clone()));

    let mut round_count = 0;
    while !games.is_empty() {
        pb_games.set_position((num_games - games.len()) as u64);
        pb_games.set_message(format!("On round: {}", round_count));
        let (games_p1, games_p2): (Vec<Backgammon>, Vec<Backgammon>) =
            games.values().map(|x| x.0).partition(|state| state.player == player_p1);

        let spinner_style =
            ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}").unwrap();
        let actions_pb = pb_play.add(
            ProgressBar::new(0)
                .with_style(spinner_style.clone())
        );
        actions_pb.enable_steady_tick(Duration::from_millis(200));
        actions_pb.set_message(format!("Calculating actions for player1: {:?}", player1.player_type));
        let actions_p1 = get_actions_for_player(&player1, &games_p1);
        actions_pb.set_message(format!("Calculating actions for player2: {:?}", player2.player_type));
        let actions_p2 = get_actions_for_player(&player2, &games_p2);
        actions_pb.set_message("playing moves...");

        let actions_and_games = actions_p1
            .iter()
            .zip(games_p1)
            .chain(actions_p2.iter().zip(games_p2));

        let mut games_to_remove = vec![];
        round_count += 1;
        for (action, game) in actions_and_games {
            // The key of the game on the games map given on creation
            let initial_idx = game.id;
            let (game_mut, curr_game) = games.get_mut(&initial_idx).unwrap();

            let player_type = if game_mut.player == -1 {player1.player_type.clone()} else {player2.player_type.clone()};
            curr_game.turns.push(Turn { roll: game_mut.roll, player: player_type, action: action.clone() });

            if action.is_empty() {
                game_mut.skip_turn();
                continue;
            }
            assert!(game_mut.get_valid_moves_len_always_2().contains(action));

            game_mut.apply_move(action);
            if let Some(winner) = Backgammon::check_win_without_player(game_mut.board) {
                if winner == player_p1 {
                    curr_game.winner = player1.player_type.clone();
                    wins_p1 += 1.
                } else {
                    curr_game.winner = player2.player_type.clone();
                }
                games_to_remove.push(initial_idx);
            }
            if round_count >= round_limit {
                games_to_remove.push(initial_idx);
                let choices = vec![-1, 1];
                let rand_winner = choices.choose(&mut rand::thread_rng()).unwrap();
                if *rand_winner == player_p1 {
                    curr_game.winner = player1.player_type.clone();
                    wins_p1 += 1.
                } else {
                    curr_game.winner = player2.player_type.clone();
                }
            }
        }
        for game_idx in games_to_remove {
            let (_, game) = games.remove(&game_idx).unwrap();
            games_played.push(game);
        }
    }
    let winrate = wins_p1 / num_games as f64;
    let wins_p1 = wins_p1 as usize;
    PlayResult {
        player1: player1.player_type,
        player2: player2.player_type,
        wins_p1,
        wins_p2: num_games - wins_p1,
        winrate,
        n_games: num_games,
        games: games_played
    }
}

fn get_actions_for_player(player: &Player, games: &[Backgammon]) -> Vec<Actions> {
    if games.is_empty() {
        return vec![];
    }

    match player.player_type {
        Agent::Model => {
            let az = player.model.as_ref().unwrap();
            let mut store = NodeStore::new();
            alpha_mcts_parallel(&mut store, games, &az.model, Some(ProgressBar::hidden()));
            let roots = store.get_root_nodes();
            let prob_tensor = get_prob_tensor_parallel(&roots)
                .pow_(1.0 / az.config.temperature)
                .to_device(tch::Device::Cpu);

            roots
                .iter()
                .enumerate()
                .map(|(processed_idx, &root)| {
                    let curr_prob_tensor = prob_tensor.get(processed_idx as i64);

                    // If prob tensor of the current state is all zeros then skip turn, has_children check just in case
                    if !curr_prob_tensor.sum(None).is_nonzero() || root.children.is_empty() {
                        return vec![];
                    }

                    // Select an action from probabilities
                    let selected_action = AlphaZero::weighted_select_tensor_idx(&curr_prob_tensor);
                    // Decode and play selected action
                    root.state.decode(selected_action as u32)
                })
                .collect_vec()
        }
        Agent::Mcts => games
            .par_iter()
            .map(|game| mct_search(*game, game.player))
            .collect(),
        Agent::Random => games
            .par_iter()
            .map(|game| {
                let valid_moves = game.get_valid_moves_len_always_2();
                match valid_moves.choose(&mut rand::thread_rng()) {
                    Some(valid_move) => valid_move.clone(),
                    None => vec![],
                }
            })
            .collect(),
        Agent::None => unreachable!(),
    }
}
