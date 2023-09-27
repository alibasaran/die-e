use std::{path::Path, fs::{File, self}, io::{Read, Write}, time::Duration, collections::HashMap, fmt};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use itertools::Itertools;
use rand::seq::SliceRandom;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde::{Serialize, Deserialize};
use nanoid::nanoid;

use crate::{backgammon::backgammon_logic::{Backgammon, Actions}, mcts::{simple_mcts::mct_search, utils::get_prob_tensor_parallel, alpha_mcts::alpha_mcts_parallel, node_store::NodeStore}, alphazero::{alphazero::{AlphaZero}}, MctsConfig};


/*
Small functions to pit basic agents into eachother and display how a game went in the terminal
*/


#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Agent {
    Random, Mcts, Model, None
}
#[derive(Serialize, Deserialize, Debug)]
pub struct Turn {
    pub roll: (u8, u8),
    pub action: Vec<(i8, i8)>,
    pub player: Agent
}
#[derive(Serialize, Deserialize, Debug)]
pub struct Game {
    id: String,
    player1: Agent,
    player2: Agent,
    pub turns: Vec<Turn>,
    pub winner: Agent,
    initial_state: Backgammon 
}

impl Game {
    pub fn new(player1: Agent, player2: Agent, state: Backgammon)-> Game {
        Game {
            id: nanoid!(),
            player1,
            player2,
            turns: vec![],
            winner: Agent::None,
            initial_state: state,
        }
    }
}

pub fn save_game(game: &Game, game_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(game_path).join(format!("{}.json", &game.id));
    let file = File::create(path)?;

    let serialized = serde_json::to_string_pretty(game)?;
    let mut writer = std::io::BufWriter::new(file);
    writer.write_all(serialized.as_bytes())?;

    Ok(())
}

pub fn load_game(directory: &str, filename: &str) -> Result<Game, Box<dyn std::error::Error>> {
    let path = Path::new(directory).join(filename);
    let file = File::open(path)?;

    let mut contents = String::new();
    let mut reader = std::io::BufReader::new(file);
    reader.read_to_string(&mut contents)?;

    let game: Game = serde_json::from_str(&contents)?;

    Ok(game)
}

pub fn print_game(directory: &str, filename: &str, wait_user_input: bool) -> Result<(), Box<dyn std::error::Error>> {
    let game = match load_game(directory, filename) {
        Ok(game) => game,
        Err(e) => panic!("Failed to load the game: {:?}", e),
    };
    println!("Game ID: {}", game.id);
    println!("Player 1: {:?}, Player 2: {:?}", game.player1, game.player2);
    println!("Game winner: {:?}", game.winner);

    println!("Initial State:");
    let mut current_state = game.initial_state;
    current_state.display_board();

    for turn in game.turns {
        println!("Player: {:?}", turn.player);
        println!("Roll: {:?}", turn.roll);
        current_state.roll = turn.roll;
        println!("Action: {:?}", turn.action);
        current_state.apply_move(&turn.action);
        println!("State after action has been played:");
        current_state.to_pretty_str();

        if wait_user_input {
            println!("Press Enter to continue...");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).expect("Failed to read input");
        }
    }

    Ok(())
}

pub fn load_all_games(directory: &str) -> Result<Vec<Game>, Box<dyn std::error::Error>> {
    let mut games = Vec::new();

    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.path().is_file() {
            let file_name = entry.file_name().into_string().unwrap();
            if file_name.ends_with(".json") {
                let game = load_game(directory, &file_name)?;
                games.push(game);
            }
        }
    }

    Ok(games)
}

pub fn print_all_game_winners() {
    let games_directory = "./games/mcts_vs_random";
    if let Ok(games) = load_all_games(games_directory) {
        println!("Loaded {} games successfully!\n", games.len());
        for (_, game) in games.iter().enumerate() {
            println!("Game {}:", &game.id);
            println!("\t Winner: {:?}", game.winner)
        }
    } else {
        eprintln!("Error loading games from directory: {}", games_directory);
    }
}

pub fn print_out_game(directory: &str, filename: &str) {
    let game = load_game(directory, filename).unwrap();
    let mut curr_state = game.initial_state;
    println!("Player1: {:?} Player 2: {:?}", game.player1, game.player2);
    println!("Turns played: {}", &game.turns.len());
    for (idx, turn) in game.turns.iter().enumerate() {
        let _player = if idx % 2 == 0 {-1} else {1};
        println!("[{}] {:?}", idx, turn);
        curr_state.apply_move(&turn.action);
        curr_state.display_board();
    }
    println!("Winner: {:?}", game.winner)
}

pub struct Player {
    pub player_type: Agent,
    pub model: Option<AlphaZero>,
}

#[derive(Debug)]
pub struct PlayResult {
    player1: Agent,
    player2: Agent,
    wins_p1: usize,
    wins_p2: usize,
    n_games: usize,
    winrate: f64,
    pub games: Vec<Game>,
}

impl fmt::Display for PlayResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Player 1: {:?}", self.player1)?;
        writeln!(f, "Player 2: {:?}", self.player2)?;
        writeln!(f, "Wins Player 1: {}", self.wins_p1)?;
        writeln!(f, "Wins Player 2: {}", self.wins_p2)?;
        writeln!(f, "Number of Games: {}", self.n_games)?;
        writeln!(f, "Winrate: {:.2}%", self.winrate)?;
        Ok(())
    }
}

impl Player {
    pub fn new(player_type: Agent, model: Option<AlphaZero>) -> Self {
        Player { player_type, model}
    }
}

pub fn play(player1: Player, player2: Player, mcts_config: &MctsConfig) -> PlayResult {
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
        let actions_p1 = get_actions_for_player(&player1, &games_p1, mcts_config);
        actions_pb.set_message(format!("Calculating actions for player2: {:?}", player2.player_type));
        let actions_p2 = get_actions_for_player(&player2, &games_p2, mcts_config);
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

fn get_actions_for_player(player: &Player, games: &[Backgammon], mcts_config: &MctsConfig) -> Vec<Actions> {
    if games.is_empty() {
        return vec![];
    }

    match player.player_type {
        Agent::Model => {
            let az = player.model.as_ref().unwrap();
            let mut store = NodeStore::new();
            alpha_mcts_parallel(&mut store, games, &az.model, mcts_config, Some(ProgressBar::hidden()));
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
            .map(|game| mct_search(*game, game.player, mcts_config))
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