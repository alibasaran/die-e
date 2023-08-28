use alphazero::encoding::encode;
use backgammon::Actions;
pub use backgammon::Backgammon;

pub mod backgammon;
pub mod mcts;
pub mod alphazero;

use mcts::{mct_search, random_play};
use serde::{Deserialize, Serialize};
use nanoid::nanoid;
use tch::{Tensor, Kind};
// use serde::{Serialize, Deserialize};
use std::fs::{File, self};
use std::io::{Write, Read};
use std::ops::{Div, Mul};
use std::path::Path;
use std::vec;
use rand::thread_rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::alphazero::alphazero::{AlphaZero, AlphaZeroConfig};
use crate::alphazero::encoding::decode;
use crate::mcts::alpha_mcts;
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
enum Agent {
    Random, Mcts, None
}
#[derive(Serialize, Deserialize, Debug)]
struct Turn {
    roll: (u8, u8),
    action: Vec<(i8, i8)>,
    player: Agent
}
#[derive(Serialize, Deserialize, Debug)]
struct Game {
    id: String,
    player1: Agent,
    player2: Agent,
    turns: Vec<Turn>,
    winner: Agent,
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

fn save_game(game: &Game) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("./games/mcts_vs_random").join(format!("{}.json", &game.id));
    let file = File::create(path)?;

    let serialized = serde_json::to_string_pretty(game)?;
    let mut writer = std::io::BufWriter::new(file);
    writer.write_all(serialized.as_bytes())?;

    Ok(())
}

fn load_game(directory: &str, filename: &str) -> Result<Game, Box<dyn std::error::Error>> {
    let path = Path::new(directory).join(filename);
    let file = File::open(path)?;

    let mut contents = String::new();
    let mut reader = std::io::BufReader::new(file);
    reader.read_to_string(&mut contents)?;

    let game: Game = serde_json::from_str(&contents)?;

    Ok(game)
}

fn load_all_games(directory: &str) -> Result<Vec<Game>, Box<dyn std::error::Error>> {
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

fn print_all_game_winners() {
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

fn print_out_game(directory: &str, filename: &str) {
    let game = load_game(directory, filename).unwrap();
    let mut curr_state = game.initial_state;
    println!("Player1: {:?} Player 2: {:?}", game.player1, game.player2);
    println!("Turns played: {}", &game.turns.len());
    for (idx, turn) in game.turns.iter().enumerate() {
        let player = if idx % 2 == 0 {-1} else {1};
        println!("[{}] {:?}", idx, turn);
        curr_state.apply_move(&turn.action, player);
        curr_state.display_board();
    }
    println!("Winner: {:?}", game.winner)
}

/*
nR8X1p-18S3r8i1N0z9dr
XPyhpUHzsEQNV50fZiAku
i28yVaNn7ezdeHXDg6b6t
odISHyAxLq_akRO68vQY0
CWYa2Xvl5A8rhaF1JZIx4
lFgDY450qEWIMhf-g3P6r
D6-KLtFq2cDHc5Glk330X
O8xGK3vYZPTeBamSkooJx
DHDEy5XzhXOuCM240gGrI
dwTqcyGzRNC43yX-LewvO
1piHmxc-Ktz7Kd1645BwV
KmlNb3QyzBld6rDK6Y6dX
PL_s4k0qocJRM6gut2-eA
hCr0-QnW98zHUdBCniess
czHFVB_iyetkyykxn1rcJ
0XNaLo3BTCcgOVpMzMtTy
d6aGH6_F9egx9OpBISBHR
*/

// fn main() {
//     // print_all_game_winners();
//     // Set rayon to use x threads
//     rayon::ThreadPoolBuilder::new().num_threads(5).build_global().unwrap();
//     (0..100).into_par_iter().for_each(|_| {
//         let game = play_mcts_vs_random();
//         dbg!("{}", &game);
//         let _ = save_game(&game);
//     });
// }

fn main() {
    let config = AlphaZeroConfig {
        learn_iterations: 1,
        self_play_iterations: 4,
        batch_size: 2,
        num_epochs: 1,
    };
    let mut az = AlphaZero::new(config);
    az.learn();
}

fn play_games_and_save(iterations: usize) {
    unimplemented!()
}

fn play_mcts_vs_random() -> Game {
    // Shuffle and select agents
    let mut shuffled_agents = vec![Agent::Random, Agent::Mcts];
    let initial_state = Backgammon::new();
    shuffled_agents.shuffle(&mut thread_rng());
    let (player1, player2) = (shuffled_agents[0].clone(), shuffled_agents[1].clone());
    
    let mut game: Game = Game::new(player1.clone(), player2.clone(), initial_state.clone());
    let mut current_state = initial_state.clone();
    let mut curr_player = player1.clone();
    let mcts_idx = if curr_player == Agent::Mcts {-1} else {1};

    println!("Player 1: {:?}, Player 2: {:?}", player1, player2);

    while Backgammon::check_win_without_player(current_state.board).is_none() {
        // Roll die
        let roll = current_state.roll_die();
        println!("Current player: {:?}", curr_player);
        println!("\tRolled die: {:?}", current_state.roll);

        // Select action depending on the current agent
        let action: Vec<(i8, i8)> = match curr_player {
            Agent::Mcts => mct_search(current_state.clone(), mcts_idx),
            Agent::Random => random_play(&current_state, -mcts_idx),
            Agent::None => unreachable!(),
        };
        println!("\tAction: {:?}", action);
        // Push turn into log
        game.turns.push(Turn { roll, player: curr_player.clone(), action: action.clone() });
        // Update current board
        current_state.board = match curr_player {
            Agent::Mcts => Backgammon::get_next_state(current_state.board, &action, mcts_idx),
            Agent::Random => Backgammon::get_next_state(current_state.board, &action, -mcts_idx),
            Agent::None => unreachable!()
        };
        println!("New board");
        current_state.display_board();
        // Switch Agents
        curr_player = if curr_player == Agent::Mcts {Agent::Random} else {Agent::Mcts}
    }
    let winner = Backgammon::check_win_without_player(current_state.board).unwrap();
    game.winner = shuffled_agents[((winner + 1)/2) as usize].clone();
    game
}
