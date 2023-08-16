pub use backgammon::Backgammon;

pub mod backgammon;
pub mod mcts;

use mcts::{mct_search, random_play};
use serde::{Deserialize, Serialize};
use nanoid::nanoid;
// use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::vec;
use rand::thread_rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::mcts::roll_die;
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

fn main() {
    // let game = Game::new(Agent::Mcts, Agent::Random, Backgammon::new());
    // let _ = save_game(&game);
    // Set rayon to use 3 threads
    rayon::ThreadPoolBuilder::new().num_threads(3).build_global().unwrap();
    (0..10).into_par_iter().for_each(|_| {
        let game = play_mcts_vs_random();
        dbg!("{}", &game);
        let _ = save_game(&game);
    });
}

fn old_main() {
    let mut bg = Backgammon::new();

    while Backgammon::check_win_without_player(bg.board).is_none() {
        let player_1_roll = roll_die();
        let player_1_action = mct_search(bg.clone(), -1, player_1_roll);

        let new_state = Backgammon::get_next_state(bg.board, 
            &player_1_action, -1);

        println!("Player 1, roll: {:?}, action: {:?}", player_1_roll, player_1_action);
        Backgammon::display_board(&new_state);

        bg.board = new_state;

        let player_2_roll = roll_die();
        let player_2_action = random_play(bg.board, 1, player_2_roll);

        let new_state = Backgammon::get_next_state(bg.board, 
            &player_2_action, 1);

        println!("Player 2, roll: {:?}, action: {:?}", player_2_roll, player_2_action);
        Backgammon::display_board(&new_state);

        bg.board = new_state;
    }

    let winner = Backgammon::check_win_without_player(bg.board).unwrap();
    println!("Winner is player {}", winner)
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
        let roll = roll_die();
        println!("Current player: {:?}", curr_player);
        println!("\tRolled die: {:?}", roll);

        // Select action depending on the current agent
        let action: Vec<(i8, i8)> = match curr_player {
            Agent::Mcts => mct_search(current_state.clone(), mcts_idx, roll),
            Agent::Random => random_play(current_state.board, -mcts_idx, roll),
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
        Backgammon::display_board(&current_state.board);
        // Switch Agents
        curr_player = if curr_player == Agent::Mcts {Agent::Random} else {Agent::Mcts}
    }
    let winner = Backgammon::check_win_without_player(current_state.board).unwrap();
    game.winner = shuffled_agents[((winner + 1)/2) as usize].clone();
    game
}
