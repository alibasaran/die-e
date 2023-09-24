use std::{path::Path, fs::{File, self}, io::{Read, Write}};
use rand::{seq::SliceRandom, thread_rng};
use serde::{Serialize, Deserialize};
use nanoid::nanoid;

use crate::{backgammon::Backgammon, mcts::{simple_mcts::mct_search, utils::random_play, alpha_mcts::alpha_mcts}, alphazero::{nnet::ResNet, alphazero::{AlphaZeroConfig, AlphaZero}}};


/*
Small functions to pit basic agents into eachother and display how a game went in the terminal
TODOs:
- Add a better Backgammon to string function.
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

pub fn print_game(directory: &str, filename: &str, wait_user_input: bool) {
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
        current_state.display_board();

        if wait_user_input {
            println!("Press Enter to continue...");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).expect("Failed to read input");
        }
    }
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

pub fn play_mcts_vs_random() -> Game {
    // Shuffle and select agents
    let mut shuffled_agents = vec![Agent::Random, Agent::Mcts];
    let initial_state = Backgammon::new();
    shuffled_agents.shuffle(&mut thread_rng());
    let (player1, player2) = (shuffled_agents[0].clone(), shuffled_agents[1].clone());
    
    let mut game: Game = Game::new(player1.clone(), player2.clone(), initial_state);
    let mut current_state = initial_state;
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
            Agent::Mcts => mct_search(current_state, mcts_idx),
            Agent::Random => random_play(&current_state),
            _ => unreachable!(),
        };
        println!("\tAction: {:?}", action);
        // Push turn into log
        game.turns.push(Turn { roll, player: curr_player.clone(), action: action.clone() });
        // Update current board
        current_state.apply_move(&action);
        current_state.is_valid();
        println!("New board");
        current_state.display_board();
        // Switch Agents
        curr_player = if curr_player == Agent::Mcts {Agent::Random} else {Agent::Mcts}
    }
    let winner = Backgammon::check_win_without_player(current_state.board).unwrap();
    game.winner = shuffled_agents[((winner + 1)/2) as usize].clone();
    game
}

pub fn play_mcts_vs_model(az: &AlphaZero) -> Game {
    // Shuffle and select agents
    let mut shuffled_agents = vec![Agent::Model, Agent::Mcts];
    let initial_state = Backgammon::new();
    shuffled_agents.shuffle(&mut thread_rng());
    let (player1, player2) = (shuffled_agents[0].clone(), shuffled_agents[1].clone());
    
    let mut game: Game = Game::new(player1.clone(), player2.clone(), initial_state);
    let mut current_state = initial_state;
    current_state.roll_die();
    let mut curr_player = player1.clone();
    let mcts_idx = if curr_player == Agent::Mcts {-1} else {1};

    println!("Player 1: {:?}, Player 2: {:?}", player1, player2);

    while Backgammon::check_win_without_player(current_state.board).is_none() {
        curr_player = if current_state.player == -1 {player1.clone()} else {player2.clone()};
        let roll = current_state.roll;
        println!("Current player: {:?}", curr_player);
        println!("\tRolled die: {:?}", current_state.roll);

        // Select action depending on the current agent
        let action: Vec<(i8, i8)> = match curr_player {
            Agent::Mcts => mct_search(current_state, mcts_idx),
            Agent::Model => az.get_next_move_for_state(&current_state),
            _ => unreachable!(),
        };
        println!("\tAction: {:?}", action);
        // Push turn into log
        game.turns.push(Turn { roll, player: curr_player.clone(), action: action.clone() });
        // Update current board
        current_state.apply_move(&action);
        println!("New board");
        current_state.display_board();
    }
    let winner = Backgammon::check_win_without_player(current_state.board).unwrap();
    game.winner = shuffled_agents[((winner + 1)/2) as usize].clone();
    game
}

pub fn play_random_vs_random() -> Game {
    // Shuffle and select agents
    let mut shuffled_agents = vec![Agent::Random, Agent::Random];
    let initial_state = Backgammon::new();
    shuffled_agents.shuffle(&mut thread_rng());
    let (player1, player2) = (shuffled_agents[0].clone(), shuffled_agents[1].clone());
    
    let mut game: Game = Game::new(player1.clone(), player2.clone(), initial_state);
    let mut current_state = initial_state;
    let mut curr_player = player1.clone();

    println!("Player 1: {:?}, Player 2: {:?}", player1, player2);

    while Backgammon::check_win_without_player(current_state.board).is_none() {
        // Roll die
        let roll = current_state.roll_die();
        println!("Current player: {:?}", curr_player);
        println!("\tRolled die: {:?}", current_state.roll);

        let action: Vec<(i8, i8)> = random_play(&current_state);

        println!("\tAction: {:?}", action);
        // Push turn into log
        game.turns.push(Turn { roll, player: curr_player.clone(), action: action.clone() });
        // Update current board
        current_state.apply_move(&action);
        current_state.is_valid();
        println!("New board");
        current_state.display_board();
    }
    let winner = Backgammon::check_win_without_player(current_state.board).unwrap();
    game.winner = shuffled_agents[((winner + 1)/2) as usize].clone();
    game
}