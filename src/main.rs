use core::panic;
use std::{
    path::{Path, PathBuf}, fs, io,
};

use config::Config;
use die_e::{
    alphazero::{alphazero::{AlphaZero, MemoryFragment}, nnet::ResNet}, MctsConfig, versus::{Agent, Player, play, save_game, print_game, PlayResult}, backgammon::backgammon_logic::Backgammon, tictactoe::TicTacToe, base::LearnableGame, constants::DEVICE
};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};


use clap::{Parser, Subcommand, ValueEnum};
use tch::nn::VarStore;

#[derive(Parser)]
struct Args {
    /// Sets a custom config file
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    #[clap(short, long, value_enum)]
    game: LearnableGames,

    // number of cpu's to use while learning, default is half of total cpus
    #[arg(short, long)]
    n_cpus: Option<usize>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Starts the learning process
    Learn {
        // path of the model
        #[arg(short, long)]
        model_path: Option<PathBuf>
    },
    Play {
        // Agent one's type, can be 'random', 'mcts', 'model'
        #[arg(short, long)]
        agent_one: Option<String>,
        // Path of model to play if agent one is of type 'model'
        #[arg(short, long)]
        model_path_one: Option<PathBuf>,
        // Agent two's type, can be 'random', 'mcts', 'model'
        #[arg(long)]
        agent_two: Option<String>,
        // Path of model to play if agent two is of type 'model'
        #[arg(long)]
        model_path_two: Option<PathBuf>,
        // Path to output game after playing
        #[arg(short, long)]
        output_path: Option<PathBuf>,
    },
    Train {
        // Path of the model to train
        #[arg(short, long)]
        model_path: Option<PathBuf>,
        // Path to output model after training
        #[arg(short, long)]
        out_path: Option<PathBuf>,
        // The run id for the data, uses all data under run id to train
        #[arg(short, long)]
        run_id: Option<String>,
        // The idx of the learn iteration, run_id must also be given
        #[arg(short, long)]
        learn: Option<String>,
        // The idx of the self_play iteration, learn_idx must also be given
        #[arg(short, long)]
        self_play: Option<String>,
    },
    Replay {
        // path of the game to load
        #[arg(short, long)]
        game_path: PathBuf
    }
}
#[derive(ValueEnum, Debug, Clone)]
enum LearnableGames {
    TicTacToe, Backgammon
}

fn main() {
    let kill = true;
    // let nnet1 = ResNet::from_path::<TicTacToe>(&PathBuf::from("./models/tictactoe/best_model.ot"));
    // let nnet2 = ResNet::from_path::<TicTacToe>(&PathBuf::from("./models/tictactoe/model_15.ot"));
    // let nnet2_vars = nnet2.vs.variables();
    // println!("All var names: {:?}", nnet2_vars.keys());
    // for (k, t) in nnet1.vs.variables().iter() {
    //     let nnet_var = nnet2_vars.get(k).unwrap();
    //     nnet_var.print();
    //     assert!(!t.isnan().sum(tch::Kind::Float).is_nonzero(), "Best model panic {}", k);
    //     assert!(!nnet_var.isnan().sum(tch::Kind::Float).is_nonzero(), "Model 1 panic {}", k);
    //     // println!("{}", t.equal(nnet_var));
    //     // if  {
    //     //     println!("Two tensors are equal, tensor name: {}", k);
    //     // }
    // }
    // let mut ttt = TicTacToe::new();
    // ttt.skip_turn(); // player 1
    // ttt.board = [
    //     -1, 0, 0,
    //     1, 1, 0, 
    //     -1, 0, 1
    // ];
    // for (k, t) in nnet1.vs.variables().iter() {
    //     t.print()
    // }
    // let (policy, eval) = nnet1.forward_t(&ttt.as_tensor().to_device(*DEVICE), false);
    // policy.print();
    // eval.print();
    // if kill {
    //     return;
    // }

    let args = Args::parse();

    // Load config
    let config_path = args.config.unwrap_or(
        PathBuf::from("./config")
    );
    let builder = Config::builder()
        .add_source(config::File::new(config_path.to_str().unwrap(), config::FileFormat::Toml));

    let config = match builder.build() {
        Ok(config) => config,
        Err(e) => panic!("Unable to build config, caught error {}", e),
    };

    let n_cpus_in_device = num_cpus::get();
    let n_cpus = match args.n_cpus {
        Some(n) => if n > n_cpus_in_device {
            panic!("Value provided in n_cpus flag ({}) is larger than total cpus in the device ({})!", n, n_cpus_in_device)
        } else {n},
        None => n_cpus_in_device / 2,
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_cpus)
        .build_global()
        .unwrap();
    println!("Number of CPU's to use {}", n_cpus);

    match args.game {
        LearnableGames::Backgammon => handle_command::<Backgammon>(args.command, &config),
        LearnableGames::TicTacToe => handle_command::<TicTacToe>(args.command, &config)
    }
}

fn handle_command<T: LearnableGame>(command: Commands, conf: &Config) {
    match command {
        Commands::Learn { model_path } => {
            let mut az = AlphaZero::from_config::<T>(model_path, conf);
            az.learn_parallel::<T>();
        },
        Commands::Play { agent_one, model_path_one, agent_two, model_path_two, output_path } => {
            let agent_one_type = match agent_one {
                Some(agent) => match agent.to_ascii_lowercase().as_str() {
                    "model" => Agent::Model,
                    "mcts" => Agent::Mcts,
                    "random" => Agent::Random,
                    _ => panic!("Incorrect specification for agent one's type.")
                }
                None => panic!("Must define a type for agent one.")
            };

            let agent_two_type = match agent_two {
                Some(agent) => match agent.to_ascii_lowercase().as_str() {
                    "model" => Agent::Model,
                    "mcts" => Agent::Mcts,
                    "random" => Agent::Random,
                    _ => panic!("Incorrect specification for agent two's type.")
                }
                None => panic!("Must define a type for agent two.")
            };

            let output_path = match output_path {
                Some(output) => output,
                None => panic!("No output path given.")
            };
            assert!(output_path.is_dir(), "Output path is not a directory or does not exist!");

            let model_one = model_path_one
                .map(|model_path| ResNet::from_path::<T>(&model_path));

            let model_two = model_path_two
                .map(|model_path| ResNet::from_path::<T>(&model_path));

            let player1 = Player{player_type: agent_one_type, model: model_one};
            let player2 = Player{player_type: agent_two_type, model: model_two};

            let temp = match conf.get_float("temperature") {
                Ok(temperature) => temperature,
                Err(e) => panic!("unable to load temperature value, check config.toml!, error: {}", e),
            };

            let play_result: PlayResult<T> = play::<T>(player1, player2, &MctsConfig::from_config(conf).unwrap(), temp);
            println!("{}\n Saving games...", play_result);
            for game in play_result.games {
                save_game(&game, output_path.to_str().unwrap()).unwrap()
            }
        },
        Commands::Train { model_path, out_path,  run_id, learn, self_play } => {
            println!("Starting training process");
            // Load training data
            let base_path_str = format!("./data/{}", T::name());
            let data_path = match (run_id, learn, self_play) {
                (None, None, None) => base_path_str,
                (Some(id), None, None) => format!("{}/run-{}", base_path_str, id),
                (Some(id), Some(learn_id), None) => format!("{}/run-{}/lrn-{}", base_path_str, id, learn_id),
                (Some(id), Some(learn_id), Some(sp_id)) => format!("{}/run-{}/lrn-{}/sp-{}", base_path_str, id, learn_id, sp_id),
                _ => panic!("the request for the training data is incorrect, run die-e learn --help for more info")
            };
            let data_path = Path::new(&data_path);
            if !data_path.exists() {
                panic!("[TRAIN] the specified path {} does not exist!", data_path.to_str().unwrap());
            }
            println!("Loading all data under {}", data_path.to_str().unwrap());
            let mut paths_to_load = vec![];
            let _ = get_all_paths_rec(data_path, &mut paths_to_load);
            let mut training_data: Vec<MemoryFragment> = paths_to_load.par_iter().flat_map(|p| 
                AlphaZero::load_training_data(p.as_path())
            ).collect();

            println!("Total memory fragments: {}", &training_data.len());
            
            // Create AZ instance for training
            let mut az = AlphaZero::from_config::<T>(model_path, conf);

            // Train and save model
            az.train(&mut training_data);
            let model_path_str = format!("./models/{}/trained_model.ot", T::name());
            let final_out_path = out_path.unwrap_or(Path::new(&model_path_str).to_path_buf());
            match az.model.vs.save(&final_out_path) {
                Ok(_) => println!("Trained model saved successfully, saved to {}", final_out_path.to_str().unwrap()),
                Err(e) => panic!("unable to save trained model {}", e),
            }
        },
        Commands::Replay { game_path } => {
            match print_game::<T>(game_path, true) {
                Ok(_) => (),
                Err(e) => panic!("unable to print game, {}", e),
            }
        }
    }

}

fn get_all_paths_rec(dir: &Path, res: &mut Vec<PathBuf>) -> io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.file_name().unwrap().to_str().unwrap().contains("sp") {
                res.push(path);
            } else {
                get_all_paths_rec(&path, res)?;
            }
        }
    }
    Ok(())
}
