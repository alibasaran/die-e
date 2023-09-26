use std::{
    collections::HashMap,
    time::Duration, path::{Path, PathBuf},
};

use config::Config;
use die_e::{
    backgammon::backgammon_logic::Backgammon,
    mcts::alpha_mcts::TimeLogger,
    constants::DEVICE, alphazero::alphazero::{AlphaZero, AlphaZeroConfig, AZ_CONFIG_KEYS},
};
use itertools::Itertools;
use tch::Tensor;

use clap::{Parser, Subcommand};

#[derive(Parser)]
struct Args {
    /// Sets a custom config file
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

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
    // TODO: Play
    Play {

    },
    Train {
        // Path of the model to train
        #[arg(short, long)]
        model_path: Option<PathBuf>,
        // The run id for the data, uses all data under run id to train
        #[arg(short, long)]
        run_id: Option<String>,
        // The idx of the learn iteration, run_id must also be given
        #[arg(short, long)]
        learn: Option<String>,
        // The idx of the self_play iteration, learn_idx must also be given
        #[arg(short, long)]
        self_play: Option<String>,
    }
}

fn main() {
    let args = Args::parse();
    // Load config
    let config_path = args.config.unwrap_or(
        PathBuf::from("./config")
    );
    let builder = Config::builder()
        .add_source(config::File::new(config_path.to_str().unwrap(), config::FileFormat::Json));

    let config = match builder.build() {
        Ok(config) => config,
        Err(e) => panic!("Unable to build config, caught error {}", e),
    };

    let n_cpus_in_device = num_cpus::get();
    let n_cpus = match args.n_cpus {
        Some(n) => if n_cpus_in_device > n {
            panic!("Value provided in n_cpus flag ({}) is larger than total cpus in the device ({})!", n, n_cpus_in_device)
        } else {n},
        None => n_cpus_in_device / 2,
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_cpus)
        .build_global()
        .unwrap();
    println!("Number of CPU's to use {}", n_cpus);

    match args.command {
        Commands::Learn { model_path } => {
            let mut az_config = match AlphaZeroConfig::from_config(&config){
                Ok(config) => config,
                Err(e) => panic!("Unable to load AlphaZero config, {}", e),
            };
            az_config.model_path = model_path;
            let mut az = AlphaZero::new(az_config);
            az.learn_parallel();
        },
        Commands::Play {  } => todo!(),
        Commands::Train { model_path, run_id, learn, self_play } => todo!(),
    }
}
