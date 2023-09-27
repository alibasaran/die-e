use std::{
    collections::HashMap,
    time::Duration, path::{Path, PathBuf}, fs, io,
};

use config::Config;
use die_e::{
    backgammon::backgammon_logic::Backgammon,
    constants::DEVICE, alphazero::alphazero::{AlphaZero, AlphaZeroConfig}, MctsConfig,
};
use itertools::Itertools;
use tch::{Tensor, nn::VarStore};

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
        // Path to output model after training
        #[arg(short, long)]
        model_save_path: Option<PathBuf>,
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
        .add_source(config::File::new(config_path.to_str().unwrap(), config::FileFormat::Toml));

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
            let mut az = AlphaZero::from_config(model_path, &config);
            az.learn_parallel();
        },
        Commands::Play {  } => todo!(),
        Commands::Train { model_path, model_save_path,  run_id, learn, self_play } => {
            println!("Starting training process");
            // Load training data
            let data_path = match (run_id, learn, self_play) {
                (None, None, None) => String::from("./data"),
                (Some(id), None, None) => format!("./data/run-{}", id),
                (Some(id), Some(learn_id), None) => format!("./data/run-{}/lrn-{}", id, learn_id),
                (Some(id), Some(learn_id), Some(sp_id)) => format!("./data/run-{}/lrn-{}/sp-{}", id, learn_id, sp_id),
                _ => panic!("the request for the training data is incorrect, run die-e learn --help for more info")
            };
            let data_path = Path::new(&data_path);
            if !data_path.exists() {
                panic!("[TRAIN] the specified path {} does not exist!", data_path.to_str().unwrap());
            }
            let mut paths_to_load = vec![];
            let _ = get_all_paths_rec(data_path, &mut paths_to_load);
            let mut training_data = paths_to_load.iter().flat_map(|p| 
                AlphaZero::load_training_data(p.as_path())
            ).collect_vec();

            println!("Total memory fragments: {}", &training_data.len());
            
            // Create AZ instance for training
            let mut az = AlphaZero::from_config(model_path, &config);

            // Train and save model
            az.train(&mut training_data);
            let final_out_path = model_save_path.unwrap_or(Path::new("./models/trained_model.ot").to_path_buf());
            match az.model.vs.save(&final_out_path) {
                Ok(_) => println!("Trained model saved successfully, saved to {}", final_out_path.to_str().unwrap()),
                Err(e) => panic!("unable to save trained model {}", e),
            }
        },
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
