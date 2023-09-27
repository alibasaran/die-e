use config::{Config, ConfigError};

#[macro_use]
extern crate lazy_static;

pub mod alphazero;
pub mod backgammon;
pub mod mcts;
pub mod versus;

/*

Constants used through all modules
*/
pub mod constants {
    use tch::Device;
    
    // https://jonathan-laurent.github.io/AlphaZero.jl/stable/reference/params/#AlphaZero.MctsParams
    pub const ACTION_SPACE_SIZE: i64 = 1352;

    pub const DEFAULT_TYPE: tch::kind::Kind = tch::Kind::Float;

    lazy_static! {
        pub static ref DEVICE: Device = {
            if tch::utils::has_mps() {
                Device::Mps
            } else {
                Device::cuda_if_available()
            }
        };
    }
}

#[derive(Debug, Clone)]
pub struct MctsConfig {
    iterations: usize,
    c: f32,
    simulate_round_limit: usize,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32
}

impl MctsConfig {
    pub fn from_config(conf: &Config) -> Result<Self, ConfigError> {
        Ok(MctsConfig {
            iterations: conf.get_int("iterations")? as usize,
            c: conf.get_float("exploration_const")? as f32,
            simulate_round_limit: conf.get_int("simulate_round_limit")? as usize,
            dirichlet_alpha: conf.get_float("DIRICHLET_ALPHA")? as f32,
            dirichlet_epsilon: conf.get_float("DIRICHLET_EPSILON")? as f32,
        })
    }
}
