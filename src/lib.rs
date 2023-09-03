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
    pub const DIRICHLET_ALPHA: f32 = 0.02;
    pub const DIRICHLET_EPSILON: f32 = 0.25;

    pub const ACTION_SPACE_SIZE: i64 = 1352;
    pub const N_SELF_PLAY_BATCHES: usize = 2048;

    pub const DEFAULT_TYPE: tch::kind::Kind = tch::Kind::Float;

    lazy_static! {
        pub static ref DEVICE: Device = {
            // CPU improves the performance on short term if MPS warms up then it is faster
            // During development setting it to  Device::Cpu
            if tch::utils::has_mps() {
                Device::Mps
            } else {
                Device::cuda_if_available()
            }
            // Device::Cpu
        };
    }
}

#[derive(Debug)]
pub struct MctsConfig {
    iterations: usize,
    c: f32,
    simulate_round_limit: usize,
}

pub const MCTS_CONFIG: MctsConfig = MctsConfig {
    iterations: 400,
    c: 1.0,
    // c: std::f32::consts::SQRT_2,
    simulate_round_limit: 100,
};
