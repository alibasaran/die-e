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

    pub const DIRICHLET_ALPHA: f32 = 0.3;
    pub const DIRICHLET_EPSILON: f32 = 0.25;

    pub const ACTION_SPACE_SIZE: i64 = 1352;
    pub const N_SELF_PLAY_BATCHES: usize = 10;

    lazy_static! {
        pub static ref DEVICE: Device = {
            // CPU improves the performans
            Device::Cpu
            // if tch::utils::has_mps() {
            //     Device::Mps
            // } else {
            //     Device::cuda_if_available()
            // }
        };
    }
}

pub struct MctsConfig {
    iterations: usize,
    c: f32,
    simulate_round_limit: usize,
}

pub const MCTS_CONFIG: MctsConfig = MctsConfig {
    iterations: 10,
    c: 1.0,
    // c: std::f32::consts::SQRT_2,
    simulate_round_limit: 100,
};
