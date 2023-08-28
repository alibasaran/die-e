pub mod node;
pub mod node_store;
pub mod mcts;
pub mod alpha_mcts;
pub mod utils;

pub struct MctsConfig {
    iterations: usize,
    c: f32,
    simulate_round_limit: usize,
}

pub const ACTION_SPACE_SIZE: i64 = 1352;

pub const MCTS_CONFIG: MctsConfig = MctsConfig {
    iterations: 5,
    c: 1.0,
    // c: std::f32::consts::SQRT_2,
    simulate_round_limit: 100,
};

