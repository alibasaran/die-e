use die_e::alphazero::alphazero::{AlphaZeroConfig, AlphaZero};


fn main() {
    let config = AlphaZeroConfig {
        temperature: 1.25,
        learn_iterations: 1,
        self_play_iterations: 4,
        batch_size: 2,
        num_epochs: 1,
    };
    let mut az = AlphaZero::new(config);
    az.learn();
}
