use die_e::{alphazero::{nnet::ResNet, alphazero::{AlphaZeroConfig, AlphaZero}}, mcts::{node_store::NodeStore, alpha_mcts::alpha_mcts_parallel}, backgammon::Backgammon, constants::N_SELF_PLAY_BATCHES};


fn main() {
    let config = AlphaZeroConfig {
        temperature: 1.,
        learn_iterations: 100,
        self_play_iterations: 1,
        batch_size: 2048,
        num_epochs: 2,
    };
    let mut az = AlphaZero::new(config);
    az.learn_parallel();
}
