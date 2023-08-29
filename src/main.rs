use die_e::{alphazero::{nnet::ResNet}, mcts::{node_store::NodeStore, alpha_mcts::alpha_mcts_parallel}, backgammon::Backgammon, constants::N_SELF_PLAY_BATCHES};


fn main() {
    // let config = AlphaZeroConfig {
    //     temperature: 1.25,
    //     learn_iterations: 1,
    //     self_play_iterations: 4,
    //     batch_size: 2,
    //     num_epochs: 1,
    // };
    // let mut az = AlphaZero::new(config);
    // az.learn();
    let mut store = NodeStore::new();
    let mut bg = Backgammon::new();
    bg.roll_die();
    let states = vec![bg; N_SELF_PLAY_BATCHES];
    let net = ResNet::default();
    alpha_mcts_parallel(&mut store, states, -1, &net, false);
}
