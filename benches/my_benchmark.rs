use criterion::{criterion_group, criterion_main, Criterion};
use die_e::{mcts::{alpha_mcts::alpha_mcts_parallel, node_store::NodeStore}, backgammon::Backgammon, alphazero::nnet::ResNet, constants::N_SELF_PLAY_BATCHES};

fn bench(c: &mut Criterion) { 
    let mut group = c.benchmark_group("alpha_mcts_parallel");   
    group.sample_size(10);
    group.bench_function("alpha_mcts_parallel", |b| b.iter(|| {
        let mut store = NodeStore::new();
        let mut bg = Backgammon::new();
        bg.roll_die();
        let states = vec![bg; N_SELF_PLAY_BATCHES];
        let net = ResNet::default();
        alpha_mcts_parallel(&mut store, &states, &net);
    }));
    group.finish()
}


criterion_group!(benches, bench);
criterion_main!(benches);
