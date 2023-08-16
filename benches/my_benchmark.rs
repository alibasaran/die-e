use criterion::{black_box, criterion_group, criterion_main, Criterion};
use die_e::{mcts::Node, backgammon::Backgammon};

fn simulate_benchmark(c: &mut Criterion) {
    let mut node = Node::new(Backgammon::new(), 0, None, None, -1, (3, 1));
    c.bench_function("simulate from start", |b| b.iter(|| node.simulate(black_box(-1))));
}

criterion_group!(benches, simulate_benchmark);
criterion_main!(benches);
