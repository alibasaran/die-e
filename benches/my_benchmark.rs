use criterion::{criterion_group, criterion_main, Criterion};
use die_e::{
    backgammon::backgammon_logic::Backgammon,
    constants::{DEVICE},
    mcts::{node_store::NodeStore, node::Node, utils::{turn_policy_to_probs_tensor, turn_policy_to_probs}},
};
use itertools::Itertools;
use rand::{thread_rng, Rng};
use tch::{Device, Kind, Tensor};

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("alpha_mcts_parallel");
    // group.sample_size(10);
    // group.bench_function("alpha_mcts_parallel", |b| b.iter(|| {
    //     let mut store = NodeStore::new();
    //     let mut bg = Backgammon::new();
    //     bg.roll_die();
    //     let states = vec![bg; N_SELF_PLAY_BATCHES];
    //     let net = ResNet::default();
    //     alpha_mcts_parallel(&mut store, &states, &net, None);
    // }));
    let mut rng = thread_rng();

    // group.bench_function("device comparison ones mps", |b| b.iter(|| {
    //     let ones = Tensor::ones(5000, (Kind::Float, Device::Mps));
    //     ones * rng.gen_range(0..100000)
    // }));

    // group.bench_function("device comparison ones cpu", |b| b.iter(|| {
    //     let ones = Tensor::ones(5000, (Kind::Float, Device::Cpu));
    //     ones * rng.gen_range(0..100000)
    // }));

    // group.bench_function("device comparison rand mps", |b| b.iter(|| {
    //     let rand1 = Tensor::rand(5000, (Kind::Float, Device::Mps));
    //     let rand2 = Tensor::rand([5000, 5000], (Kind::Float, Device::Mps));
    //     rand1.matmul(&rand2)
    // }));

    // group.bench_function("device comparison rand cpu", |b| b.iter(|| {
    //     let rand1 = Tensor::rand(5000, (Kind::Float, Device::Cpu));
    //     let rand2 = Tensor::rand([5000, 5000], (Kind::Float, Device::Cpu));
    //     rand1.matmul(&rand2)
    // }));
    // 2.86s
    // group.bench_function("device comparison model cpu", |b| {
    //     b.iter(|| {
    //         let net = ResNet::with_device(Device::Cpu);
    //         let rand2 = Tensor::rand([512, 6, 4, 6], (Kind::Float, Device::Cpu));
    //         net.forward_t(&rand2, false)
    //     })
    // });

    // group.bench_function("type comparison model mps", |b| {
    //     b.iter(|| {
    //         let net = ResNet::default();
    //         let rand2 = Tensor::rand([512, 6, 4, 6], (Kind::Float, Device::Mps));
    //         net.forward_t(&rand2, false)
    //     })
    // });

    // group.bench_function("type comparison half model mps", |b| {
    //     b.iter(|| {
    //         let net = ResNet::default();
    //         let rand2 = Tensor::rand([2048, 6, 4, 6], (Kind::Half, Device::Mps));
    //         net.forward_policy(&rand2, false)
    //     })
    // });

    group.bench_function("turn policy to probs tensor", |b| {
        b.iter(|| {
            let mut state = Backgammon::default();
            state.roll_die();
            let node = Node::new(state, 0, None, None, 0.);
            let policy = Tensor::rand(1352, (Kind::Float, *DEVICE));
            turn_policy_to_probs_tensor(&policy, &node)
        })
    });

    group.bench_function("turn policy to probs vec", |b| {
        b.iter(|| {
            let mut state = Backgammon::default();
            state.roll_die();
            let node = Node::new(state, 0, None, None, 0.);
            let policy = Tensor::rand(1352, (Kind::Float, *DEVICE));
            turn_policy_to_probs(&policy, &node)
        })
    });

    group.bench_function("alpha expand tensor", |b| {
        b.iter(|| {
            let mut state = Backgammon::default();
            state.roll_die();
            let mut store = NodeStore::new();
            store.add_node(state, None, None, Some(state.roll), 0.);
            let mut node = store.get_node(0);
            let policy = Tensor::rand(1352, (Kind::Float, Device::Cpu));
            node.alpha_expand_tensor(&mut store, &policy);
        })
    });

    group.bench_function("alpha expand vec", |b| {
        b.iter(|| {
            let mut state = Backgammon::default();
            state.roll_die();
            let mut store = NodeStore::new();
            store.add_node(state, None, None, Some(state.roll), 0.);
            let policy = (0..1352).map(|_| rng.gen_range(0..20) as f32).collect_vec();
            let mut node = store.get_node(0);
            node.alpha_expand(&mut store, policy);
        })
    });

    group.finish()
}

criterion_group!(benches, bench);
criterion_main!(benches);
