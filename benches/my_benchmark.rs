use criterion::{criterion_group, criterion_main, Criterion};
use die_e::{
    alphazero::nnet::ResNet,
    backgammon::Backgammon,
    constants::N_SELF_PLAY_BATCHES,
    mcts::{alpha_mcts::alpha_mcts_parallel, node_store::NodeStore},
};
use rand::{thread_rng, Rng};
use tch::{nn::VarStore, Device, Kind, Tensor};

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
    group.bench_function("device comparison model cpu", |b| {
        b.iter(|| {
            let net = ResNet::with_device(Device::Cpu);
            let rand2 = Tensor::rand([512, 6, 4, 6], (Kind::Float, Device::Cpu));
            net.forward_t(&rand2, false)
        })
    });

    group.bench_function("type comparison model mps", |b| {
        b.iter(|| {
            let net = ResNet::default();
            let rand2 = Tensor::rand([512, 6, 4, 6], (Kind::Float, Device::Mps));
            net.forward_t(&rand2, false)
        })
    });

    // group.bench_function("type comparison half model mps", |b| {
    //     b.iter(|| {
    //         let net = ResNet::default();
    //         let rand2 = Tensor::rand([2048, 6, 4, 6], (Kind::Half, Device::Mps));
    //         net.forward_policy(&rand2, false)
    //     })
    // });

    group.finish()
}

criterion_group!(benches, bench);
criterion_main!(benches);
