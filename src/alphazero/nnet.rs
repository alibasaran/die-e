use std::{default, ops::Add};

use arrayvec::ArrayVec;
use rayon::vec;
use tch::{
    nn::{self, ModuleT, VarStore},
    utils::has_mps,
    Tensor,
};

use crate::constants::DEVICE;

/*
Constants

retrieved from: https://www.chessprogramming.org/AlphaZero#Network_Architecture
*/

const FILTERS: i64 = 256;
const CONV_OUTPUT_SIZE: i64 = 24; // board height * board width
const N_RES_BLOCKS: usize = 19;

#[derive(Debug)]
pub struct ResBlock {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    bn1: nn::BatchNorm,
    bn2: nn::BatchNorm,
}

impl nn::ModuleT for ResBlock {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        xs.apply_t(&self.conv1, train)
            .apply_t(&self.bn1, train)
            .relu()
            .apply_t(&self.conv2, train)
            .apply_t(&self.bn2, train)
            .add(xs)
            .relu()
    }
}

impl ResBlock {
    fn new(root: &nn::Path, config: nn::ConvConfig) -> Self {
        ResBlock {
            conv1: nn::conv2d(root, FILTERS, FILTERS, 3, config),
            conv2: nn::conv2d(root, FILTERS, FILTERS, 3, config),
            bn1: nn::batch_norm2d(root, FILTERS, Default::default()),
            bn2: nn::batch_norm2d(root, FILTERS, Default::default()),
        }
    }
}

/*
TODO: Implement ResNet
*/
#[derive(Debug)]
pub struct ResNet {
    vs: nn::VarStore,
    init_block: nn::SequentialT,
    res_layer: nn::SequentialT,
    policy_head: nn::SequentialT,
    value_head: nn::SequentialT,
}

const INPUT_SIZE: i64 = 6; // arbitrary
const POLICY_OUTPUT_SIZE: i64 = 1352; // arbitrary

impl Default for ResNet {
    fn default() -> Self {
        let conv_config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        println!("*DEVICE: {:?}", *DEVICE);
        let vs = nn::VarStore::new(*DEVICE);
        let root = vs.root();

        let init_block = nn::seq_t()
            .add(nn::conv2d(&root, INPUT_SIZE, FILTERS, 3, conv_config))
            .add(nn::batch_norm2d(&root, FILTERS, Default::default()))
            .add_fn(Tensor::relu);

        let mut res_layer = nn::seq_t();
        for _ in 0..N_RES_BLOCKS {
            let res_block = ResBlock::new(&vs.root(), conv_config);
            res_layer = res_layer.add(res_block);
        }

        let policy_head = nn::seq_t()
            .add(nn::conv2d(&root, FILTERS, 2, 3, conv_config))
            .add(nn::batch_norm2d(&root, 2, Default::default()))
            .add_fn(Tensor::relu)
            .add_fn(|x| x.flatten(1, -1))
            .add(nn::linear(
                &root,
                2 * CONV_OUTPUT_SIZE, /* conv output size */
                POLICY_OUTPUT_SIZE,
                Default::default(),
            ));

        let value_head = nn::seq_t()
            .add(nn::conv2d(&root, FILTERS, 1, 3, conv_config))
            .add(nn::batch_norm2d(&root, 1, Default::default()))
            .add_fn(Tensor::relu)
            .add_fn(|x| x.flatten(1, -1))
            .add(nn::linear(
                root,
                CONV_OUTPUT_SIZE, /* conv output size */
                1,
                Default::default(),
            ))
            .add_fn(Tensor::tanh);

        ResNet {
            vs,
            init_block,
            res_layer,
            policy_head,
            value_head,
        }
    }
}

impl ResNet {
    pub fn new(vs: VarStore) -> Self {
        ResNet {
            vs,
            ..Default::default()
        }
    }

    pub fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        let new_x = xs
            .apply_t(&self.init_block, train)
            .apply_t(&self.res_layer, train);

        let policy = new_x.apply_t(&self.policy_head, train).to_device_(
            *DEVICE,
            tch::Kind::Float,
            false,
            false,
        );

        let value = new_x.apply_t(&self.value_head, train).to_device_(
            *DEVICE,
            tch::Kind::Float,
            false,
            false,
        );

        (policy, value)
    }
}
