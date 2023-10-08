use std::{ops::Add, path::PathBuf};

use tch::{
    nn::{self, VarStore},
    Tensor,
};

use crate::{constants::DEVICE, base::LearnableGame};

/*
Constants

retrieved from: https://www.chessprogramming.org/AlphaZero#Network_Architecture
*/

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
    fn new(root: &nn::Path, config: nn::ConvConfig, num_filters: i64) -> Self {
        ResBlock {
            conv1: nn::conv2d(root, num_filters, num_filters, 3, config),
            conv2: nn::conv2d(root, num_filters, num_filters, 3, config),
            bn1: nn::batch_norm2d(root, num_filters, Default::default()),
            bn2: nn::batch_norm2d(root, num_filters, Default::default()),
        }
    }
}

#[derive(Debug)]
pub struct ResNet {
    pub vs: nn::VarStore,
    init_block: nn::SequentialT,
    res_layer: nn::SequentialT,
    policy_head: nn::SequentialT,
    value_head: nn::SequentialT,
}

impl ResNet {
    pub fn new<T: LearnableGame>(vs: VarStore) -> Self {
        let conv_config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let root = vs.root();

        let init_block = nn::seq_t()
            .add(nn::conv2d(&root, T::N_INPUT_CHANNELS, T::N_FILTERS, 3, conv_config))
            .add(nn::batch_norm2d(&root, T::N_FILTERS, Default::default()))
            .add_fn(Tensor::relu);

        let mut res_layer = nn::seq_t();
        for _ in 0..T::N_RES_BLOCKS {
            let res_block = ResBlock::new(&vs.root(), conv_config, T::N_FILTERS);
            res_layer = res_layer.add(res_block);
        }

        let policy_head = nn::seq_t()
            .add(nn::conv2d(&root, T::N_FILTERS, 32, 3, conv_config))
            .add(nn::batch_norm2d(&root, 32, Default::default()))
            .add_fn(Tensor::relu)
            .add_fn(|x| x.flatten(1, -1))
            .add(nn::linear(
                &root,
                32 * T::CONV_OUTPUT_SIZE, /* conv output size */
                T::ACTION_SPACE_SIZE,
                Default::default(),
            ));

        let value_head = nn::seq_t()
            .add(nn::conv2d(&root, T::N_FILTERS, 3, 3, conv_config))
            .add(nn::batch_norm2d(&root, 3, Default::default()))
            .add_fn(Tensor::relu)
            .add_fn(|x| x.flatten(1, -1))
            .add(nn::linear(
                root,
                3 * T::CONV_OUTPUT_SIZE, /* conv output size */
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
    
    pub fn from_path<T: LearnableGame>(model_path: &PathBuf) -> Self {
        let vs = VarStore::new(*DEVICE);
        let mut nnet = Self::new::<T>(vs);
        let model_path_as_str = model_path.to_str().unwrap();
        match nnet.vs.load(model_path) {
            Ok(_) => println!("Successfully loaded model on path {}", model_path_as_str),
            Err(e) => panic!("unable to load model on path {}, error: {}", &model_path_as_str, e),
        }
        nnet
    }

    pub fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        let new_x = xs
            .apply_t(&self.init_block, train)
            .apply_t(&self.res_layer, train);
        
        let policy = new_x
            .apply_t(&self.policy_head, train)
            .softmax(1, None);

        let value = new_x
            .apply_t(&self.value_head, train);

        (policy, value)
    }

    // Similer to forward_t, only difference is policy is not softmaxed
    // Only used in AlphaZero::train, policy loss is calculated by cross entropy loss which already does a softmax
    pub fn forward_train(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        let new_x = xs
            .apply_t(&self.init_block, train)
            .apply_t(&self.res_layer, train);
        
        let policy = new_x
            .apply_t(&self.policy_head, train);
        let value = new_x
            .apply_t(&self.value_head, train);

        (policy, value)
    }

    pub fn forward_policy(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply_t(&self.init_block, train)
            .apply_t(&self.res_layer, train)
            .apply_t(&self.policy_head, train)
            .softmax(1, None)
    }
}
