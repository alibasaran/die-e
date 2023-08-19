use std::ops::Add;

use tch::nn;

/*
Constants

retrieved from: https://www.chessprogramming.org/AlphaZero#Network_Architecture
*/

const FILTERS: i64 = 256;
const N_RES_BLOCKS: u8 = 19;

#[derive(Debug)]
struct ResBlock {
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
struct ResNet {}
