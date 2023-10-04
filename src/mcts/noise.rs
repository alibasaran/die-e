use rand::thread_rng;
use rand_distr::{Dirichlet, Distribution};
use tch::Tensor;

use crate::{constants::{DEVICE, DEFAULT_TYPE}, base::LearnableGame};

use super::{node::Node, node_store::NodeStore};

impl <T: LearnableGame> Node <T> {
    pub fn apply_dirichlet(&self, alpha: f32, eps: f32, store: &mut NodeStore<T>) {
        assert!(
            self.visits > 0.,
            "unable to apply dirichlet, node has no visits!"
        );
        let dirichlet = Dirichlet::new(&vec![alpha; self.children.len()]).unwrap();
        let sample = dirichlet.sample(&mut thread_rng());
        for (child_idx, noise) in self.children.iter().zip(sample) {
            let child = store.get_node_as_mut(*child_idx);
            child.policy = noise * eps + child.policy * (1. - eps)
        }
    }
}

/*
    Apply dirichlet noise to a tensor
*/
pub fn apply_dirichlet(tensor: &Tensor, alpha: f32, eps: f32) -> Tensor {
    let n_policies = tensor.size()[1] as usize;
    let dirichlet = Dirichlet::new(&vec![alpha; n_policies]).unwrap();
    let diriclet_tensor = Tensor::from_slice(&dirichlet.sample(&mut thread_rng()))
        .to_device_(*DEVICE, DEFAULT_TYPE, false, false)
        .unsqueeze(0);
    (1. - eps) * tensor + eps * diriclet_tensor
}
