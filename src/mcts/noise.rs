use rand::thread_rng;
use rand_distr::{Dirichlet, Distribution};

use super::{node::Node, node_store::NodeStore};


impl Node {
    pub fn apply_dirichlet(&self, alpha: f32, eps: f32, store: &mut NodeStore) {
        assert!(self.visits > 0., "unable to apply dirichlet, node has no visits!");
        let dirichlet = Dirichlet::new(&vec![alpha; self.children.len()]).unwrap();
        let sample = dirichlet.sample(&mut thread_rng());
        for (child_idx, noise) in self.children.iter().zip(sample) {
            let child = store.get_node_as_mut(*child_idx);
            child.policy = noise * eps + child.policy * (1. - eps)
        }
    }
}