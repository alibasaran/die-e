use crate::{
    tictactoe::TicTacToe,
    mcts::node_store::NodeStore,
    base::LearnableGame,
    constants::{DEFAULT_TYPE, DEVICE}
};
use tch::{Tensor, Device};

mod tests_util {

    use super::*;
    
    mod turn_policy_to_probs_tensor {

        use super::*;
        use crate::mcts::utils::turn_policy_to_probs_tensor_parallel;

        #[test]
        fn works_as_expected() {
            let mut store = NodeStore::new();
            for _ in 0..10 {
                store.add_node(TicTacToe::new(), None, None, 0.0);
            }
            let policy = Tensor::rand([10, TicTacToe::ACTION_SPACE_SIZE], (DEFAULT_TYPE, Device::Cpu));
            // policy.print();
            let indices = Vec::from_iter(0..10);
            let prob_tensor = turn_policy_to_probs_tensor_parallel(&store, &indices, &policy);

            assert_eq!(prob_tensor.size(), vec![10, TicTacToe::ACTION_SPACE_SIZE]);
            // Since this is policies converted to a tensor of probabilities
            // All sums of batches should be 1
            let sum = prob_tensor.sum_dim_intlist(1, false, None);
            // Tensor::equal behaves randomly
            assert!(sum.allclose(&sum.ones_like(), 1e-5, 1e-8, false))
        }
    }

    mod get_prob_tensor {
        use super::*;
        use crate::mcts::utils::get_prob_tensor_parallel;

        #[test]
        fn works_as_expected() {
            let mut store = NodeStore::new();
            let root = store.add_node(TicTacToe::new(), None, None, 0.0);
            for i in 1..9 {
                let child_idx = store.add_node(TicTacToe::new(), Some(root), Some(i), 0.0);
                let child_mut = store.get_node_as_mut(child_idx);
                child_mut.visits = 10.;
            }
            let mut root = store.get_node(0);
            root.children = Vec::from_iter(1..9);
            store.set_node(&root);

            let prob_tensor = get_prob_tensor_parallel(&[&root], &store);
            assert_eq!(prob_tensor.size(), vec![1, TicTacToe::ACTION_SPACE_SIZE]);
            // Since this is policies converted to a tensor of probabilities
            // All sums of batches should be 1
            let sum = prob_tensor.sum_dim_intlist(1, false, None);
            // Tensor::equal behaves randomly
            assert!(sum.allclose(&sum.ones_like(), 1e-5, 1e-8, false))
        }
    }
}