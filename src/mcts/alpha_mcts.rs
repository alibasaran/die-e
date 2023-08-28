use std::cmp::Ordering;

use indicatif::ProgressIterator;
use tch::Tensor;

use crate::{backgammon::Backgammon, alphazero::nnet::ResNet};

use super::{node_store::NodeStore, node::Node, MCTS_CONFIG, mcts::backpropagate, utils::{turn_policy_to_probs, get_prob_tensor}};

fn alpha_select_leaf_node(node_idx: usize, store: &NodeStore) -> usize {
    let node = store.get_node(node_idx);
    if node.children.is_empty() {
        return node.idx;
    }
    alpha_select_leaf_node(select_alpha(node_idx, store).idx, store)
}

fn select_alpha(node_idx: usize, store: &NodeStore) -> Node {
    let node = store.get_node(node_idx);
    node.children
        .iter()
        .map(|child_idx| store.get_node(*child_idx))
        .max_by(|a, b| {
            a.alpha_ucb(store)
                .partial_cmp(&b.alpha_ucb(store))
                .unwrap_or(Ordering::Equal)
        })
        .expect("select_alpha called on node without children!")
}

pub fn alpha_mcts(state: &Backgammon, player: i8, net: &ResNet, root_is_double: bool) -> Option<Tensor> {
    // Check if game already is terminal at root
    if Backgammon::check_win_without_player(state.board).is_some() {
        return None;
    }
    let mut store = NodeStore::new();
    let roll = state.roll;
    let root_node_idx = store.add_node(state.clone(), None, None, player, Some(roll), root_is_double, 0.0);
    let pb_iter = (0..MCTS_CONFIG.iterations).progress().with_message("AlphaMCTS");

    for _ in pb_iter {
        // Don't forget to save the node later into the store
        let idx = alpha_select_leaf_node(root_node_idx, &store);
        let mut selected_node = store.get_node(idx);

        let value: f32;

        if !selected_node.is_terminal() {
            let (mut policy, eval) = net.forward_t(&state.as_tensor(player as i64, selected_node.is_double_move), false);

            policy = policy.softmax(1, tch::Kind::Float)
                .permute([1, 0]);
            let policy_vec = turn_policy_to_probs(&policy, &selected_node);
            value = eval.double_value(&[0]) as f32;
            selected_node.alpha_expand(&mut store, policy_vec);
        } else {
            value = ((Backgammon::check_win_without_player(selected_node.state.board).unwrap() + 1) / 2) as f32;
        }

        backpropagate(idx, value, &mut store);
    }
    get_prob_tensor(state, root_node_idx, &store, player)
}