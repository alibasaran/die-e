use std::ops::Div;

use itertools::{multiunzip, Itertools};
use rand::seq::SliceRandom;
use tch::Tensor;

use crate::{
    backgammon::{Actions, Backgammon},
    constants::{ACTION_SPACE_SIZE, DEVICE, DEFAULT_TYPE},
};

use super::{node::Node, node_store::NodeStore};

pub fn get_prob_tensor(
    state: &Backgammon,
    root_node_idx: usize,
    store: &NodeStore
) -> Option<Tensor> {
    let children = store.get_node(root_node_idx).children;
    if children.is_empty() {
        return None;
    }
    let result = Tensor::full(ACTION_SPACE_SIZE, 0, (DEFAULT_TYPE, *DEVICE));
    let mut idxs: Vec<i64> = vec![];
    let mut visits: Vec<f32> = vec![];
    for child in children {
        let child_node = store.get_node(child);
        let encoded_action = state.encode(&child_node.action_taken.unwrap()) as i64;
        idxs.push(encoded_action);
        visits.push(child_node.visits)
    }
    let visits_tensor = Tensor::from_slice(&visits).to_device(*DEVICE);
    let indices_tensor = Tensor::from_slice(&idxs).to_device(*DEVICE);
    let probs = result.index_put(&[Some(indices_tensor)], &visits_tensor, false);
    let prob_sum = probs.sum(Some(DEFAULT_TYPE));
    Some(probs.div(prob_sum))
}
/**
 * Given a vec of root nodes of size N
 * returns a tensor of action pick probabilities N 1352
 * where tensor of size 1352 contains mostly zeros with values on only the encoded values of the node's expandable moves
 */
pub fn get_prob_tensor_parallel(nodes: &[&Node]) -> Tensor {
    let mut result = Tensor::zeros([nodes.len() as i64, ACTION_SPACE_SIZE], (DEFAULT_TYPE, *DEVICE));
    let (xs, ys, vals): (Vec<i32>, Vec<i32>, Vec<f32>) = multiunzip(nodes.iter().enumerate().flat_map(|(processed_idx, &node)| {
        node.expandable_moves.iter().map(move |actions| {
            // Idx of tensor N_i, the expandable move encoded, the value
            (processed_idx as i32, node.state.encode(actions) as i32, node.visits)
        })
    }));
    let xs_tensor = Tensor::from_slice(&xs);
    let ys_tensor = Tensor::from_slice(&ys);
    let vals_tensor = Tensor::from_slice(&vals).to_device(*DEVICE);

    let _ = result.index_put_(&[Some(xs_tensor), Some(ys_tensor)], &vals_tensor, false);
    let sum = result.sum(None);
    result / sum
}

pub fn turn_policy_to_probs_tensor_parallel(store: &NodeStore, node_indices: Vec<usize>, policy: &Tensor) -> Tensor {
    let mut mask = policy.zeros_like();
    let (xs, ys): (Vec<i32>, Vec<i32>) = node_indices.iter().flat_map(|i| {
        let node = store.get_node_ref(*i);
        node.expandable_moves.iter().map(move |actions| {
            (*i as i32, node.state.encode(actions) as i32)
        })
    }).unzip();
    let _ = mask.index_put_(&[Some(Tensor::from_slice(&xs)), Some(Tensor::from_slice(&ys))], &Tensor::from(1_f32), false);
    let selected_moves_tensor = policy * mask;
    let moves_sum = selected_moves_tensor.sum(None);
    selected_moves_tensor / moves_sum
}

pub fn turn_policy_to_probs_tensor(policy: &Tensor, node: &Node) -> Tensor {
    let mut result = Tensor::zeros_like(policy);
    let indices = node.expandable_moves.iter().map(|m| {
        node.state.encode(m) as i32
    }).collect_vec();
    let indices_tensor = vec![Some(Tensor::from_slice(&indices))];
    let values_to_put = policy.index(&indices_tensor);
    let _ = result.index_put_(&indices_tensor, &values_to_put, false);
    let sum = result.sum(None);
    result / sum
}

pub fn turn_policy_to_probs(policy: &Tensor, node: &Node) -> Vec<f32> {
    let mut values: Vec<f32> = vec![0.0; 1352];
    let mut encoded_values: Vec<usize> = Vec::with_capacity(node.expandable_moves.len());
    for action in &node.expandable_moves {
        let encoded_value = node.state.encode(action);
        encoded_values.push(encoded_value as usize);
        values[encoded_value as usize] = policy.double_value(&[encoded_value as i64]) as f32;
    }
    let sum: f32 = values.iter().sum();

    for i in 0..encoded_values.len() {
        let encoded_value = *encoded_values.get(i).unwrap();
        let prob = values[encoded_value] / sum;
        values[encoded_value] = prob;
    }

    values
}

pub fn random_play(bg: &Backgammon) -> Actions {
    let mut rng = rand::thread_rng();
    let moves = bg.get_valid_moves();

    if moves.is_empty() {
        return vec![];
    }

    return moves.choose(&mut rng).unwrap().to_vec();
}
