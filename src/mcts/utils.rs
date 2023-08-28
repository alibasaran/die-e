use std::ops::Div;

use rand::seq::SliceRandom;
use tch::Tensor;

use crate::{
    alphazero::encoding::encode,
    backgammon::{Actions, Backgammon}, constants::DEVICE,
};



use super::{node::Node, node_store::NodeStore, ACTION_SPACE_SIZE};

pub fn get_prob_tensor(
    state: &Backgammon,
    root_node_idx: usize,
    store: &NodeStore,
    player: i8,
) -> Option<Tensor> {
    let result = Tensor::full(ACTION_SPACE_SIZE, 0, (tch::Kind::Float, *DEVICE));
    let mut idxs: Vec<i64> = vec![];
    let mut visits: Vec<f32> = vec![];
    let children = store.get_node(root_node_idx).children;
    if children.is_empty() {
        return None;
    }
    for child in children {
        let child_node = store.get_node(child);
        let encoded_action = encode(child_node.action_taken.unwrap(), state.roll, player) as i64;
        idxs.push(encoded_action);
        visits.push(child_node.visits)
    }
    let visits_tensor = Tensor::from_slice(&visits).to_device(*DEVICE);
    let indices_tensor = Tensor::from_slice(&idxs).to_device(*DEVICE);
    let probs = result.index_put(&[Some(indices_tensor)], &visits_tensor, false);
    let prob_sum = probs.sum(Some(tch::Kind::Float));
    Some(probs.div(prob_sum))
}

pub fn turn_policy_to_probs(policy: &Tensor, node: &Node) -> Vec<f32> {
    let mut values: Vec<f32> = vec![0.0; 1352];
    let mut encoded_values: Vec<usize> = Vec::with_capacity(node.expandable_moves.len());
    for action in &node.expandable_moves {
        let encoded_value = encode(action.clone(), node.state.roll, node.player);
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

pub fn random_play(bg: &Backgammon, player: i8) -> Actions {
    let mut rng = rand::thread_rng();
    let moves = Backgammon::get_valid_moves(bg, player);

    if moves.is_empty() {
        return vec![];
    }

    return moves.choose(&mut rng).unwrap().to_vec();
}