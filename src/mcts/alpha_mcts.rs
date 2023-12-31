use std::{cmp::Ordering, collections::HashSet};


use indicatif::{ProgressIterator, ProgressBar};
use itertools::Itertools;

use tch::Tensor;

use crate::{alphazero::nnet::ResNet, constants::DEVICE, mcts::{noise::apply_dirichlet, utils::{turn_policy_to_probs_tensor, turn_policy_to_probs_tensor_parallel}}, MctsConfig, base::LearnableGame};

use super::{node_store::NodeStore, node::Node, simple_mcts::backpropagate, utils::{turn_policy_to_probs, get_prob_tensor}};


fn alpha_select_leaf_node<T: LearnableGame>(node_idx: usize, store: &NodeStore<T>, c: f32) -> usize {
    let node = store.get_node(node_idx);
    if node.children.is_empty() {
        return node.idx;
    }
    alpha_select_leaf_node(select_alpha(node_idx, store, c).idx, store, c)
}

fn select_alpha<T: LearnableGame>(node_idx: usize, store: &NodeStore<T>, c: f32) -> Node<T> {
    let node = store.get_node(node_idx);
    node.children
        .iter()
        .map(|child_idx| store.get_node(*child_idx))
        .max_by(|a, b| {
            a.alpha_ucb(store, c)
                .partial_cmp(&b.alpha_ucb(store, c))
                .unwrap_or(Ordering::Equal)
        })
        .expect("select_alpha called on node without children!")
}

pub fn apply_dirichlet_to_root<T: LearnableGame>(root_idx: usize, store: &mut NodeStore<T>, net: &ResNet, state: &impl LearnableGame, mcts_config: &MctsConfig) {
    let mut root_node = store.get_node(root_idx);
    
    let policy = net.forward_policy(&state.as_tensor().to_device(*DEVICE), false);

    let policy_vec = turn_policy_to_probs(&policy, &root_node);
    root_node.alpha_expand(store, policy_vec);
    if root_node.children.len() < 2 {
        // Don't apply dirichlet on a node with a single child
        return
    }
    // Set visits to 1
    root_node.visits = 1.;
    // Apply dirichlet to root policies
    root_node.apply_dirichlet(mcts_config.dirichlet_alpha, mcts_config.dirichlet_epsilon, store);
    store.set_node(&root_node)
}

pub fn alpha_mcts<T: LearnableGame>(state: &T, net: &ResNet, mcts_config: &MctsConfig) -> Option<Tensor> {
    // Set no_grad_guard
    let _guard = tch::no_grad_guard();
    
    // Check if game already is terminal at root
    if state.check_winner().is_some() {
        return None;
    }
    let mut store = NodeStore::new();
    let root_node_idx = store.add_node(*state, None, None, 0.0);
    
    apply_dirichlet_to_root(root_node_idx, &mut store, net, state, mcts_config);
    
    let pb_iter = (0..mcts_config.iterations).progress().with_message("AlphaMCTS");
    for _ in pb_iter {
        // Don't forget to save the node later into the store
        let idx = alpha_select_leaf_node(root_node_idx, &store, mcts_config.c);
        let mut selected_node = store.get_node(idx);

        let value = if !selected_node.is_terminal() {
            let (policy, eval) = net.forward_t(&selected_node.state.as_tensor().to_device(*DEVICE), false);

            let policy_vec = turn_policy_to_probs(&policy, &selected_node);
            selected_node.alpha_expand(&mut store, policy_vec);
            eval.double_value(&[0]) as f32
        } else {
            ((state.check_winner().unwrap() + 1) / 2) as f32
        };

        backpropagate(idx, value, &mut store);
    }
    get_prob_tensor(state, root_node_idx, &store)
}


/*
    Similar to alpha_mcts, however this function mutates the NodeStore rather than returning probabilities
*/
pub fn alpha_mcts_parallel<T: LearnableGame>(store: &mut NodeStore<T>, states: &[T], net: &ResNet, mcts_config: &MctsConfig, pb: Option<ProgressBar>) {
    // Set no_grad_guard
    let _guard = tch::no_grad_guard();
    assert!(store.is_empty(), "AlphaMCTS paralel expects an empty store");
    
    // Convert all states a tensor
    let states_vec = states.iter().map(|state| state.as_tensor()).collect_vec();
    let states_tensor = Tensor::stack(
        &states_vec,
        0
    ).squeeze_dim(1).to_device(*DEVICE);

    // Get policy tensor
    let policy = net.forward_policy(&states_tensor, false);

    // Apply dirichlet to root policy tensor
    let policy_dir = apply_dirichlet(&policy, mcts_config.dirichlet_alpha, mcts_config.dirichlet_epsilon);

    // Create root node for each game state
    for (_, state) in states.iter().enumerate() {
        store.add_node(*state, None, None, 0.0);
    }

    // Expand root node for each game state
    // Move policy to CPU because alpha expand makes multiple double_value(idx) calls,
    // Takes too long when tensors are in GPU
    let prob_tensor = turn_policy_to_probs_tensor_parallel(store, &(0..states.len()).collect_vec(), &policy_dir)
        .to_device(tch::Device::Cpu);
    for i in 0..states.len() {
        // Create root node for each game state
        let mut root = store.get_node(i);

        root.visits = 1.;
        // let prob_tensor = turn_policy_to_probs_tensor(&policy_dir.get(i as i64), &root);
        // Expand root
        root.alpha_expand_tensor(store, &prob_tensor.get(i as i64));
    }
    
    /*
        Create two vectors:
            - Games:
                - values ranging from 0 to N_SELF_PLAY_BATCHES
                - used to track how many games are still not completed
            - Selected_nodes:
                - N_SELF_PLAY_BATCHES node values
                - selected_nodes[i] refers to the node that was last selected for game[i]

    */
    let games: HashSet<usize> = HashSet::from_iter(0..states.len());

    // Game id to node idx mapping
    let mut selected_nodes_idxs = vec![0; states.len()];

    let pb = match pb {
        Some(p) => p,
        None => ProgressBar::new(mcts_config.iterations as u64)
    };

    for _ in 0..mcts_config.iterations {
        // Clear selected node indices
        let mut node_selected = false;
        pb.inc(1);
        for &game_idx in games.iter() {
            let idx = alpha_select_leaf_node(game_idx, store, mcts_config.c);
            let selected_node = store.get_node(idx);
            
            if let Some(winner) = selected_node.state.check_winner() {
                let game_root = store.get_node(game_idx);
                let root_player = game_root.state.get_player();
                let value = if winner == root_player {1.}
                    else if winner == -root_player {-1.}
                    else {0.};
                backpropagate(idx, value as f32, store);
            } else {
                node_selected = true;
                selected_nodes_idxs[game_idx] = selected_node.idx;
            }
        }

        if !node_selected {
            continue;
        }
            
        // Convert ongoing (not terminal) games into a tensor vec
        let selected_states_vec = selected_nodes_idxs.iter().map(|node_idx| {
            let node = store.get_node(*node_idx);
            node.state.as_tensor()
        }).collect_vec();

        let selected_states_tensor = Tensor::stack(
            &selected_states_vec,
            0
        ).squeeze_dim(1).to_device(*DEVICE);

        // Calculate policies
        let (policy, eval) = net.forward_t(&selected_states_tensor, false);
        
        // Expand and backprop selected nodes with their respective calculated policies and evals
        let eval = eval.to_device(tch::Device::Cpu);
        let policy = policy.to_device(tch::Device::Cpu);

        for (processed_idx, &node_idx) in selected_nodes_idxs.iter().enumerate() {
            let mut node = store.get_node(node_idx);
            let (policy_i, eval_i) = (policy.get(processed_idx as i64), eval.get(processed_idx as i64));
            let prob_tensor = turn_policy_to_probs_tensor(&policy_i, &node);
            // Expand selected node
            node.alpha_expand_tensor(store, &prob_tensor);
            let value = eval_i.double_value(&[0]) as f32;
            backpropagate(node.idx, value, store);
        }
    }
}
