use std::cmp::Ordering;



use crate::{MctsConfig, base::LearnableGame};

use super::{node::Node, node_store::NodeStore};


pub fn mct_search<T: LearnableGame>(state: T, player: i8, mcts_config: &MctsConfig) -> T::Move {
    // Check if game already is terminal at root
    if state.check_winner().is_some() {
        return T::EMPTY_MOVE;
    }

    let mut store = NodeStore::new();
    let root_node_idx = store.add_node(state, None, None, 0.0);
    // let pb_iter = .progress().with_message("MCTS");
    for _ in 0..mcts_config.iterations {
        // Don't forget to save the node later into the store
        let idx = select_leaf_node(root_node_idx, &store, mcts_config.c);
        let mut selected_node = store.get_node(idx);


        while !selected_node.is_fully_expanded() {
            let new_node_idx = selected_node.expand(&mut store);
            let mut new_node = store.get_node(new_node_idx);
            let result = new_node.simulate(player, mcts_config.simulate_round_limit);
            backpropagate(new_node_idx, result, &mut store)
        }
    }
    select_win_pct(root_node_idx, &store)
}

pub fn select_ucb<T: LearnableGame>(node_idx: usize, store: &NodeStore<T>, c: f32) -> Node<T> {
    let node = store.get_node(node_idx);
    node.children
        .iter()
        .map(|child_idx| store.get_node(*child_idx))
        .max_by(|a, b| {
            a.ucb(store, c)
                .partial_cmp(&b.ucb(store, c))
                .unwrap_or(Ordering::Equal)
        })
        .expect("select_ucb called on node without children!")
}

pub fn select_win_pct<T: LearnableGame>(node_idx: usize, store: &NodeStore<T>) -> T::Move {
    let node = store.get_node(node_idx);
    let best_child = node
        .children
        .iter()
        .map(|child_idx| store.get_node(*child_idx))
        .max_by(|a, b| {
            a.win_pct()
                .partial_cmp(&b.win_pct())
                .unwrap_or(Ordering::Equal)
        });
    match best_child {
        Some(child) => child.action_taken.unwrap_or(T::EMPTY_MOVE),
        None => T::EMPTY_MOVE,
    }
}

pub fn select_leaf_node<T: LearnableGame>(node_idx: usize, store: &NodeStore<T>, c: f32) -> usize {
    let node = store.get_node(node_idx);
    if node.children.is_empty() {
        return node.idx;
    }
    select_leaf_node(select_ucb(node_idx, store, c).idx, store, c)
}

pub fn backpropagate<T: LearnableGame>(node_idx: usize, result: f32, store: &mut NodeStore<T>) {
    let node = store.get_node_as_mut(node_idx);
    node.visits += 1.0;
    node.value += result;
    if let Some(parent) = node.parent {
        backpropagate(parent, result, store)
    }
}
