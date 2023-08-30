use std::cmp::Ordering;

use indicatif::ProgressIterator;

use crate::{backgammon::{Actions, Backgammon}, MCTS_CONFIG};

use super::{node::Node, node_store::NodeStore};


pub fn mct_search(state: Backgammon, player: i8) -> Actions {
    // Check if game already is terminal at root
    if Backgammon::check_win_without_player(state.board).is_some() {
        return vec![];
    }

    let mut store = NodeStore::new();
    let roll = state.roll;
    let root_node_idx = store.add_node(state, None, None, Some(roll), 0.0);
    let pb_iter = (0..MCTS_CONFIG.iterations).progress().with_message("MCTS");
    for _ in pb_iter {
        // Don't forget to save the node later into the store
        let idx = select_leaf_node(root_node_idx, &store);
        let mut selected_node = store.get_node(idx);

        // if selected_node.is_terminal() {
        //     let result = selected_node.simulate(player);
        //     backpropagate(selected_node.idx, result, &mut store);
        // }

        while !selected_node.is_fully_expanded() {
            let new_node_idx = selected_node.expand(&mut store);
            let mut new_node = store.get_node(new_node_idx);
            let result = new_node.simulate(player);
            backpropagate(new_node_idx, result, &mut store)
        }
    }
    select_win_pct(root_node_idx, &store)
}

pub fn select_ucb(node_idx: usize, store: &NodeStore) -> Node {
    let node = store.get_node(node_idx);
    node.children
        .iter()
        .map(|child_idx| store.get_node(*child_idx))
        .max_by(|a, b| {
            a.ucb(store)
                .partial_cmp(&b.ucb(store))
                .unwrap_or(Ordering::Equal)
        })
        .expect("select_ucb called on node without children!")
}

pub fn select_win_pct(node_idx: usize, store: &NodeStore) -> Actions {
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
        Some(child) => child.action_taken.unwrap_or(vec![]),
        None => vec![],
    }
}

pub fn select_leaf_node(node_idx: usize, store: &NodeStore) -> usize {
    let node = store.get_node(node_idx);
    if node.children.is_empty() {
        return node.idx;
    }
    select_leaf_node(select_ucb(node_idx, store).idx, store)
}

pub fn backpropagate(node_idx: usize, result: f32, store: &mut NodeStore) {
    let node = store.get_node_as_mut(node_idx);
    node.visits += 1.0;
    node.value += result;
    if let Some(parent) = node.parent {
        backpropagate(parent, result, store)
    }
}
