use std::{cmp::Ordering, ops::Div};

use crate::backgammon::{Actions, Backgammon};
use rand::Rng;

#[derive(Clone)]
struct Node {
    state: Backgammon,
    idx: usize,
    parent: Option<usize>,
    children: Vec<usize>,
    visits: f32,
    value: f32,
    action_taken: Option<Actions>,
    expandable_moves: Vec<Actions>,
    player: i8,
}

#[derive(Clone)]
struct NodeStore {
    nodes: Vec<Node>,
}

impl NodeStore {
    fn new() -> Self {
        NodeStore { nodes: vec![] }
    }

    fn add_node(
        &mut self,
        state: Backgammon,
        parent: Option<usize>,
        action_taken: Option<Actions>,
        player: i8,
    ) -> usize {
        let idx = self.nodes.len();
        let new_node = Node::new(state, idx + 1, parent, action_taken, player);
        self.nodes.push(new_node);
        idx + 1
    }

    fn get_node_as_mut(&mut self, idx: usize) -> &mut Node {
        &mut self.nodes[idx]
    }

    fn get_node(&self, idx: usize) -> &Node {
        &self.nodes[idx]
    }
}

impl Node {
    fn new(
        state: Backgammon,
        idx: usize,
        parent: Option<usize>,
        action_taken: Option<Actions>,
        player: i8,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let roll = (rng.gen_range(0..=6), rng.gen_range(0..=6));
        let moves = Backgammon::get_valid_moves(roll, state.board, player);
        Node {
            state,
            parent,
            idx,
            children: Vec::new(),
            action_taken,
            expandable_moves: moves,
            player,
            visits: 0.0,
            value: 0.0,
        }
    }

    fn is_fully_expanded(&self) -> bool {
        self.expandable_moves.is_empty() && !self.children.is_empty()
    }

    fn ucb(&self, store: &NodeStore) -> f32 {
        match self.parent {
            Some(parent_idx) => {
                let parent = store.get_node(parent_idx);
                let q_value = 1.0 - (self.value.div(self.visits));
                q_value + (CONFIG.c * parent.visits.ln().div(self.visits).sqrt())
            }
            None => f32::INFINITY,
        }
    }

    fn expand(&mut self, store: &mut NodeStore) -> usize {
        if self.expandable_moves.is_empty() {
            panic!("expand() called on node with no expandable moves")
        }
        let move_idx = rand::thread_rng().gen_range(0..self.expandable_moves.len());

        let action_taken = self.expandable_moves.remove(move_idx);
        let next_state =
            Backgammon::get_next_state(self.state.board, action_taken.clone(), self.player);
        let child_backgammon = Backgammon::init_with_board(next_state);

        let child_idx = store.add_node(
            child_backgammon,
            Some(self.idx),
            Some(action_taken),
            -self.player,
        );
        self.children.push(child_idx);
        child_idx
    }
}

fn select(node: Node, store: &NodeStore) -> Node {
    node.children
        .iter()
        .map(|child_idx| store.get_node(*child_idx))
        .max_by(|a, b| {
            a.ucb(store)
                .partial_cmp(&b.ucb(store))
                .unwrap_or(Ordering::Equal)
        })
        .cloned()
        .expect("select called on node without children!")
}

fn backpropagate(node_idx: usize, result: f32, store: &mut NodeStore) {
    let mut node = store.get_node_as_mut(node_idx);
    node.visits += 1.0;
    node.value += result;
    if let Some(parent) = node.parent {
        backpropagate(parent, result, store)
    }
}

struct MctsConfig {
    iterations: usize,
    c: f32,
}
const CONFIG: MctsConfig = MctsConfig {
    iterations: 1000,
    c: 1.0,
};

fn mct_search(state: Backgammon, player: i8) -> Actions {
    let store = NodeStore::new();
    let root = Node::new(state, 0, None, None, player);

    for iteration in 0..CONFIG.iterations {
        let mut node = root.clone();
        while node.is_fully_expanded() {
            node = select(node, &store)
        }
    }

    unimplemented!()
}
