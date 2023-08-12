use std::ops::Div;

use crate::backgammon::{Backgammon, Actions};
use rand::Rng;

#[derive(Clone)]
struct Node {
    state: Backgammon,
    parent: Option<Box<Node>>,
    children: Vec<Node>,
    visits: f32,
    value: f32,
    action_taken: Option<Actions>,
    expandable_moves: Vec<Actions>,
    player: i8
}

impl Node {
    fn new(state: Backgammon, parent: Option<Node>, action_taken: Option<Actions>, player: i8) -> Self {
        let mut rng = rand::thread_rng();
        let roll = (rng.gen_range(0..=6), rng.gen_range(0..=6));
        let moves = Backgammon::get_valid_moves(roll, state.board, player);
        Node {
            state,
            parent: parent.map(Box::from),
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
}

fn ucb(node: &Node, child: &Node) -> f32 {
    let q_value = 1.0 - (child.value.div(child.visits));
    q_value + (CONFIG.c * node.visits.ln().div(child.visits).sqrt())
}

fn select(node: Node) -> Node {
    let mut best_child: Option<Node> = None;
    let mut best_ucb = f32::NEG_INFINITY;
    for child in node.children.iter() {
        let curr_ucb = ucb(&node, child);
        if curr_ucb > best_ucb {
            best_child = Some(child.clone());
            best_ucb = curr_ucb;
        }
    }
    match best_child {
        Some(child) => child,
        None => panic!("select called on node without children!")
    }
}

fn backpropagate(node: &mut Node, result: i8) {
    while let Some(parent) = node.parent.as_mut() {
    }
}

struct MctsConfig {
    iterations: usize,
    c: f32
}
const CONFIG: MctsConfig = MctsConfig {
    iterations: 1000,
    c: 1.0
};

fn mct_search(state: Backgammon, player: i8) -> Actions {
    let mut root = Node::new(state, None, None, player);

    for iteration in 0..CONFIG.iterations {
        let mut node = root.clone();
        while node.is_fully_expanded() {
            node = select(node)
        }
    }

    unimplemented!()
}