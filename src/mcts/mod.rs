use std::{ops::Div, cmp::Ordering};

use crate::backgammon::{Actions, Backgammon};
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
    player: i8,
}

impl Node {
    fn new(
        state: Backgammon,
        parent: Option<&Node>,
        action_taken: Option<Actions>,
        player: i8,
    ) -> Self {
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

    fn ucb(&self) -> f32 {
        match &self.parent {
            Some(parent) => {
                let q_value = 1.0 - (self.value.div(self.visits));
                q_value + (CONFIG.c * parent.visits.ln().div(self.visits).sqrt())
            }
            None => f32::INFINITY,
        }
    }

    fn expand(&self) -> Node {
        if self.expandable_moves.is_empty() {
            panic!("expand() called on node with no expandable moves")
        }
        let move_idx = rand::thread_rng().gen_range(0..self.expandable_moves.len());

        let action_taken = self.expandable_moves.remove(move_idx);
        let next_state = Backgammon::get_next_state(self.state.board, action_taken, self.player);
        let child_backgammon = Backgammon::init_with_board(next_state);

        let child_node = Node::new(child_backgammon, Some(self), Some(action_taken), -self.player);
        self.children.push(child_node);
        return child_node;
    }
}


fn select(node: Node) -> Node {
    node.children.iter()
    .max_by(|a, b| a.ucb().partial_cmp(&b.ucb()).unwrap_or(Ordering::Equal))
    .cloned()
    .expect("select called on node without children!")
}

fn backpropagate(node: &mut Node, result: f32) {
    node.visits += 1.0;
    node.value += result;
    if let Some(parent) = node.parent.as_mut() {
        backpropagate(parent, result)
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
    let mut root = Node::new(state, None, None, player);

    for iteration in 0..CONFIG.iterations {
        let mut node = root.clone();
        while node.is_fully_expanded() {
            node = select(node)
        }
    }

    unimplemented!()
}
