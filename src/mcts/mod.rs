use crate::backgammon::{Backgammon, Board, Actions};

struct Node {
    state: Backgammon,
    parent: Option<usize>,
    children: Vec<usize>,
    visits: usize,
    wins: usize,
}

impl Node {
    fn new(state: Backgammon, parent: Option<usize>) -> Self {
        Node {
            state,
            parent,
            children: Vec::new(),
            visits: 0,
            wins: 0,
        }
    }
}

fn uct(node: &Node, total_visits: u32) -> f64 {
    unimplemented!()
}

fn select(node: &mut Node) -> &mut Node {
    unimplemented!()
}

fn simulate(node: &Node) -> i8 {
    unimplemented!()
}

fn backpropagate(node: &mut Node, result: i8) {
    unimplemented!()
}

fn mcts(root_state: Board, player: i8, iterations: u32) -> Actions {
    unimplemented!()
}