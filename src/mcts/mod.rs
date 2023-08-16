use std::{cmp::Ordering, ops::Div};
use crate::backgammon::{Actions, Backgammon, Board};
use indicatif::ProgressIterator;
use rand::{seq::SliceRandom, Rng};

#[derive(Clone, Debug)]
pub struct Node {
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

pub fn roll_die() -> (u8, u8) {
    let mut rng = rand::thread_rng();
    (rng.gen_range(1..=6), rng.gen_range(1..=6))
}

#[derive(Clone)]
pub struct NodeStore {
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
        roll: (u8, u8),
    ) -> usize {
        let idx = self.nodes.len();
        let new_node = Node::new(state, idx, parent, action_taken, player, roll);
        self.nodes.push(new_node);
        idx
    }

    fn set_node(&mut self, node: &Node) {
        self.nodes[node.idx] = node.clone()
    }

    fn get_node(&self, idx: usize) -> Node {
        self.nodes[idx].clone()
    }

    fn get_node_as_mut(&mut self, idx: usize) -> &mut Node {
        &mut self.nodes[idx]
    }
}

impl Node {
    pub fn new(
        state: Backgammon,
        idx: usize,
        parent: Option<usize>,
        action_taken: Option<Actions>,
        player: i8,
        roll: (u8, u8),
    ) -> Self {
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
        self.expandable_moves.is_empty()
    }

    fn is_terminal(&self) -> bool {
        Backgammon::check_win(self.state.board, self.player)
    }

    fn ucb(&self, store: &NodeStore) -> f32 {
        match self.parent {
            Some(parent_idx) => {
                let parent = store.get_node(parent_idx);
                let q_value = self.value.div(self.visits);
                q_value + (CONFIG.c * parent.visits.ln().div(self.visits).sqrt())
            }
            None => f32::INFINITY,
        }
    }

    fn win_pct(&self) -> f32 {
        self.value.div(self.visits)
    }

    fn expand(&mut self, store: &mut NodeStore) -> usize {
        if self.expandable_moves.is_empty() {
            panic!("expand() called on node with no expandable moves")
        }
        let move_idx = rand::thread_rng().gen_range(0..self.expandable_moves.len());

        let action_taken = self.expandable_moves.remove(move_idx);
        let next_state =
            Backgammon::get_next_state(self.state.board, &action_taken, self.player);
        let child_backgammon = Backgammon::init_with_board(next_state);

        let roll = roll_die();
        let child_idx = store.add_node(
            child_backgammon,
            Some(self.idx),
            Some(action_taken),
            -self.player,
            roll,
        );
        self.children.push(child_idx);
        store.set_node(self);
        child_idx
    }

    pub fn simulate(&mut self, player: i8) -> f32 {
        let mut rng = rand::thread_rng();
        let mut curr_state = self.state.board;
        let mut curr_player = self.player;

        for _ in 0..CONFIG.simulate_round_limit {
            if let Some(winner) = Backgammon::check_win_without_player(curr_state) {
                return (((winner / player) + 1) / 2) as f32;
            }

            let roll = (rng.gen_range(1..=6), rng.gen_range(1..=6));
            let valid_moves = Backgammon::get_valid_moves(roll, curr_state, curr_player);

            if !valid_moves.is_empty() {
                let move_to_play = valid_moves.choose(&mut rng).unwrap();
                curr_state =
                    Backgammon::get_next_state(curr_state, move_to_play, curr_player);
            }

            curr_player = -curr_player;
        }
        rng.gen_range(0..=1) as f32
    }
}

fn select_ucb(node_idx: usize, store: &NodeStore) -> Node {
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

fn select_win_pct(node_idx: usize, store: &NodeStore) -> Actions {
    let node = store.get_node(node_idx);
    let best_child = node.children
        .iter()
        .map(|child_idx| store.get_node(*child_idx))
        .max_by(|a, b| {
            a.win_pct()
                .partial_cmp(&b.win_pct())
                .unwrap_or(Ordering::Equal)
        });
    match best_child {
        Some(child) => child.action_taken.unwrap_or(vec![]),
        None => vec![]
    }
}

fn select_leaf_node(node_idx: usize, store: &NodeStore) -> usize {
    let node = store.get_node(node_idx);
    if node.children.is_empty() {
        return node.idx;
    }
    select_leaf_node(select_ucb(node_idx, store).idx, store)
}

fn backpropagate(node_idx: usize, result: f32, store: &mut NodeStore) {
    let node = store.get_node_as_mut(node_idx);
    node.visits += 1.0;
    node.value += result;
    if let Some(parent) = node.parent {
        backpropagate(parent, result, store)
    }
}

struct MctsConfig {
    iterations: usize,
    c: f32,
    simulate_round_limit: usize,
}

const CONFIG: MctsConfig = MctsConfig {
    iterations: 800,
    c: std::f32::consts::SQRT_2,
    simulate_round_limit: 200,
};

pub fn mct_search(state: Backgammon, player: i8, roll: (u8, u8)) -> Actions {
    // Check if game already is terminal at root
    if Backgammon::check_win_without_player(state.board).is_some() {
        return vec![]
    }
    
    let mut store = NodeStore::new();
    let root_node_idx = store.add_node(state, None, None, player, roll);
    let pb_iter = (0..CONFIG.iterations).progress().with_message("MCTS");
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

pub fn random_play(state: Board, player: i8, roll: (u8, u8)) -> Actions {
    let mut rng = rand::thread_rng();
    let moves = Backgammon::get_valid_moves(roll, state, player);

    if moves.is_empty() {
        return vec![];
    }

    return moves.choose(&mut rng).unwrap().to_vec();
}

fn pretty_print_tree(node_store: &NodeStore, index: usize, depth: usize, current_depth: usize) {
    if current_depth > depth {
        return;
    }

    let node = &node_store.get_node(index);
    let indent = "  ".repeat(current_depth);

    println!(
        "{}[{}] Action: {:?} \t\tVisits: {:.2} \tValue: {:.2} \tUCB: {:.5} \tWin_Pct: {:.3}",
        indent,
        index,
        node.action_taken,
        node.visits,
        node.value,
        node.ucb(node_store),
        node.win_pct()
    );

    for &child_index in &node.children {
        pretty_print_tree(node_store, child_index, depth, current_depth + 1);
    }
}
