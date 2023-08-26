use crate::alphazero::encoding::encode;
use crate::alphazero::nnet::{ResNet, get_device};
use crate::backgammon::{Actions, Backgammon};
use indicatif::ProgressIterator;
use rand::{seq::SliceRandom, Rng};
use std::{cmp::Ordering, ops::Div};
use tch::Kind;
use tch::Tensor;

#[derive(Clone, Debug)]
pub struct Node {
    state: Backgammon,
    idx: usize,
    parent: Option<usize>,
    children: Vec<usize>,
    visits: f32,
    value: f32,
    policy: f32,
    is_double_move: bool,
    action_taken: Option<Actions>,
    expandable_moves: Vec<Actions>,
    player: i8,
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
        roll: Option<(u8, u8)>,
        is_double_move: bool,
        policy: f32,
    ) -> usize {
        let idx = self.nodes.len();
        let new_node = if let Some(roll) = roll {
            Node::new_with_roll(state, idx, parent, action_taken, player, roll, is_double_move, policy)
        } else {
            Node::new(state, idx, parent, action_taken, player, is_double_move, policy)
        };
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
        mut state: Backgammon,
        idx: usize,
        parent: Option<usize>,
        action_taken: Option<Actions>,
        player: i8,
        is_double_move: bool,
        policy: f32,
    ) -> Self {
        state.roll_die();
        let moves = state.get_valid_moves_len_always_2(player);
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
            policy,
            is_double_move,
        }
    }

    pub fn new_with_roll(
        mut state: Backgammon,
        idx: usize,
        parent: Option<usize>,
        action_taken: Option<Actions>,
        player: i8,
        roll: (u8, u8),
        is_double_move: bool,
        policy: f32,
    ) -> Self {
        state.roll = roll;
        let moves = state.get_valid_moves_len_always_2(player);
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
            is_double_move,
            policy,
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

    fn alpha_ucb(&self, store: &NodeStore) -> f32 {
        let q_value = if self.visits == 0.0 {
            0.0
        } else {
            self.value.div(self.visits)
        };
        match self.parent {
            Some(parent_idx) => {
                let parent = store.get_node(parent_idx);
                q_value + (CONFIG.c * parent.visits.sqrt().div(self.visits + 1.0)) * self.policy
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
        let next_state = Backgammon::get_next_state(self.state.board, &action_taken, self.player);
        let child_backgammon = Backgammon::init_with_board(next_state);

        let child_idx = if self.state.roll.0 != self.state.roll.1 || self.is_double_move {
            store.add_node(
                child_backgammon,
                Some(self.idx),
                Some(action_taken),
                -self.player,
                None,
                false,
                0.0
            )
        } else {
            store.add_node(
                child_backgammon,
                Some(self.idx),
                Some(action_taken),
                self.player,
                Some(self.state.roll),
                true,
                0.0
            )
        };
        
        self.children.push(child_idx);
        store.set_node(self);
        child_idx
    }

    fn alpha_expand(&mut self, store: &mut NodeStore, policy: Vec<f32>) {
        for action in self.expandable_moves.iter() {
            let next_state = Backgammon::get_next_state(self.state.board, action, self.player);
            let child_backgammon = Backgammon::init_with_board(next_state);
            let encoded_value = encode(
                action.clone(),
                self.state.roll,
                self.player,
            );
            let value = policy[encoded_value as usize];

            let child_idx = if self.state.roll.0 != self.state.roll.1 || self.is_double_move {
                store.add_node(
                    child_backgammon,
                    Some(self.idx),
                    Some(action.to_vec()),
                    -self.player,
                    None,
                    false,
                    value
                )
            } else {
                store.add_node(
                    child_backgammon,
                    Some(self.idx),
                    Some(action.to_vec()),
                    self.player,
                    Some(self.state.roll),
                    true,
                    value
                )
            };
            self.children.push(child_idx);
        }
        store.set_node(self);
    }

    pub fn simulate(&mut self, player: i8) -> f32 {
        let mut rng = rand::thread_rng();
        let mut curr_state = self.state.clone();
        let mut curr_player = self.player;

        for _ in 0..CONFIG.simulate_round_limit {
            if let Some(winner) = Backgammon::check_win_without_player(curr_state.board) {
                return (((winner / player) + 1) / 2) as f32;
            }

            curr_state.roll_die();
            let valid_moves = curr_state.get_valid_moves(curr_player);

            if !valid_moves.is_empty() {
                let move_to_play = valid_moves.choose(&mut rng).unwrap();
                curr_state.board =
                    Backgammon::get_next_state(curr_state.board, move_to_play, curr_player);
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

fn select_leaf_node(node_idx: usize, store: &NodeStore) -> usize {
    let node = store.get_node(node_idx);
    if node.children.is_empty() {
        return node.idx;
    }
    select_leaf_node(select_ucb(node_idx, store).idx, store)
}

fn alpha_select_leaf_node(node_idx: usize, store: &NodeStore) -> usize {
    let node = store.get_node(node_idx);
    if node.children.is_empty() {
        return node.idx;
    }
    alpha_select_leaf_node(select_alpha(node_idx, store).idx, store)
}

fn select_alpha(node_idx: usize, store: &NodeStore) -> Node {
    let node = store.get_node(node_idx);
    node.children
        .iter()
        .map(|child_idx| store.get_node(*child_idx))
        .max_by(|a, b| {
            a.alpha_ucb(store)
                .partial_cmp(&b.alpha_ucb(store))
                .unwrap_or(Ordering::Equal)
        })
        .expect("select_alpha called on node without children!")
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
    c: 1.0,
    // c: std::f32::consts::SQRT_2,
    simulate_round_limit: 100,
};

pub fn mct_search(state: Backgammon, player: i8) -> Actions {
    // Check if game already is terminal at root
    if Backgammon::check_win_without_player(state.board).is_some() {
        return vec![];
    }

    let mut store = NodeStore::new();
    let roll = state.roll;
    let root_node_idx = store.add_node(state, None, None, player, Some(roll), false, 0.0);
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

pub const ACTION_SPACE_SIZE: i64 = 1352;

pub fn alpha_mcts(state: &Backgammon, player: i8, net: &ResNet) -> Option<Tensor> {
    // Check if game already is terminal at root
    if Backgammon::check_win_without_player(state.board).is_some() {
        return None;
    }

    let mut store = NodeStore::new();
    let roll = state.roll;
    let root_node_idx = store.add_node(state.clone(), None, None, player, Some(roll), false, 0.0);
    let pb_iter = (0..CONFIG.iterations).progress().with_message("AlphaMCTS");

    for _ in pb_iter {
        // Don't forget to save the node later into the store
        let idx = alpha_select_leaf_node(root_node_idx, &store);
        let mut selected_node = store.get_node(idx);

        let value: f32;

        if !selected_node.is_terminal() {
            let (mut policy, eval) = net.forward_t(&state.as_tensor(player as i64), false);

            policy = policy.softmax(1, Kind::Float)
                .permute([1, 0]);
            let policy_vec = turn_policy_to_probs(&policy, &selected_node);
            value = eval.double_value(&[0]) as f32;
            selected_node.alpha_expand(&mut store, policy_vec);
        } else {
            value = ((Backgammon::check_win_without_player(selected_node.state.board).unwrap() + 1) / 2) as f32;
        }

        backpropagate(idx, value, &mut store);
    }
    let result = Tensor::full(ACTION_SPACE_SIZE, 0, (tch::Kind::Float, get_device()));
    let mut idxs: Vec<i64> = vec![];
    let mut visits: Vec<f32> = vec![];
    for child in store.get_node(root_node_idx).children {
        let child_node = store.get_node(child);
        let encoded_action = encode(child_node.action_taken.unwrap(), state.roll, player) as i64;
        idxs.push(encoded_action);
        visits.push(child_node.visits)
    }
    let visits_tensor = Tensor::from_slice(&visits).to_device(get_device());
    let indices_tensor = Tensor::from_slice(&idxs).to_device(get_device());
    let probs = result.index_put(&[Some(indices_tensor)], &visits_tensor, false);
    let prob_sum = probs.sum(Some(tch::Kind::Float));
    Some(probs.div(prob_sum))
}

fn turn_policy_to_probs(policy: &Tensor, node: &Node) -> Vec<f32> {
    let mut values: Vec<f32> = vec![0.0; 1352];
    let mut encoded_values: Vec<usize> = Vec::with_capacity(node.expandable_moves.len());
    for action in &node.expandable_moves {
        let encoded_value = encode(
            action.clone(),
            node.state.roll,
            node.player,
        );
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
