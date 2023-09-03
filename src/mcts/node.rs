use crate::MCTS_CONFIG;
use crate::backgammon::{Actions, Backgammon};
use rand::{seq::SliceRandom, Rng};
use tch::Tensor;
use std::ops::Div;

use super::node_store::NodeStore;

#[derive(Clone, Debug)]
pub struct Node {
    pub state: Backgammon,
    pub idx: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub visits: f32,
    pub value: f32,
    pub policy: f32,
    pub action_taken: Option<Actions>,
    pub expandable_moves: Vec<Actions>,
}

impl Node {
    pub fn new(
        mut state: Backgammon,
        idx: usize,
        parent: Option<usize>,
        action_taken: Option<Actions>,
        policy: f32,
    ) -> Self {
        state.roll_die();
        let moves = state.get_valid_moves_len_always_2();
        Node {
            state,
            parent,
            idx,
            children: Vec::new(),
            action_taken,
            expandable_moves: moves,
            visits: 0.0,
            value: 0.0,
            policy,
        }
    }

    pub fn empty() -> Self {
        Node {
            state: Backgammon::new(),
            parent: None,
            idx: 0,
            children: Vec::new(),
            action_taken: None,
            expandable_moves: Vec::new(),
            visits: 0.0,
            value: 0.0,
            policy: 0.,
        }
    }

    pub fn new_with_roll(
        mut state: Backgammon,
        idx: usize,
        parent: Option<usize>,
        action_taken: Option<Actions>,
        roll: (u8, u8),
        policy: f32,
    ) -> Self {
        state.roll = roll;
        let moves = state.get_valid_moves_len_always_2();
        Node {
            state,
            parent,
            idx,
            children: Vec::new(),
            action_taken,
            expandable_moves: moves,
            visits: 0.0,
            value: 0.0,
            policy,
        }
    }

    pub fn is_fully_expanded(&self) -> bool {
        self.expandable_moves.is_empty()
    }

    pub fn is_terminal(&self) -> bool {
        Backgammon::check_win(self.state.board, self.state.player)
    }

    pub fn ucb(&self, store: &NodeStore) -> f32 {
        match self.parent {
            Some(parent_idx) => {
                let parent = store.get_node(parent_idx);
                let q_value = self.value.div(self.visits);
                q_value + (MCTS_CONFIG.c * parent.visits.ln().div(self.visits).sqrt())
            }
            None => f32::INFINITY,
        }
    }

    pub fn alpha_ucb(&self, store: &NodeStore) -> f32 {
        let q_value = if self.visits == 0.0 {
            0.0
        } else {
            self.value.div(self.visits)
        };
        match self.parent {
            Some(parent_idx) => {
                let parent = store.get_node(parent_idx);
                q_value
                    + (MCTS_CONFIG.c * parent.visits.sqrt().div(self.visits + 1.0)) * self.policy
            }
            None => f32::INFINITY,
        }
    }

    pub fn win_pct(&self) -> f32 {
        self.value.div(self.visits)
    }

    pub fn expand(&mut self, store: &mut NodeStore) -> usize {
        if self.expandable_moves.is_empty() {
            panic!("expand() called on node with no expandable moves")
        }
        let move_idx = rand::thread_rng().gen_range(0..self.expandable_moves.len());

        let action_taken = self.expandable_moves.remove(move_idx);
        let next_state = Backgammon::get_next_state(self.state.board, &action_taken, self.state.player);

        let child_idx = if self.state.roll.0 != self.state.roll.1 || self.state.is_second_play {
            store.add_node(
                Backgammon::init_with_fields(next_state, -self.state.player, false),
                Some(self.idx),
                Some(action_taken),
                None,
                0.0,
            )
        } else {
            store.add_node(
                Backgammon::init_with_fields(next_state, self.state.player, true),
                Some(self.idx),
                Some(action_taken),
                Some(self.state.roll),
                0.0,
            )
        };

        self.children.push(child_idx);
        store.set_node(self);
        child_idx
    }

    pub fn alpha_expand(&mut self, store: &mut NodeStore, policy: Vec<f32>) {
        for action in self.expandable_moves.iter() {
            let next_state = Backgammon::get_next_state(self.state.board, action, self.state.player);
            let encoded_value = self.state.encode(&action);
            let value = policy[encoded_value as usize];

            let child_idx = if self.state.roll.0 != self.state.roll.1 || self.state.is_second_play {
                store.add_node(
                    Backgammon::init_with_fields(next_state, -self.state.player, false),
                    Some(self.idx),
                    Some(action.to_vec()),
                    None,
                    value,
                )
            } else {
                store.add_node(
                    Backgammon::init_with_fields(next_state, self.state.player, true),
                    Some(self.idx),
                    Some(action.to_vec()),
                    Some(self.state.roll),
                    value,
                )
            };
            self.children.push(child_idx);
        }
        store.set_node(self);
    }

    pub fn alpha_expand_tensor(&mut self, store: &mut NodeStore, policy: Tensor) {
        for action in self.expandable_moves.iter() {
            let next_state = Backgammon::get_next_state(self.state.board, action, self.state.player);
            let encoded_value = self.state.encode(action);
            let value = policy.double_value(&[encoded_value.into()]) as f32;

            let child_idx = if self.state.roll.0 != self.state.roll.1 || self.state.is_second_play {
                store.add_node(
                    Backgammon::init_with_fields(next_state, -self.state.player, false),
                    Some(self.idx),
                    Some(action.to_vec()),
                    None,
                    value,
                )
            } else {
                store.add_node(
                    Backgammon::init_with_fields(next_state, self.state.player, true),
                    Some(self.idx),
                    Some(action.to_vec()),
                    Some(self.state.roll),
                    value,
                )
            };
            self.children.push(child_idx);
        }
        store.set_node(self);
    }

    pub fn simulate(&mut self, player: i8) -> f32 {
        let mut rng = rand::thread_rng();
        let mut curr_state = self.state;
        let mut curr_player = self.state.player;

        for _ in 0..MCTS_CONFIG.simulate_round_limit {
            if let Some(winner) = Backgammon::check_win_without_player(curr_state.board) {
                return (((winner / player) + 1) / 2) as f32;
            }

            curr_state.roll_die();
            let valid_moves = curr_state.get_valid_moves();

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
