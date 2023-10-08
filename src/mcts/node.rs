use crate::base::LearnableGame;
use rand::seq::SliceRandom;
use tch::Tensor;
use std::ops::Div;

use super::node_store::NodeStore;

#[derive(Debug)]
pub struct Node<T: LearnableGame> {
    pub state: T,
    pub idx: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub visits: f32,
    pub value: f32,
    pub policy: f32,
    pub action_taken: Option<T::Move>,
    pub expandable_moves: Vec<T::Move>,
}


impl<T: LearnableGame> Clone for Node<T>
where
    T: Clone, // Make sure T is Clone
    T::Move: Clone, // Make sure T::Move is Clone
{
    fn clone(&self) -> Self {
        Node {
            state: self.state,
            idx: self.idx,
            parent: self.parent,
            children: self.children.clone(),
            visits: self.visits,
            value: self.value,
            policy: self.policy,
            action_taken: self.action_taken.clone(),
            expandable_moves: self.expandable_moves.clone(),
        }
    }
}

impl <T: LearnableGame> Node<T>{
    pub fn new(
        state: T,
        idx: usize,
        parent: Option<usize>,
        action_taken: Option<T::Move>,
        policy: f32,
    ) -> Self {
        let moves = state.get_valid_moves();
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
            state: T::new(),
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

    pub fn is_fully_expanded(&self) -> bool {
        self.expandable_moves.is_empty()
    }

    pub fn is_terminal(&self) -> bool {
        self.state.check_winner().is_some()
    }

    pub fn ucb(&self, store: &NodeStore<T>, c: f32) -> f32 {
        match self.parent {
            Some(parent_idx) => {
                let parent = store.get_node(parent_idx);
                let exploitation = self.value / self.visits;
                let exploration = (c * parent.visits.ln() / self.visits).sqrt();
                exploitation + exploration
            }
            None => f32::INFINITY,
        }
    }

    pub fn alpha_ucb(&self, store: &NodeStore<T>, c: f32) -> f32 {
        let q_value = if self.visits == 0.0 {
            0.0
        } else {
            self.value.div(self.visits)
        };
        match self.parent {
            Some(parent_idx) => {
                let parent = store.get_node(parent_idx);
                q_value
                    + (c * parent.visits.sqrt().div(self.visits + 1.0)) * self.policy
            }
            None => f32::INFINITY,
        }
    }

    pub fn win_pct(&self) -> f32 {
        self.value.div(self.visits)
    }

    pub fn expand(&mut self, store: &mut NodeStore<T>) -> usize {
        if self.expandable_moves.is_empty() {
            panic!("expand() called on node with no expandable moves")
        }

        let action_taken = self.expandable_moves.pop().unwrap();
        let mut next_state = self.state;
        next_state.apply_move(&action_taken);

        let child_idx = store.add_node(
            next_state,
            Some(self.idx),
            Some(action_taken),
            0.0,
        );

        self.children.push(child_idx);
        store.set_node(self);
        child_idx
    }

    pub fn alpha_expand(&mut self, store: &mut NodeStore<T>, policy: Vec<f32>) {
        for action in self.expandable_moves.drain(..) {
            let encoded_value = self.state.encode(&action);
            let value = policy[encoded_value as usize];
            let mut next_state = self.state;
            next_state.apply_move(&action);

            let child_idx = store.add_node(
                next_state,
                Some(self.idx),
                Some(action.clone()),
                value,
            );
            self.children.push(child_idx);
        }
        store.set_node(self);
    }

    pub fn alpha_expand_tensor(&mut self, store: &mut NodeStore<T>, policy: &Tensor) {
        for action in self.expandable_moves.drain(..) {
            let encoded_value = self.state.encode(&action);
            let value = policy.double_value(&[encoded_value.into()]) as f32;
            
            let mut next_state = self.state;
            next_state.apply_move(&action);

            let child_idx = store.add_node(
                next_state,
                Some(self.idx),
                Some(action.clone()),
                value,
            );
            self.children.push(child_idx);
        }
        store.set_node(self);
    }

    pub fn simulate(&self, player: i8, sim_limit: usize) -> f32 {
        let mut rng = rand::thread_rng();
        let mut curr_state = self.state;

        for _ in 0..sim_limit {
            if let Some(winner) = self.state.check_winner() {
                return if winner == player {1.}
                else if winner == -player {-1.}
                else {0.};
            }
            let valid_moves = curr_state.get_valid_moves();

            if !valid_moves.is_empty() {
                let move_to_play = valid_moves.choose(&mut rng).unwrap();
                curr_state.apply_move(move_to_play);
            } else {
                curr_state.skip_turn();
            }
        }
        0.
    }
}
