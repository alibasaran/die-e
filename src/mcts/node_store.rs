use std::fmt;

use itertools::Itertools;

use crate::{base::LearnableGame};

use super::node::Node;

pub struct NodeStore<T: LearnableGame> {
    nodes: Vec<Node<T>>,
}

impl <T: LearnableGame> fmt::Display for NodeStore<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NodeStore<size={}>", self.nodes.len())
    }
}

impl <T: LearnableGame> NodeStore<T> {
    pub fn new() -> Self {
        NodeStore { nodes: vec![] }
    }

    pub fn get_root_nodes(&self) -> Vec<&Node<T>> {
        self.nodes.iter().filter(|&node| node.parent.is_none()).collect_vec()
    }

    pub fn add_node(
        &mut self,
        state: T,
        parent: Option<usize>,
        action_taken: Option<T::Move>,
        policy: f32,
    ) -> usize {
        let idx = self.nodes.len();
        let new_node = Node::new(state, idx, parent, action_taken, policy);
        self.nodes.push(new_node);
        idx
    }

    pub fn set_node(&mut self, node: &Node<T>) {
        self.nodes[node.idx] = node.clone();
    }

    pub fn get_node(&self, idx: usize) -> Node<T> {
        self.nodes.get(idx).unwrap().clone()
    }

    pub fn get_node_ref(&self, idx: usize) -> &Node<T> {
        self.nodes.get(idx).unwrap()
    }

    pub fn get_node_as_mut(&mut self, idx: usize) -> &mut Node<T> {
        &mut self.nodes[idx]
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn pretty_print(&self, index: usize, depth: usize, c: f32) {
        self._pretty_print(index, depth, 0, c)
    }


    fn _pretty_print(&self, index: usize, depth: usize, current_depth: usize, c: f32) {
        if current_depth > depth {
            return;
        }
    
        let node = self.get_node(index);
        let indent = "  ".repeat(current_depth);
    
        println!(
            "{}[{}] Action: {:?} \t\tVisits: {:.2} \tValue: {:.2} \tUCB: {:.5} \tWin_Pct: {:.3}",
            indent,
            index,
            node.action_taken,
            node.visits,
            node.value,
            node.ucb(self, c),
            node.win_pct()
        );
    
        for &child_index in &node.children {
            self._pretty_print(child_index, depth, current_depth + 1, c);
        }
    }
}