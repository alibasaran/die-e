use std::fmt;

use crate::backgammon::{Backgammon, Actions};

use super::node::Node;

#[derive(Clone)]
pub struct NodeStore {
    nodes: Vec<Node>,
}

impl fmt::Display for NodeStore {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NodeStore<size={}>", self.nodes.len())
    }
}

impl NodeStore {
    pub fn new() -> Self {
        NodeStore { nodes: vec![] }
    }

    pub fn add_node(
        &mut self,
        state: Backgammon,
        parent: Option<usize>,
        action_taken: Option<Actions>,
        roll: Option<(u8, u8)>,
        policy: f32,
    ) -> usize {
        let idx = self.nodes.len();
        let new_node = if let Some(roll) = roll {
            Node::new_with_roll(state, idx, parent, action_taken, roll, policy)
        } else {
            Node::new(state, idx, parent, action_taken, policy)
        };
        self.nodes.push(new_node);
        idx
    }

    pub fn set_node(&mut self, node: &Node) {
        self.nodes[node.idx] = node.clone()
    }

    pub fn get_node(&self, idx: usize) -> Node {
        match self.nodes.get(idx) {
            None => { self.pretty_print(0, 0); panic!("there is no node at idx: {}, {}", idx, self) }
            Some(node) => node.clone()
        }
    }

    pub fn get_node_ref(&self, idx: usize) -> &Node {
        match self.nodes.get(idx) {
            None => { self.pretty_print(0, 0); panic!("there is no node at idx: {}, {}", idx, self) }
            Some(node) => node
        }
    }

    pub fn get_node_as_mut(&mut self, idx: usize) -> &mut Node {
        &mut self.nodes[idx]
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn pretty_print(&self, index: usize, depth: usize) {
        self._pretty_print(index, depth, 0)
    }


    fn _pretty_print(&self, index: usize, depth: usize, current_depth: usize) {
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
            node.ucb(self),
            node.win_pct()
        );
    
        for &child_index in &node.children {
            self._pretty_print(child_index, depth, current_depth + 1);
        }
    }
}