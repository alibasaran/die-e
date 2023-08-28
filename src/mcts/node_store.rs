use crate::backgammon::{Backgammon, Actions};

use super::node::Node;

#[derive(Clone)]
pub struct NodeStore {
    nodes: Vec<Node>,
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

    pub fn set_node(&mut self, node: &Node) {
        self.nodes[node.idx] = node.clone()
    }

    pub fn get_node(&self, idx: usize) -> Node {
        self.nodes[idx].clone()
    }

    pub fn get_node_as_mut(&mut self, idx: usize) -> &mut Node {
        &mut self.nodes[idx]
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