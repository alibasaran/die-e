use std::{cmp::Ordering, collections::HashSet, time::{SystemTime, UNIX_EPOCH}, sync::Mutex};

use arrayvec::ArrayVec;
use indicatif::{ProgressIterator, MultiProgress, ProgressBar};
use itertools::Itertools;

use rayon::prelude::{IntoParallelIterator, ParallelIterator, IntoParallelRefIterator};
use tch::Tensor;

use crate::{backgammon::Backgammon, alphazero::nnet::ResNet, constants::{DIRICHLET_ALPHA, DIRICHLET_EPSILON, DEVICE}, mcts::{noise::apply_dirichlet, utils::{turn_policy_to_probs_tensor, turn_policy_to_probs_tensor_parallel}}, MCTS_CONFIG};

use super::{node_store::NodeStore, node::Node, simple_mcts::backpropagate, utils::{turn_policy_to_probs, get_prob_tensor}};

pub struct TimeLogger {
    start_time: SystemTime,
    last_log: SystemTime
}

impl Default for TimeLogger {
    fn default() -> Self {
        TimeLogger { start_time: UNIX_EPOCH, last_log: UNIX_EPOCH }
    }
}

const LOCK_LOGGER: bool = false;

impl TimeLogger {
    pub fn start(&mut self) {
        if LOCK_LOGGER {
            return;
        }
        self.start_time = SystemTime::now();
        self.last_log = SystemTime::now();
        println!("Started TimeLogger")
    }

    pub fn log(&mut self, msg: &str) {
        if LOCK_LOGGER {
            return;
        }
        println!("{}: {} ms", msg, SystemTime::now().duration_since(self.last_log).unwrap().as_millis());
        self.last_log = SystemTime::now();
    }
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

pub fn apply_dirichlet_to_root(root_idx: usize, store: &mut NodeStore, net: &ResNet, state: &Backgammon) {
    let mut root_node = store.get_node(root_idx);
    
    let (mut policy, _) = net.forward_t(&state.as_tensor().to_device(*DEVICE), false);

    policy = policy.softmax(1, None)
        .permute([1, 0]);
    let policy_vec = turn_policy_to_probs(&policy, &root_node);
    root_node.alpha_expand(store, policy_vec);
    if root_node.children.len() < 2 {
        // Don't apply dirichlet on a node with a single child
        return
    }
    // Set visits to 1
    root_node.visits = 1.;
    // Apply dirichlet to root policies
    root_node.apply_dirichlet(DIRICHLET_ALPHA, DIRICHLET_EPSILON, store);
    store.set_node(&root_node)
}

pub fn alpha_mcts(state: &Backgammon, net: &ResNet) -> Option<Tensor> {
    // Set no_grad_guard
    let _guard = tch::no_grad_guard();
    
    // Check if game already is terminal at root
    if Backgammon::check_win_without_player(state.board).is_some() {
        return None;
    }
    let mut store = NodeStore::new();
    let roll = state.roll;
    let root_node_idx = store.add_node(*state, None, None, Some(roll), 0.0);
    
    apply_dirichlet_to_root(root_node_idx, &mut store, net, state);
    
    let pb_iter = (0..MCTS_CONFIG.iterations).progress().with_message("AlphaMCTS");
    for _ in pb_iter {
        // Don't forget to save the node later into the store
        let idx = alpha_select_leaf_node(root_node_idx, &store);
        let mut selected_node = store.get_node(idx);

        let value: f32;

        if !selected_node.is_terminal() {
            let (mut policy, eval) = net.forward_t(&selected_node.state.as_tensor().to_device(*DEVICE), false);

            policy = policy.softmax(1, None)
                .permute([1, 0]);
            let policy_vec = turn_policy_to_probs(&policy, &selected_node);
            selected_node.alpha_expand(&mut store, policy_vec);
            value = eval.double_value(&[0]) as f32;
        } else {
            value = ((Backgammon::check_win_without_player(selected_node.state.board).unwrap() + 1) / 2) as f32;
        }

        backpropagate(idx, value, &mut store);
    }
    get_prob_tensor(state, root_node_idx, &store)
}


/*
    Similar to alpha_mcts, however this function mutates the NodeStore rather than returning probabilities
*/
pub fn alpha_mcts_parallel(store: &mut NodeStore, states: &[Backgammon], net: &ResNet, pb: Option<ProgressBar>, bypass_empty_check: bool) {
    // Set no_grad_guard
    let _guard = tch::no_grad_guard();
    assert!(store.is_empty() || bypass_empty_check, "AlphaMCTS paralel expects an empty store");
    
    // Convert all states a tensor
    let states_vec = states.iter().map(|state| state.as_tensor()).collect_vec();
    let states_tensor = Tensor::stack(
        &states_vec,
        0
    ).squeeze_dim(1).to_device(*DEVICE);

    // Get policy tensor
    let policy = net.forward_policy(&states_tensor, false);

    // Apply dirichlet to root policy tensor
    let policy_dir = apply_dirichlet(&policy, DIRICHLET_ALPHA, DIRICHLET_EPSILON);

    // Create root node for each game state
    for (_, state) in states.iter().enumerate() {
        store.add_node(*state, None, None, Some(state.roll), 0.0);
    }

    // Expand root node for each game state
    // Move policy to CPU because alpha expand makes multiple double_value(idx) calls,
    // Takes too long when tensors are in GPU
    let prob_tensor = turn_policy_to_probs_tensor_parallel(store, (0..states.len()).collect_vec(), &policy_dir)
        .to_device(tch::Device::Cpu);
    for i in 0..states.len() {
        // Create root node for each game state
        let mut root = store.get_node(i);

        root.visits = 1.;
        // let prob_tensor = turn_policy_to_probs_tensor(&policy_dir.get(i as i64), &root);
        // Expand root
        root.alpha_expand_tensor(store, &prob_tensor.get(i as i64));
    }
    
    /*
        Create two vectors:
            - Games:
                - values ranging from 0 to N_SELF_PLAY_BATCHES
                - used to track how many games are still not completed
            - Selected_nodes:
                - N_SELF_PLAY_BATCHES node values
                - selected_nodes[i] refers to the node that was last selected for game[i]

    */
    let mut games: HashSet<usize> = HashSet::from_iter(0..states.len());
    let mut selected_nodes_idxs = vec![0; states.len()];

    let pb = match pb {
        Some(p) => p,
        None => ProgressBar::new(MCTS_CONFIG.iterations as u64)
    };

    for _ in 0..MCTS_CONFIG.iterations {
        let mut ended_games = vec![];
        for game_idx in games.iter() {
            let idx = alpha_select_leaf_node(*game_idx, store);
            let selected_node = store.get_node(idx);
    
            if selected_node.is_terminal() {
                let value = ((Backgammon::check_win_without_player(selected_node.state.board).unwrap() + 1) / 2) as f32;
                backpropagate(idx, value, store);
                ended_games.push(*game_idx);
            } else {
                selected_nodes_idxs[*game_idx] = selected_node.idx
            }
        }

        // Remove finished games
        for game_to_remove in ended_games {
            games.remove(&game_to_remove);
        }

        // No games to search, end search
        if games.is_empty() {
            return
        }
        
        // Convert ongoing (not terminal) games into a tensor
        let selected_states_vec = games.iter().map(|idx| {
            let node = store.get_node(selected_nodes_idxs[*idx]);
            node.state.as_tensor()
        }).collect_vec();
        let selected_states_tensor = Tensor::stack(
            &selected_states_vec,
            0
        ).squeeze_dim(1).to_device(*DEVICE);

        // Calculate policies
        let (policy, eval) = net.forward_t(&selected_states_tensor, false);
        
        // Expand and backprop selected nodes with their respective calculated policies and evals
        let eval = eval.to_device(tch::Device::Cpu);
        let policy = policy.to_device(tch::Device::Cpu);
        for (i, game) in games.iter().enumerate() {
            let mut node = store.get_node(selected_nodes_idxs[*game]);
            let (policy_i, eval_i) = (policy.get(i as i64), eval.get(i as i64));
            let prob_tensor = turn_policy_to_probs_tensor(&policy_i, &node);
            // Expand root
            node.alpha_expand_tensor(store, &prob_tensor);
            let value = eval_i.double_value(&[0]) as f32;
            backpropagate(node.idx, value, store);
        }
        pb.inc(1)
    }
}
