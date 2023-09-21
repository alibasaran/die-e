use std::{path::Path, collections::HashSet};

use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use rand::{thread_rng, seq::SliceRandom};
use rand_distr::{WeightedIndex, Distribution};
use tch::{nn::VarStore, Tensor};

use crate::{constants::DEVICE, backgammon::Backgammon, mcts::{alpha_mcts::{alpha_mcts, alpha_mcts_parallel}, node_store::NodeStore, utils::get_prob_tensor_parallel}, MCTS_CONFIG};

use super::{alphazero::AlphaZero, nnet::ResNet};

impl AlphaZero {
    /**
     * Plays against the current best model, saves if 55% better
     */
    pub fn play_vs_best_model(&self) {
        let mut vs = VarStore::new(*DEVICE);
        let best_model_path = "./models/best_model.ot";
        match vs.load(Path::new(best_model_path)) {
            Ok(_) => (),
            Err(_) => match self.model.vs.save(best_model_path) {
                Ok(_) => {
                    self.pb
                        .println(format!(
                            "No best model was found, saved current model successfully, path: {}",
                            &best_model_path
                        ))
                        .unwrap();
                    return;
                }
                Err(e) => {
                    self.pb
                        .println(format!(
                            "Unable to save model: {}, caught error: {}",
                            &best_model_path, e
                        ))
                        .unwrap();
                    return;
                }
            },
        }
        // Create vs copy because we move the vs into the ResNet
        let mut vs_copy = VarStore::new(*DEVICE);
        vs_copy.copy(&vs).unwrap();

        let is_model_better = match self.model_vs_model(&self.model, &ResNet::new(vs)) {
            Some(1) => true,
            Some(2) | None => false,
            Some(_) => unreachable!(),
        };
        if is_model_better {
            match vs_copy.save(best_model_path) {
                Ok(_) => self.pb.println("new model was better! saved").unwrap(),
                Err(_) => self
                    .pb
                    .println("new model was better! couldn't save :(")
                    .unwrap(),
            }
        }
    }

    pub fn play_vs_model(&self, other_model: &ResNet) -> Option<usize> {
        self.model_vs_model_parallel(&self.model, other_model)
    }

    /**
     * Playes 100 games, model1 vs model2
     * Returns None if no model achieves 55% winrate
     * Returns the model if a model does
     */
    fn model_vs_model(&self, model1: &ResNet, model2: &ResNet) -> Option<usize> {
        let total_games = 100;
        let mut model1_wins = 0;

        for i in 0..total_games {
            let mut bg = Backgammon::new();
            // Make half of the games start with the second model
            if i < 50 {
                bg.skip_turn()
            }
            bg.roll_die();

            loop {
                // Get probabilities from mcts
                let mcts_res = if bg.player == -1 {
                    alpha_mcts(&bg, model1)
                } else {
                    alpha_mcts(&bg, model2)
                };
                let mut pi = match mcts_res {
                    Some(pi) => pi,
                    None => {
                        bg.player *= -1;
                        bg.roll_die();
                        continue;
                    }
                };

                // Apply temperature to pi
                let temperatured_pi = pi.pow_(1.0 / self.config.temperature);
                // Select an action from probabilities
                let selected_action = Self::weighted_select_tensor_idx(&temperatured_pi);

                // Decode and play selected action
                let decoded_action = bg.decode(selected_action as u32);
                bg.apply_move(&decoded_action);

                if let Some(winner) = Backgammon::check_win_without_player(bg.board) {
                    if winner == -1 {
                        model1_wins += 1;
                    }
                    break;
                }
            }
        }
        if model1_wins >= 55 {
            Some(1)
        } else if model1_wins <= 45 {
            Some(2)
        } else {
            None
        }
    }

    /**
     * Playes 100 games, model1 vs model2
     * Returns None if no model achieves 55% winrate
     * Returns the model if a model does
     */
    pub fn model_vs_model_parallel(&self, model1: &ResNet, model2: &ResNet) -> Option<usize> {
        let total_games = 100;
        let mut m1_wins = 0.;
        // games where it is model1's turn
        // games where it is model2's turn
        let player_m1 = -1;

        let sty = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap();

        let mut games = (0..total_games)
            .map(|idx| {
                let mut bg = Backgammon::new();
                if idx >= total_games / 2 {
                    bg.skip_turn();
                }
                bg.roll_die();
                bg.id = idx;
                bg
            })
            .collect_vec();

        let mut ongoing_games_idxs: HashSet<usize> = HashSet::from_iter(0..total_games);

        let pb_games = self.pb.add(
            ProgressBar::new(total_games as u64)
                .with_message("Finished games")
                .with_style(sty.clone()),
        );
        let mut mcts_count = 0;
        while !ongoing_games_idxs.is_empty() {
            pb_games.set_position((games.len() - ongoing_games_idxs.len()).try_into().unwrap());
            let ongoing_games = ongoing_games_idxs.iter().map(|idx| games[*idx]);
            // Partition states by player
            let (states_m1, states_m2): (Vec<Backgammon>, Vec<Backgammon>) =
                ongoing_games.partition(|state| state.player == player_m1);
            let mut store1 = NodeStore::new();
            let mut store2 = NodeStore::new();
            // Process states by model depending on their player
            if !states_m1.is_empty() {
                alpha_mcts_parallel(&mut store1, &states_m1, model1, None, true);
            }
            if !states_m2.is_empty() {
                alpha_mcts_parallel(&mut store2, &states_m2, model2, None, true);
            }
            mcts_count += 1;

            // Get all root nodes in the store
            let mut roots = store1.get_root_nodes();
            roots.extend(store2.get_root_nodes());

            let prob_tensor = get_prob_tensor_parallel(&roots)
                .pow_(1.0 / self.config.temperature)
                .to_device(tch::Device::Cpu);

            let mut games_to_remove = vec![];
            for (processed_idx, &root) in roots.iter().enumerate() {
                let curr_prob_tensor = prob_tensor.get(processed_idx as i64);
                let initial_idx = root.state.id;
                let state = games.get_mut(initial_idx).unwrap();

                // If prob tensor of the current state is all zeros then skip turn, has_children check just in case
                if !curr_prob_tensor.sum(None).is_nonzero() || root.children.is_empty() {
                    state.skip_turn();
                    continue;
                }

                // Select an action from probabilities
                let selected_action = Self::weighted_select_tensor_idx(&curr_prob_tensor);

                // Decode and play selected action
                let decoded_action = state.decode(selected_action as u32);
                state.apply_move(&decoded_action);

                if let Some(winner) = Backgammon::check_win_without_player(state.board) {
                    if winner == player_m1 {
                        m1_wins += 1.
                    }
                    games_to_remove.push(initial_idx);
                }
                if mcts_count >= MCTS_CONFIG.simulate_round_limit {
                    games_to_remove.push(initial_idx);
                    let choices = vec![-1, 1];
                    let rand_winner = choices.choose(&mut thread_rng()).unwrap();
                    if *rand_winner == player_m1 {
                        m1_wins += 1.
                    }
                }
            }
            for i in games_to_remove.iter() {
                ongoing_games_idxs.remove(i);
            }
        }
        let winrate = m1_wins / total_games as f64;
        if winrate >= 0.55 {
            Some(1)
        } else if m1_wins <= 0.45 {
            Some(2)
        } else {
            None
        }
    }
}
