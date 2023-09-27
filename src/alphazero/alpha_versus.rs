use std::{path::Path, collections::HashMap};

use indicatif::{ProgressBar, ProgressStyle};

use rand::{thread_rng, seq::SliceRandom};
use tch::nn::VarStore;

use crate::{constants::DEVICE, backgammon::backgammon_logic::Backgammon, mcts::{alpha_mcts::{alpha_mcts, alpha_mcts_parallel}, node_store::NodeStore, utils::get_prob_tensor_parallel}};

use super::{alphazero::AlphaZero, nnet::ResNet};

impl AlphaZero {
    /**
     * Plays against the current best model, saves if 55% better
     */
    pub fn play_vs_best_model(&self) {
        let mut vs_best_model = VarStore::new(*DEVICE);
        let best_model_path = "./models/best_model.ot";
        match vs_best_model.load(Path::new(best_model_path)) {
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
        let is_model_better = match self.model_vs_model_parallel(&self.model, &ResNet::new(vs_best_model)) {
            Some(1) => {self.pb.println("new model was better!").unwrap(); true},
            Some(2) => {self.pb.println("current best model is still better!").unwrap(); false},
            None => {self.pb.println("new model vs current best was inconclusive, keeping current best!").unwrap(); false}
            Some(_) => unreachable!(),
        };
        if is_model_better {
            match self.model.vs.save(best_model_path) {
                Ok(_) => self.pb.println("saved new best model").unwrap(),
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
        let total_games = 10;
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
                    alpha_mcts(&bg, model1, &self.mcts_config)
                } else {
                    alpha_mcts(&bg, model2, &self.mcts_config)
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
        if model1_wins >= 6 {
            Some(1)
        } else if model1_wins <= 4 {
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

        let mut games: HashMap<usize, Backgammon> = HashMap::from_iter((0..total_games)
            .map(|idx| {
                let mut bg = Backgammon::new();
                if idx >= total_games / 2 {
                    bg.skip_turn();
                }
                bg.roll_die();
                bg.id = idx;
                (idx, bg)
            }));

        let pb_games = self.pb.add(
            ProgressBar::new(total_games as u64)
                .with_message("Finished games")
                .with_style(sty.clone()),
        );
        let mut mcts_count = 0;
        while !games.is_empty() {
            pb_games.set_position((total_games - games.len()).try_into().unwrap());
            pb_games.set_message(format!("on mcts run: {}", mcts_count + 1));

            // Partition states by player
            let (states_m1, states_m2): (Vec<Backgammon>, Vec<Backgammon>) =
                games.values().partition(|state| state.player == player_m1);
            let mut store1 = NodeStore::new();
            let mut store2 = NodeStore::new();
            // Process states by model depending on their player
            if !states_m1.is_empty() {
                let pb_mcts_m1 = self.pb.add(
                    ProgressBar::new(self.mcts_config.iterations.try_into().unwrap())
                    .with_message(format!("Model 1 AlphaMCTS - {} games", states_m1.len()))
                    .with_style(sty.clone()),

                );
                alpha_mcts_parallel(&mut store1, &states_m1, model1, &self.mcts_config,  Some(pb_mcts_m1));
            }
            if !states_m2.is_empty() {
                let pb_mcts_m2 = self.pb.add(
                    ProgressBar::new(self.mcts_config.iterations.try_into().unwrap())
                        .with_message(format!("Model 2 AlphaMCTS - {} games", states_m2.len()))
                        .with_style(sty.clone()),
                );
                alpha_mcts_parallel(&mut store2, &states_m2, model2, &self.mcts_config, Some(pb_mcts_m2));
            }
            mcts_count += 1;

            // Get all root nodes in the store
            let mut roots = store1.get_root_nodes();
            let store2_roots = store2.get_root_nodes();
            roots.extend(store2_roots);
            
            let prob_tensor = get_prob_tensor_parallel(&roots)
                .pow_(1.0 / self.config.temperature)
                .to_device(tch::Device::Cpu);

            let mut games_to_remove = vec![];
            for (processed_idx, &root) in roots.iter().enumerate() {
                let initial_idx = root.state.id;
                let curr_prob_tensor = prob_tensor.get(processed_idx as i64);

                let state = games.get_mut(&initial_idx).unwrap();

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
                if mcts_count >= self.mcts_config.simulate_round_limit {
                    games_to_remove.push(initial_idx);
                    let choices = vec![-1, 1];
                    let rand_winner = choices.choose(&mut thread_rng()).unwrap();
                    if *rand_winner == player_m1 {
                        m1_wins += 1.
                    }
                }
            }
            for i in games_to_remove.iter() {
                games.remove(i);
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
