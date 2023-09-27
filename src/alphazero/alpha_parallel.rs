use std::{fs, path::Path, collections::HashMap};

use indicatif::{ProgressStyle, ProgressBar};
use itertools::Itertools;

use crate::{backgammon::backgammon_logic::Backgammon, MCTS_CONFIG, mcts::{node_store::NodeStore, alpha_mcts::alpha_mcts_parallel, utils::get_prob_tensor_parallel}};

use super::alphazero::{AlphaZero, MemoryFragment};
use nanoid::nanoid;


impl AlphaZero {
    
    pub fn learn_parallel(&mut self) {
        let run_id = nanoid!();
        let runpath_base = format!("./data/run-{}", &run_id);
        println!("Staring up run with run_id: {}", &run_id);
        let _ = fs::create_dir(&runpath_base);
        let sty = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-");

        let pb_learn = self.pb.add(
            ProgressBar::new(self.config.learn_iterations as u64)
                .with_message("Learn Parallel!")
                .with_style(sty.clone()),
        );

        let _ = self.pb.println("Starting learn parallel...");
        let pb_self_play = self.pb.add(
            ProgressBar::new(self.config.self_play_iterations as u64)
                .with_message("Self play")
                .with_style(sty.clone()),
        );
        let pb_train = self.pb.add(
            ProgressBar::new(self.config.num_epochs as u64)
                .with_message("Training")
                .with_style(sty.clone()),
        );

        for l_i in 0..self.config.learn_iterations {
            // Create dir for current learn iteration
            let lrn_path = format!("{}/lrn-{}", &runpath_base, l_i);
            let _ = fs::create_dir(&lrn_path);

            pb_self_play.reset();
            // Get samples for training
            let mut memory: Vec<MemoryFragment> = vec![];
            for sp_i in 0..self.config.self_play_iterations {
                pb_self_play.set_message(format!("Self-play iteration #{}", sp_i + 1));

                let mut res = self.self_play_parallel();
                memory.append(&mut res);
                pb_self_play.set_message(format!("Saving training data... Self-play iteration #{}", sp_i + 1));

                // Make self play dir
                let sp_dir_path = format!("{}/sp-{}", &lrn_path, sp_i);
                let _ = fs::create_dir(&sp_dir_path);
                self.save_training_data(&memory, Path::new(&sp_dir_path));
                pb_self_play.set_message(format!("Self-play iteration #{} complete, saved training data", sp_i + 1));
                pb_self_play.inc(1);

            }
            pb_self_play.set_message("Self-play finished");

            // Train
            pb_train.reset();
            for _ in 0..self.config.num_epochs {
                self.train(&mut memory);
                pb_train.inc(1);
            }
            // Save model
            let model_save_path = format!("./models/model_{}.ot", l_i);
            match self.model.vs.save(&model_save_path) {
                Ok(_) => println!(
                    "Iteration {} saved successfully, path: {}",
                    l_i, &model_save_path
                ),
                Err(e) => println!(
                    "Unable to save model: {}, caught error: {}",
                    &model_save_path, e
                ),
            }
            self.play_vs_best_model();
            pb_learn.inc(1);
        }
    }

    pub fn self_play_parallel(&self) -> Vec<MemoryFragment> {
        let n_batches: usize = self.config.num_self_play_batches;
        let mut states: HashMap<usize, (usize, Backgammon)> = (0..n_batches)
            .map(|idx| {
                let mut bg = Backgammon::new();
                bg.roll_die();
                (idx, (idx, bg))
            })
            .collect();

        let mut memories = Vec::from_iter((0..n_batches).map(|_| Vec::<MemoryFragment>::new()));
        let mut all_memories = vec![];

        let mut n_rounds = vec![0; n_batches];

        let sty = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-");

        let self_play_pb = self.pb.add(ProgressBar::new(n_batches as u64));
        self_play_pb.set_message(format!("Self play - {} batches", n_batches));
        self_play_pb.set_style(sty.clone());

        let mut mcts_runs = 0;
        while !states.is_empty() {
            self_play_pb.set_position((n_batches - states.len()) as u64);
            self_play_pb.set_message(format!("Self play - {} batches, on mcts run {}", n_batches, mcts_runs));

            // Mutates store, does not return anything
            let mut store = NodeStore::new();
            let state_to_process = states.iter().map(|tup| tup.1 .1).collect_vec();
            let mcts_pb = self.pb.add(
                ProgressBar::new(MCTS_CONFIG.iterations as u64)
                    .with_style(sty.clone())
                    .with_message("Alpha mcts parallel")
                    .with_finish(indicatif::ProgressFinish::AndClear),
            );
            // Fill store with games
            alpha_mcts_parallel(&mut store, &state_to_process, &self.model, Some(mcts_pb));
            mcts_runs += 1;
            // processed idx: the index of the state in remaining processed states
            // init_idx: the index of the state in the initial creation of the states vector
            // state: the backgammon state itself
            // states_to_remove: list of finished states to remove after each iteration
            let mut states_to_remove = vec![];
            // We can do store.get_root_nodes(); as well but since we know the first 0..states.len() are the roots it is faster
            let roots = (0..states.len()).map(|i| store.get_node_ref(i)).collect_vec();
            let prob_tensor = get_prob_tensor_parallel(&roots)
                .pow_(1.0 / self.config.temperature) // Apply temperature
                .to_device(tch::Device::Cpu); // Move to CPU for faster access

            for (processed_idx, (init_idx, state)) in states.values_mut().enumerate() {
                let curr_prob_tensor = prob_tensor.get(processed_idx as i64);

                // Check round limit
                if n_rounds[*init_idx] >= MCTS_CONFIG.simulate_round_limit {
                    let curr_memory = memories[*init_idx].iter().map(|mem| MemoryFragment {
                        outcome: 0,
                        ps: mem.ps.shallow_clone(),
                        state: mem.state.shallow_clone(),
                    });
                    all_memories.extend(curr_memory);
                    states_to_remove.push(*init_idx);
                }

                // If prob tensor of the current state is all zeros then skip turn, has_children check just in case
                if !curr_prob_tensor.sum(None).is_nonzero() || store.get_node(processed_idx).children.is_empty() {
                    n_rounds[*init_idx] += 1;
                    state.skip_turn();
                    continue;
                }

                // Select an action from probabilities
                let selected_action = Self::weighted_select_tensor_idx(&curr_prob_tensor);

                // Save results to memory
                memories[*init_idx].push(MemoryFragment {
                    outcome: state.player,
                    ps: curr_prob_tensor,
                    state: state.as_tensor(),
                });

                // Decode and play selected action
                let decoded_action = state.decode(selected_action as u32);
                state.apply_move(&decoded_action);

                // Increment round
                n_rounds[*init_idx] += 1;

                if let Some(winner) = Backgammon::check_win_without_player(state.board) {
                    let curr_memory = memories[*init_idx].iter().map(|mem| MemoryFragment {
                        outcome: if mem.outcome == winner { 1 } else { -1 },
                        ps: mem.ps.shallow_clone(),
                        state: mem.state.shallow_clone(),
                    });
                    all_memories.extend(curr_memory);
                    states_to_remove.push(*init_idx);
                }
            }
            // remove finished states
            for state_idx in states_to_remove {
                states.remove(&state_idx);
            }
        }
        all_memories
    }

}