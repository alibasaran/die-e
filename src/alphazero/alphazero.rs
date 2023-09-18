use arrayvec::ArrayVec;
use indicatif::{MultiProgress, ProgressBar, ProgressIterator, ProgressStyle};
use itertools::{multiunzip, Itertools};
use rand::{distributions::WeightedIndex, prelude::Distribution, seq::SliceRandom, thread_rng};
use serde::Serialize;
use std::{
    cmp::min,
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};
use tch::{
    nn::{self, Adam, Optimizer, OptimizerConfig, VarStore},
    Tensor,
};

use super::nnet::ResNet;

use crate::{
    backgammon::Backgammon,
    constants::{DEFAULT_TYPE, DEVICE, N_SELF_PLAY_BATCHES},
    mcts::{
        alpha_mcts::{alpha_mcts, alpha_mcts_parallel, TimeLogger},
        node_store::NodeStore,
        utils::{get_prob_tensor, get_prob_tensor_parallel},
    },
    MCTS_CONFIG,
};

#[derive(Debug)]
pub struct AlphaZeroConfig {
    pub temperature: f64,
    pub learn_iterations: usize,
    pub self_play_iterations: usize,
    pub num_epochs: usize,
    pub batch_size: usize,
}

pub struct AlphaZero {
    model: ResNet,
    optimizer: Optimizer,
    config: AlphaZeroConfig,
    pb: MultiProgress,
}
#[derive(Debug)]
pub struct MemoryFragment {
    pub outcome: i8,   // Outcome of game
    pub ps: Tensor,    // Probabilities
    pub state: Tensor, // Encoded game state
}

/*
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
*/

// struct Game {
//     idx: usize,
//     state: Backgammon,
// }

// impl Game {
//     fn new(idx: usize, state: Backgammon) -> Self {
//         Game {
//             idx,
//             state
//         }
//     }
// }

impl AlphaZero {
    pub fn new(config: AlphaZeroConfig) -> Self {
        let vs = nn::VarStore::new(*DEVICE);
        let opt = Adam::default().wd(1e-4).build(&vs, 1e-4).unwrap();

        println!(
            "\n
        Device: {:?}
        AlphaZero initialized:
        \n{:?}
        \nMCTS Config:
        \n{:?}
        ",
            *DEVICE, config, MCTS_CONFIG
        );

        AlphaZero {
            model: ResNet::new(vs),
            optimizer: opt,
            config,
            pb: MultiProgress::new(),
        }
    }

    pub fn save_training_data(&self, data: &[MemoryFragment], name: &str) {
        let (outcomes, ps_values, states): (Vec<i8>, Vec<Tensor>, Vec<Tensor>) =
            multiunzip(data.iter().map(|fragment| {
                (
                    fragment.outcome,
                    fragment.ps.shallow_clone(),
                    fragment.state.shallow_clone(),
                )
            }));
        // One channel for outcomes, one channel for ps, one channel for states,
        // Because mps is not supported we convert to cpu
        let ps = Tensor::stack(&ps_values, 0).to_device(tch::Device::Cpu).unsqueeze(2);
        let states = Tensor::stack(&states, 0).to_device(tch::Device::Cpu).squeeze();
        let outcomes = Tensor::from_slice(&outcomes);

        let path_base = format!("./data/{}", name);
        let path = Path::new(&path_base);
        let parent = path.parent().unwrap();
        if !parent.exists() {
            let _ = fs::create_dir(parent);
        }
        let path = format!("./data/{}", name);
        match (
            ps.save(path.clone() + "_ps"),
            states.save(path.clone() + "_states"),
            outcomes.save(path.clone() + "_outcomes"),
        ) {
            (Ok(_), Ok(_), Ok(_)) => (),
            _ => panic!("unable to save training data!"),
        }
    }

    pub fn load_training_data(&self, name: &str) -> Vec<MemoryFragment> {
        let path = Path::new("./data").join(name);
        if !path.parent().unwrap().exists() {
            panic!("there is no path: {:?}", &path)
        }
        let path = path.to_str().unwrap().to_owned();
        let ps = Tensor::load(path.clone() + "_ps").unwrap();
        let states = Tensor::load(path.clone() + "_states").unwrap();
        let outcomes = Tensor::load(path.clone() + "_outcomes").unwrap();

        let data_size = ps.size()[0];
        println!("{:?}", data_size);
        (0..data_size).map(|data_idx| {
            MemoryFragment {
                outcome: outcomes.get(data_idx).int64_value(&[]) as i8,
                ps: ps.get(data_idx).squeeze().shallow_clone(),
                state: states.get(data_idx).unsqueeze(2).shallow_clone(),
            }
        }).collect_vec()
    }

    pub fn self_play(&self) -> Vec<MemoryFragment> {
        println!("Started self play");
        let mut bg = Backgammon::new();
        bg.roll_die();

        let mut memory: Vec<MemoryFragment> = vec![];
        // Is second play is a workaround for the case where doubles are rolled
        // Ex. when a player rolls 6,6 they can play 6 4-times
        // rather than the usual two that would be played on a normal role
        // So we feed doubles into the network as two back to back normal roles from the same player
        loop {
            println!("Rolled die: {:?}", bg.roll);
            println!("Player: {}", bg.player);
            bg.display_board();
            // Get probabilities from mcts
            let mut pi = match alpha_mcts(&bg, &self.model) {
                Some(pi) => pi,
                None => {
                    println!("No valid moves!");
                    bg.player *= -1;
                    bg.roll_die();
                    continue;
                }
            };

            // Apply temperature to pi
            let temperatured_pi = pi.pow_(1.0 / self.config.temperature);
            // Select an action from probabilities
            let selected_action = Self::weighted_select_tensor_idx(&temperatured_pi);

            // Save results to memory
            memory.push(MemoryFragment {
                outcome: bg.player,
                ps: pi,
                state: bg.as_tensor(),
            });

            // Decode and play selected action
            let decoded_action = bg.decode(selected_action as u32);
            println!("Played action: {:?}\n\n", decoded_action);
            bg.apply_move(&decoded_action);

            if let Some(winner) = Backgammon::check_win_without_player(bg.board) {
                return memory
                    .iter()
                    .map(|mem| MemoryFragment {
                        outcome: if mem.outcome == winner { 1 } else { -1 },
                        ps: mem.ps.shallow_clone(),
                        state: mem.state.shallow_clone(),
                    })
                    .collect();
            }
        }
    }

    pub fn self_play_parallel(&self) -> Vec<MemoryFragment> {
        let mut states: HashMap<usize, (usize, Backgammon)> = (0..N_SELF_PLAY_BATCHES)
            .map(|idx| {
                let mut bg = Backgammon::new();
                bg.roll_die();
                (idx, (idx, bg))
            })
            .collect();

        let mut memories: [Vec<MemoryFragment>; N_SELF_PLAY_BATCHES] =
            std::array::from_fn(|_| vec![]);
        let mut all_memories = vec![];

        let mut n_rounds = [0; N_SELF_PLAY_BATCHES];

        let sty = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-");

        let self_play_pb = self.pb.add(ProgressBar::new(N_SELF_PLAY_BATCHES as u64));
        self_play_pb.set_message(format!("Self play - {} batches", N_SELF_PLAY_BATCHES));
        self_play_pb.set_style(sty.clone());

        let mut mcts_runs = 0;
        while !states.is_empty() {
            self_play_pb.set_position((N_SELF_PLAY_BATCHES - states.len()) as u64);
            self_play_pb.set_message(format!("Self play - {} batches, on mcts run {}", N_SELF_PLAY_BATCHES, mcts_runs));

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
            let prob_tensor = get_prob_tensor_parallel((0..states.len()).collect_vec(), &store)
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

    fn train(&mut self, memory: &mut Vec<MemoryFragment>) {
        let mut rng = thread_rng();
        memory.shuffle(&mut rng);
        for batch_idx in (0..memory.len()).step_by(self.config.batch_size) {
            let sample = &memory[batch_idx..min(batch_idx + self.config.batch_size, memory.len())];
            let (outcomes, ps_values, states): (Vec<i8>, Vec<Tensor>, Vec<Tensor>) =
                multiunzip(sample.iter().map(|fragment| {
                    (
                        fragment.outcome,
                        fragment.ps.shallow_clone(),
                        fragment.state.shallow_clone(),
                    )
                }));

            // Format tensors to process
            let outcome_tensor = Tensor::from_slice(&outcomes).unsqueeze(1).to_device_(
                *DEVICE,
                DEFAULT_TYPE,
                true,
                false,
            );

            let ps_tensor = Tensor::stack(&ps_values, 0).to_device_(
                *DEVICE,
                DEFAULT_TYPE,
                true,
                false,
            );

            let state_tensor = Tensor::stack(&states, 0).squeeze_dim(1).to_device_(
                *DEVICE,
                DEFAULT_TYPE,
                true,
                false,
            );

            let (out_policy, out_value) = self.model.forward_t(&state_tensor, true);
            let out_policy = out_policy.squeeze();

            // Calculate loss
            let policy_loss = out_policy.cross_entropy_loss::<Tensor>(
                &ps_tensor,
                None,
                tch::Reduction::Mean,
                -100,
                0.0,
            );
            let outcome_loss = out_value.mse_loss(&outcome_tensor, tch::Reduction::Mean);
            let loss = policy_loss + outcome_loss;

            self.optimizer.zero_grad();
            loss.backward();
            self.optimizer.step();
        }
    }

    pub fn learn(&mut self) {
        for i in 0..self.config.learn_iterations {
            let mut memory: Vec<MemoryFragment> = vec![];
            for _ in 0..self.config.self_play_iterations {
                let mut res = self.self_play();
                memory.append(&mut res);
            }
            for _ in 0..self.config.num_epochs {
                self.train(&mut memory);
            }
            let model_save_path = format!("./models/model_{}.ot", i);
            match self.model.vs.save(&model_save_path) {
                Ok(_) => println!(
                    "Iteration {} saved successfully, path: {}",
                    i, &model_save_path
                ),
                Err(e) => println!(
                    "Unable to save model: {}, caught error: {}",
                    &model_save_path, e
                ),
            }
        }
    }

    pub fn learn_parallel(&mut self) {
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
            pb_self_play.reset();
            // Get samples for training
            let mut memory: Vec<MemoryFragment> = vec![];
            for sp_i in 0..self.config.self_play_iterations {
                pb_self_play.set_message(format!("Self-play iteration #{}", sp_i + 1));

                let mut res = self.self_play_parallel();
                memory.append(&mut res);
                pb_self_play.set_message(format!("Saving training data... Self-play iteration #{}", sp_i + 1));
                self.save_training_data(&memory, &format!("lrn-{}/sp-{}", l_i, sp_i));
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

    pub fn weighted_select_tensor_idx(pi: &Tensor) -> usize {
        let mut rng = thread_rng();
        let weights_iter = match pi.iter::<f64>() {
            Ok(iter) => iter,
            Err(err) => panic!("cannot convert pi to iterator, got error: {}", err),
        };
        let dist = WeightedIndex::new(weights_iter).unwrap();
        dist.sample(&mut rng)
    }

    /**
     * Playes 100 games, model1 vs model2
     * Returns None if no model achieves 55% winrate
     * Returns the model if a model does
     */
    pub fn model_vs_model(&self, model1: &ResNet, model2: &ResNet) -> Option<usize> {
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
}
