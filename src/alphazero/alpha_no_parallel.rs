use crate::{backgammon::backgammon_logic::Backgammon, mcts::alpha_mcts::alpha_mcts};

use super::alphazero::{AlphaZero, MemoryFragment};


impl AlphaZero {
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

}