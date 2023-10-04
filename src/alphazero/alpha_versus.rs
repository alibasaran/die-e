use std::{collections::HashMap, path::{Path, PathBuf}};

use indicatif::{ProgressBar, ProgressStyle};

use rand::{seq::SliceRandom, thread_rng};
use tch::nn::VarStore;

use crate::{
    backgammon::backgammon_logic::{Backgammon},
    base::LearnableGame,
    constants::DEVICE,
    mcts::{
        alpha_mcts::{alpha_mcts, alpha_mcts_parallel},
        node_store::NodeStore,
        utils::get_prob_tensor_parallel,
    }, versus::{play, Player, Agent},
};

use super::{alphazero::AlphaZero, nnet::ResNet};

impl AlphaZero {
    /**
     * Plays against the current best model, saves if 55% better
     */
    pub fn play_vs_best_model<T: LearnableGame>(&self) {
        let best_model_path_str = format!("./models/{}/best_model.ot", T::name());
        let best_model_path = PathBuf::from(best_model_path_str);
        if !best_model_path.exists() {
            println!("No best model was found, saving current model as best...");
            match self.model.vs.save(&best_model_path) {
                Ok(_) => println!("model saved!"),
                Err(e) => println!("unable to save model, caught error: {}", e),
            }
            return;
        }
        let nnet_best = ResNet::from_path::<T>(&best_model_path);
        // Create vs copy because we move the vs into the ResNet
        let is_model_better = match self.play_vs_model::<T>(nnet_best) {
            Some(1) => {
                self.pb.println("new model was better!").unwrap();
                true
            }
            Some(2) => {
                self.pb
                    .println("current best model is still better!")
                    .unwrap();
                false
            }
            None => {
                self.pb
                    .println("new model vs current best was inconclusive, keeping current best!")
                    .unwrap();
                false
            }
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

    pub fn play_vs_model<T: LearnableGame>(&self, other_model: ResNet) -> Option<usize> {
        let vs_self = VarStore::new(*DEVICE);
        let mut nnet_self = ResNet::new::<T>(vs_self);
        nnet_self.vs.copy(&self.model.vs)
            .unwrap_or_else(|e| panic!("unable to copy self into another ResNet, caught error: {}", e));
        let self_model_p = Player {
            player_type: Agent::Model,
            model: Some(nnet_self)
        };
        let other_model_p = Player {
            player_type: Agent::Model,
            model: Some(other_model)
        };
        let match_res = play::<T>(self_model_p, other_model_p, &self.mcts_config, self.config.temperature);
        println!("Match result: {}", match_res);
        if match_res.winrate >= 0.55 {
            Some(1)
        } else if match_res.winrate <= 0.45 {
            Some(2)
        } else {
            None
        }
    }
}
