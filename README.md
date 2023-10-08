# die-e - AlphaZero for Backgammon

**die-e** is a Rust-based project that primarily focuses on implementing the AlphaZero algorithm for the classic game of Backgammon. The project's name, "die-e," is a playful reference to the Turkish word "dayi," which often refers to the best players in the family. Just like a skilled uncle in a family, this project aims to master Backgammon through AI-powered learning.

## Overview

The core functionality of **die-e** is to apply the AlphaZero algorithm to learn and play Backgammon. Additionally, it offers the capability to play Tic-Tac-Toe using the same AI model.

## CLI Arguments

Here are the main CLI arguments and commands that **die-e** supports:

### Arguments:

1. `--config` (or `-c`): Specify a filepath to the configuration file that defines various parameters used in the AlphaZero learning process and MCTS (Monte Carlo Tree Search). Below is an example configuration:

    ```ini
    # AlphaZero parameters
    temperature = 1.25
    learn_iterations = 100
    num_epochs = 4
    training_batch_size = 256
    self_play_iterations = 4
    num_self_play_batches = 1024
    
    # MCTS parameters
    iterations = 100
    exploration_const = 2
    simulate_round_limit = 400
    dirichlet_alpha = 0.3
    dirichlet_epsilon = 0.25
    
    # Optimizer parameters
    wd = 0.0001
    lr = 0.001
    ```

2. `--game` (or `-g`): Indicates whether the game to be played or learned is Backgammon or Tic-Tac-Toe.

3. `--n-cpus` (or `-n`): Specifies the number of CPU cores to utilize for learning. By default, it uses half of the total available CPU cores.

### Commands:

**die-e** CLI provides several commands to control various aspects of the project:

#### 1. Learn:

- `Learn`: Starts the learning process.

    - `--model_path`: Path to the model.

#### 2. Play:

- `Play`: Allows you to play a game with different agents.

    - `--agent_one`: Type of Agent One (can be 'random,' 'mcts,' 'model').
    
    - `--model_path_one`: Path to the model for Agent One (if applicable).
    
    - `--agent_two`: Type of Agent Two (can be 'random,' 'mcts,' 'model').
    
    - `--model_path_two`: Path to the model for Agent Two (if applicable).
    
    - `--output_path`: Path to save the output game.

#### 3. Train:

- `Train`: Initiates the training process.

    - `--model_path`: Path to the model to train.
    
    - `--out_path`: Path to save the trained model.
    
    - `--run_id`: The run ID for the data (uses all data under this ID for training).
    
    - `--learn`: The index of the learn iteration (requires `run_id`).
    
    - `--self_play`: The index of the self-play iteration (requires `learn` and `run_id`).

#### 4. Replay:

- `Replay`: Allows you to replay a saved game.

    - `--game_path`: Path of the game to load.

## Example Usages:

1. To start the learning process for Backgammon with a custom model path:

    ```shell
    die-e learn --game backgammon --model_path my_custom_model.pth
    ```

2. To play a game of Backgammon between a random agent and a model agent and save the game:

    ```shell
    die-e play --game backgammon --agent_one random --agent_two model --model_path_two my_model.pth --output_path game_output.json
    ```

3. To initiate the training process for Backgammon using a specific model and save the trained model:

    ```shell
    die-e train --game backgammon --model_path my_model.pth --out_path trained_model.pth --run_id my_run --learn 1 --self_play 2
    ```

4. To replay a saved game:

    ```shell
    die-e replay --game_path saved_game.json
    ```

**die-e** primarily focuses on mastering the game of Backgammon using the AlphaZero algorithm, with Tic-Tac-Toe available as a secondary feature for some additional fun.