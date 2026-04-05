# ai_hydra/constants/DNNet.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Final

from ai_hydra.constants.DGame import DGameField
from ai_hydra.constants.DHydraTui import DLabel, DField
import torch.nn as nn
import torch.optim as optim


class DGRU:
    """
    GRU Model defaults
    """

    BATCH_SIZE: Final[int] = 64
    CLOSER_TO_FOOD: Final[float] = 0.0
    DROPOUT_P_VALUE: Final[float] = 0.1
    DOWNSHIFT_COUNT_THRESHOLD: Final[int] = 50
    INITIAL_EPSILON: Final[float] = 0.999
    EMPTY_MOVE_REWARD: Final[float] = 0.0
    EPSILON_DECAY_RATE: Final[float] = 0.98
    FURTHER_FROM_FOOD: Final[float] = 0.0
    GAMMA: Final[float] = 0.97
    HIDDEN_SIZE: Final[int] = 224
    LEARNING_RATE: Final[float] = 0.0005
    MAX_BUCKETS: Final[int] = 20
    MAX_FRAMES: Final[int] = 125000
    MAX_GEAR: Final[int] = 26
    MAX_HARD_RESET_EPISODES: Final[int] = 500
    MAX_MOVES_MULTIPLIER: Final[int] = 150
    MAX_MOVES_PENALTY: Final[float] = 0.0
    MAX_STAGNANT_EPISODES: Final[int] = 300
    MAX_TRAINING_FRAMES: Final[int] = 512
    MCTS_DEPTH: Final[int] = 5
    MCTS_EXPLORE_P_VALUE: Final[float] = 0.9
    MCTS_GATE_P_VALUE: Final[float] = 0.01
    MCTS_ITER: Final[int] = 100
    MINIMUM_EPSILON: Final[float] = 0.0
    NICE_P_VALUE: Final[float] = 0.005
    NICE_STEPS: Final[int] = 20
    NUM_COOLDOWN_EPISODES: Final[int] = 100
    OUTPUT_SIZE: Final[int] = 3
    GRU_LAYERS: Final[int] = 3
    SEQ_LENGTH: Final[int] = 4
    TAU: Final[float] = 0.001
    UPSHIFT_COUNT_THRESHOLD: Final[int] = 150


class DRNN:
    """
    RNN Model defaults
    """

    BATCH_SIZE: Final[int] = 64
    CLOSER_TO_FOOD: Final[float] = 0.1
    DROPOUT_P_VALUE: Final[float] = 0.1
    DOWNSHIFT_COUNT_THRESHOLD: Final[int] = 50
    INITIAL_EPSILON: Final[float] = 0.96
    EMPTY_MOVE_REWARD: Final[float] = 0.0
    EPSILON_DECAY_RATE: Final[float] = 0.97
    FURTHER_FROM_FOOD: Final[float] = -0.1
    GAMMA: Final[float] = 0.96
    HIDDEN_SIZE: Final[int] = 192
    LEARNING_RATE: Final[float] = 0.002
    MAX_BUCKETS: Final[int] = 20
    MAX_FRAMES: Final[int] = 150000
    MAX_GEAR: Final[int] = 26
    MAX_HARD_RESET_EPISODES: Final[int] = 500
    MAX_MOVES_MULTIPLIER: Final[int] = 100
    MAX_MOVES_PENALTY: Final[float] = -10.0
    MAX_STAGNANT_EPISODES: Final[int] = 300
    MAX_TRAINING_FRAMES: Final[int] = 512
    MCTS_DEPTH: Final[int] = 10
    MCTS_EXPLORE_P_VALUE: Final[float] = 0.9
    MCTS_GATE_P_VALUE: Final[float] = 0.01
    MCTS_ITER: Final[int] = 100
    MINIMUM_EPSILON: Final[float] = 0.0
    NICE_P_VALUE: Final[float] = 0.005
    NICE_STEPS: Final[int] = 20
    NUM_COOLDOWN_EPISODES: Final[int] = 200
    OUTPUT_SIZE: Final[int] = 3
    RNN_LAYERS: Final[int] = 3
    SEQ_LENGTH: Final[int] = 4
    TAU: Final[float] = 0.001
    UPSHIFT_COUNT_THRESHOLD: Final[int] = 300


class DEpsilonField:
    """
    Epsilon-greedy fields.
    """

    INITIAL: Final[str] = "initial"
    MINIMUM: Final[str] = "minimum"
    DECAY_RATE: Final[str] = "decay"


class DLinear:
    """
    Constants for the Linear neural network
    """

    BATCH_SIZE: Final[int] = 64
    CLOSER_TO_FOOD: Final[float] = 0.1
    DOWNSHIFT_COUNT_THRESHOLD: Final[int] = 50
    DROPOUT_P: Final[float] = 0.1
    EMPTY_MOVE_REWARD: Final[float] = 0.0
    EPSILON_DECAY_RATE: Final[float] = 0.985
    FURTHER_FROM_FOOD: Final[float] = -0.1
    GAMMA: Final[float] = 0.9
    HIDDEN_SIZE: Final[int] = 192
    INITIAL_EPSILON: Final[float] = 0.99
    LEARNING_RATE: Final[float] = 0.00005
    LINEAR_LAYERS: Final[int] = 2
    MAX_BUCKETS: Final[int] = 20
    MAX_FRAMES: Final[int] = 125000
    MAX_GEAR: Final[int] = 26
    MAX_HARD_RESET_EPISODES: Final[int] = 500
    MAX_MOVES_MULTIPLIER: Final[int] = 100
    MAX_MOVES_PENALTY: Final[float] = -10.0
    MAX_STAGNANT_EPISODES: Final[int] = 300
    MAX_TRAINING_FRAMES: Final[int] = 512
    MCTS_DEPTH: Final[int] = 10
    MCTS_EXPLORE_P_VALUE: Final[float] = 0.9
    MCTS_GATE_P_VALUE: Final[float] = 0.01
    MCTS_ITER: Final[int] = 100
    MINIMUM_EPSILON: Final[float] = 0.0
    NICE_P_VALUE: Final[float] = 0.005
    NICE_STEPS: Final[int] = 20
    NUM_COOLDOWN_EPISODES: Final[int] = 200
    OUTPUT_SIZE: Final[int] = 3  # left / straight / right
    SEQ_LENGTH: Final[int] = 4
    TAU: Final[float] = 0.001
    UPSHIFT_COUNT_THRESHOLD: Final[int] = 300


class DNetDef:
    """
    Neural network constants.
    """

    # State map and hyperparameters
    SNAKE_DANGER_FEATURES: Final[int] = 3
    WALL_DANGER_FEATURES: Final[int] = 3
    DIRECTION_FEATURES: Final[int] = 4
    FOOD_FEATURES: Final[int] = 6
    MOVE_DELAY: Final[float] = 0.02
    PER_STEP: Final[bool] = True
    STATE_LENGTH_BITS: Final[int] = 7

    # Derived
    INPUT_SIZE: Final[int] = (
        SNAKE_DANGER_FEATURES
        + WALL_DANGER_FEATURES
        + DIRECTION_FEATURES
        + FOOD_FEATURES
        + STATE_LENGTH_BITS
    )


class DNetField:
    """
    Field names used in the NN
    """

    ACTION: Final[str] = DGameField.ACTION
    BATCH_SIZE: Final[str] = "batch_size"
    CLOSER_TO_FOOD: Final[str] = "closer_to_food"
    COLLISION_PENALTY: Final[str] = "collision_penalty"
    CUR_EPSILON: Final[str] = "cur_epsilon"
    DONE: Final[str] = DGameField.DONE
    DOWNSHIFT_COUNT_THRESHOLD: Final[str] = "downshift_count_threshold"
    DROPOUT_P: Final[str] = "dropout_p"
    EMPTY_MOVE_REWARD: Final[str] = "empty_move_reward"
    EPSILON_DECAY: Final[str] = "epsilon_decay"
    FINAL_SCORE: Final[str] = "final_score"
    FOOD_REWARD: Final[str] = "food_reward"
    FURTHER_FROM_FOOD: Final[str] = "further_from_food"
    GAMMA: Final[str] = "gamma"
    HIDDEN_SIZE: Final[str] = "hidden_size"
    INITIAL_EPSILON: Final[str] = "initial_epsilon"
    LEARNING_RATE: Final[str] = "learning_rate"
    LOSS: Final[str] = "loss"
    MAX_BUCKETS: Final[str] = "max_buckets"
    MAX_FRAMES: Final[str] = "max_frames"
    MAX_GEAR: Final[str] = "max_gear"
    MAX_HARD_RESET_EPISODES: Final[str] = "max_hard_reset_episodes"
    MAX_MOVES_MULTIPLIER: Final[str] = "max_moves_multiplier"
    MAX_MOVES_PENALTY: Final[str] = "max_moves_penalty"
    MAX_STAGNANT_EPISODES: Final[str] = "max_stagnant_episodes"
    MAX_TRAINING_FRAMES: Final[str] = "max_training_frames"
    MCTS_DEPTH: Final[str] = "mcts_depth"
    MCTS_ITER: Final[str] = "mcts_iter"
    MCTS_EXPLORE_P_VALUE: Final[str] = "mcts_explore_p_value"
    MCTS_GATE_P_VALUE: Final[str] = "mcts_gate_p_value"
    MIN_EPSILON: Final[str] = "min_epsilon"
    MODEL_TYPE: Final[str] = "model_type"
    MOVE_DELAY: Final[str] = "move_delay"
    NEXT_STATE: Final[str] = "next_state"
    NICE_P_VALUE: Final[str] = "nice_p_value"
    NICE_STEPS: Final[str] = "nice_steps"
    NUM_COOLDOWN_EPISODES: Final[str] = "num_cooledown_episodes"
    PER_STEP: Final[str] = "per_step"
    RANDOM_SEED: Final[str] = "random_seed"
    REWARD: Final[str] = DGameField.REWARD
    LAYERS: Final[str] = "layers"
    TAU: Final[str] = "tau"
    SEQ_LENGTH: Final[str] = "sequence_length"
    SIM_PAUSED: Final[str] = "sim_paused"
    STATE: Final[str] = "state"
    UPSHIFT_COUNT_THRESHOLD: Final[str] = "upshift_count_threshold"


class DRecurrentTrainer:
    """
    RNN Trainer defaults
    """

    CRITERION = nn.SmoothL1Loss
    OPTIM = optim.Adam
    TAU: Final[float] = 0.005
    UPDATE_FREQ: Final[int] = 100


MODEL_TYPE_TABLE: Final[dict] = {
    DField.LINEAR: DLabel.LINEAR,
    DField.RNN: DLabel.RNN,
    DField.GRU: DLabel.GRU,
}

MODEL_TYPES: Final[list] = [
    (DLabel.LINEAR, DField.LINEAR),
    (DLabel.RNN, DField.RNN),
    (DLabel.GRU, DField.GRU),
]
