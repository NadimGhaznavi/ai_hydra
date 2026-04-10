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


class DEpsilonField:
    """
    Epsilon-greedy fields.
    """

    INITIAL: Final[str] = "initial"
    MINIMUM: Final[str] = "minimum"
    DECAY_RATE: Final[str] = "decay"


class DNetDef:
    """
    Neural network constants.
    """

    MOVE_DELAY: Final[float] = 0.02
    PER_STEP: Final[bool] = True
    INPUT_SIZE: Final[int] = 51
    OUTPUT_SIZE: Final[int] = 3


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
    MCTS_SCORE_THRESH: Final[str] = "mcts_score_thresh"
    MCTS_STEPS: Final[str] = "mcts_steps"
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
