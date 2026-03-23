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
    DROPOUT_P_VALUE: Final[float] = 0.0
    INITIAL_EPSILON: Final[float] = 0.999
    EPSILON_DECAY_RATE: Final[float] = 0.98
    GAMMA: Final[float] = 0.96
    HIDDEN_SIZE: Final[int] = 96
    LEARNING_RATE: Final[float] = 0.002
    MINIMUM_EPSILON: Final[float] = 0.0
    NICE_P_VALUE: Final[float] = 0.0
    NICE_STEPS: Final[int] = 10
    OUTPUT_SIZE: Final[int] = 3
    GRU_LAYERS: Final[int] = 3
    SEQ_LENGTH: Final[int] = 4
    TAU: Final[float] = 0.001


class DRNN:
    """
    RNN Model defaults
    """

    BATCH_SIZE: Final[int] = 64
    DROPOUT_P_VALUE: Final[float] = 0.1
    INITIAL_EPSILON: Final[float] = 0.999
    EPSILON_DECAY_RATE: Final[float] = 0.995
    GAMMA: Final[float] = 0.96
    HIDDEN_SIZE: Final[int] = 320
    LEARNING_RATE: Final[float] = 0.002
    MINIMUM_EPSILON: Final[float] = 0.0
    NICE_P_VALUE: Final[float] = 0.025
    NICE_STEPS: Final[int] = 10
    OUTPUT_SIZE: Final[int] = 3
    RNN_LAYERS: Final[int] = 5
    SEQ_LENGTH: Final[int] = 4
    TAU: Final[float] = 0.001


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
    GAMMA: Final[float] = 0.9
    HIDDEN_SIZE: Final[int] = 192
    DROPOUT_P: Final[float] = 0.1
    LEARNING_RATE: Final[float] = 0.00005
    OUTPUT_SIZE: Final[int] = 3  # left / straight / right
    INITIAL_EPSILON: Final[float] = 0.99
    MINIMUM_EPSILON: Final[float] = 0.0
    NICE_P_VALUE: Final[float] = 0.005
    NICE_STEPS: Final[int] = 10
    EPSILON_DECAY_RATE: Final[float] = 0.98


class DNetDef:
    """
    Neural network constants.
    """

    # State map and hyperparameters
    DANGER_FEATURES: Final[int] = 3
    DIRECTION_FEATURES: Final[int] = 4
    FOOD_FEATURES: Final[int] = 2
    MOVE_DELAY: Final[float] = 0.02
    PER_STEP: Final[bool] = True
    STATE_LENGTH_BITS: Final[int] = 7

    # Derived
    INPUT_SIZE: Final[int] = (
        DANGER_FEATURES
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
    CUR_EPSILON: Final[str] = "cur_epsilon"
    DONE: Final[str] = DGameField.DONE
    DROPOUT_P: Final[str] = "dropout_p"
    EPSILON_DECAY: Final[str] = "epsilon_decay"
    FINAL_SCORE: Final[str] = "final_score"
    GAMMA: Final[str] = "gamma"
    HIDDEN_SIZE: Final[str] = "hidden_size"
    INITIAL_EPSILON: Final[str] = "initial_epsilon"
    LEARNING_RATE: Final[str] = "learning_rate"
    LOSS: Final[str] = "loss"
    MIN_EPSILON: Final[str] = "min_epsilon"
    MODEL_TYPE: Final[str] = "model_type"
    MOVE_DELAY: Final[str] = "move_delay"
    NEXT_STATE: Final[str] = "next_state"
    PER_STEP: Final[str] = "per_step"
    RANDOM_SEED: Final[str] = "random_seed"
    REWARD: Final[str] = DGameField.REWARD
    LAYERS: Final[str] = "layers"
    TAU: Final[str] = "tau"
    SEQ_LENGTH: Final[str] = "sequence_length"
    SIM_PAUSED: Final[str] = "sim_paused"
    STATE: Final[str] = "state"


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
