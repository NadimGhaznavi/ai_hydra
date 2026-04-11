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
    CLOSER_TO_FOOD: Final[float] = 0.1
    DROPOUT_P_VALUE: Final[float] = 0.1
    DOWNSHIFT_COUNT_THRESHOLD: Final[int] = 50
    INITIAL_EPSILON: Final[float] = 0.99
    EMPTY_MOVE_REWARD: Final[float] = 0.0
    EPSILON_DECAY_RATE: Final[float] = 0.97
    FURTHER_FROM_FOOD: Final[float] = -0.1
    GAMMA: Final[float] = 0.95
    HIDDEN_SIZE: Final[int] = 192
    LEARNING_RATE: Final[float] = 0.0004
    MAX_BUCKETS: Final[int] = 20
    MAX_FRAMES: Final[int] = 125000
    MAX_GEAR: Final[int] = 26
    MAX_HARD_RESET_EPISODES: Final[int] = 500
    MAX_MOVES_MULTIPLIER: Final[int] = 100
    MAX_MOVES_PENALTY: Final[float] = -10
    MAX_STAGNANT_EPISODES: Final[int] = 300
    MAX_TRAINING_FRAMES: Final[int] = 512
    MINIMUM_EPSILON: Final[float] = 0.0
    NICE_P_VALUE: Final[float] = 0.005
    NICE_STEPS: Final[int] = 20
    NUM_COOLDOWN_EPISODES: Final[int] = 100
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
    MINIMUM_EPSILON: Final[float] = 0.0
    NICE_P_VALUE: Final[float] = 0.005
    NICE_STEPS: Final[int] = 20
    NUM_COOLDOWN_EPISODES: Final[int] = 200
    RNN_LAYERS: Final[int] = 3
    SEQ_LENGTH: Final[int] = 4
    TAU: Final[float] = 0.001
    UPSHIFT_COUNT_THRESHOLD: Final[int] = 300


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
    GAMMA: Final[float] = 0.95
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
    MINIMUM_EPSILON: Final[float] = 0.0
    NICE_P_VALUE: Final[float] = 0.005
    NICE_STEPS: Final[int] = 20
    NUM_COOLDOWN_EPISODES: Final[int] = 200
    SEQ_LENGTH: Final[int] = 4
    TAU: Final[float] = 0.001
    UPSHIFT_COUNT_THRESHOLD: Final[int] = 300


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
