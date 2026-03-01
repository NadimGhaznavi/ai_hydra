# ai_hydra/constants/DNet.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Final

from ai_hydra.constants.DGame import DGameField


class DEpsilonDef:
    """
    Epsilon-greedy constants.
    """

    INITIAL: Final[float] = 0.99
    MINIMUM: Final[float] = 0.1
    DECAY_RATE: Final[float] = 0.95


class DNetDef:
    """
    Neural network constants.
    """

    # State map
    DANGER_FEATURES: Final[int] = 3
    DIRECTION_FEATURES: Final[int] = 4
    FOOD_FEATURES: Final[int] = 2
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
    DONE: Final[str] = DGameField.DONE
    CUR_EPSILON: Final[str] = "cur_epsilon"
    NEXT_STATE: Final[str] = "next_state"
    REWARD: Final[str] = DGameField.REWARD
    STATE: Final[str] = "state"


class DLinear:
    """
    Constants for the Linear neural network
    """

    HIDDEN_SIZE: Final[int] = 192
    DROPOUT_P: Final[float] = 0.0
    OUTPUT_SIZE: Final[int] = 3  # left / straight / right
