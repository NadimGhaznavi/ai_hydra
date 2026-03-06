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


class DEpsilonDef:
    """
    Epsilon-greedy defaults.
    """

    INITIAL: Final[float] = 0.99
    MINIMUM: Final[float] = 0.0
    DECAY_RATE: Final[float] = 0.98


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

    HIDDEN_SIZE: Final[int] = 192
    DROPOUT_P: Final[float] = 0.1
    LEARNING_RATE: Final[float] = 0.00005
    OUTPUT_SIZE: Final[int] = 3  # left / straight / right


class DLookahead:
    """
    Lookahead constants.
    """

    ON: Final[str] = "🟢"
    OFF: Final[str] = "🔴"
    UNKNOWN: Final[str] = "⚪"


class DLookaheadDef:
    """
    Lookahead defaults.
    """

    PROBABILITY: Final[float] = 0.25


class DNetDef:
    """
    Neural network constants.
    """

    # State map
    DANGER_FEATURES: Final[int] = 3
    DIRECTION_FEATURES: Final[int] = 4
    FOOD_FEATURES: Final[int] = 2
    GAMMA: Final[float] = 0.9
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
    FINAL_SCORE: Final[str] = "final_score"
    LOOKAHEAD_ON: Final[str] = "lookahead_on"
    LOSS: Final[str] = "loss"
    MOVE_DELAY: Final[str] = "move_delay"
    NEXT_STATE: Final[str] = "next_state"
    PER_STEP: Final[str] = "per_step"
    REWARD: Final[str] = DGameField.REWARD
    STATE: Final[str] = "state"
