"""
constants/DReplayMemory.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

from ai_snake_lab.utils.ConstGroup import ConstGroup

from ai_snake_lab.constants.DLabels import DLabel


class MEM_TYPE(ConstGroup):
    """Replay memory constants"""

    ADAPTIVE: str = "adaptive"
    ADAPTIVE_LABEL: str = "Adaptive"
    NONE: str = "none"
    NONE_LABEL: str = "None"
    RANDOM_GAME: str = "random_game"
    RANDOM_GAME_LABEL: str = "Random Game"
    SHUFFLE: str = "shuffle"
    SHUFFLE_LABEL: str = "Random Frames"

    MEM_TYPE_TABLE: dict = {
        NONE: NONE_LABEL,
        ADAPTIVE: ADAPTIVE_LABEL,
        RANDOM_GAME: RANDOM_GAME_LABEL,
        SHUFFLE: SHUFFLE_LABEL,
        DLabel.N_SLASH_A: DLabel.N_SLASH_A,
    }

    MEMORY_TYPES: list = [
        (NONE_LABEL, NONE),
        (ADAPTIVE_LABEL, ADAPTIVE),
        (RANDOM_GAME_LABEL, RANDOM_GAME),
        (SHUFFLE_LABEL, SHUFFLE),
    ]


class MEM(ConstGroup):
    """Replay memory constants"""

    MIN_GAMES: int = 200
    NO_DATA: int = -1
