# ai_hydra/constants/DReplayMemory.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Final


class DMemory:
    BATCH_SIZE: Final[int] = 64
    MIN_GAMES: Final[int] = 20
    MIN_FRAMES: Final[int] = 1000
    MAX_MEM_SIZE: Final[int] = 50000
    NO_DATA: Final[str] = "no_data"
    NO_MEMORY: Final[str] = "no_memory"
    RAN_FRAMES: Final[str] = "random_frames"
    RAN_GAMES: Final[str] = "random_games"
