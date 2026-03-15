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
    MAX_CHUNKS: Final[int] = 20000  # Used by RNN II
    MIN_CHUNKS: Final[int] = 100  # Used by RNN II
    MIN_FRAMES: Final[int] = 1500  # Used by Linear
    MAX_MEM_SIZE: Final[int] = 50000  # Used by Linear
