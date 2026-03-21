# ai_hydra/constants/DReplayMemory.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Final


class DMemField:
    MAX_BUCKETS: Final[str] = "max_buckets"
    MAX_FRAMES: Final[str] = "max_frames"
    THRESHOLDS_REQUIRED: Final[str] = "thresholds_required"


class DMemDef:
    DOWNSHIFT_COUNT_THRESHOLD: Final[int] = 50
    MAX_FRAMES: Final[int] = 125000
    MAX_BUCKETS: Final[int] = 20
    NUM_COOLDOWN_EPISODES: Final[int] = 300
    THRESHOLD_BUCKETS: Final[list] = [17, 18, 19]
    THRESHOLDS_REQUIRED: Final[int] = 10
    UPSHIFT_COUNT_THRESHOLD: Final[int] = 150


class DMemory:
    # Simple Replay Memory - Linear model
    MIN_FRAMES: Final[int] = 1500
    MAX_MEM_SIZE: Final[int] = 75000


"""
Defines the threshold score for changing "gears". A gear change results
in a longer sequence length and smaller batch size.

GEAR => { ( THRESHOLD, SEQ_LENGTH, BATCH_SIZE ), ... }

- ATH Replay Memory starts in gear 0
  - seq_length is 4, batch_size is 128
- When the AI achieves a score of 8 ten times, it shifts up a gear
  - seq_length is now 8, batch_size i 64
"""

ATH_GEARBOX = {
    1: (4, 64),
    2: (6, 48),
    3: (8, 32),
    4: (12, 24),
    5: (16, 16),
    6: (20, 13),
    7: (24, 10),
    8: (28, 9),
    9: (30, 9),
    10: (32, 8),
    11: (34, 7),
    12: (36, 6),
    13: (38, 5),
    14: (40, 4),
}

V2_ATH_GEARBOX = {
    1: (4, 64),
    2: (8, 32),
    3: (16, 16),
    4: (24, 10),
    5: (28, 9),
    6: (32, 8),
    7: (40, 6),
    8: (48, 5),
    9: (56, 4),
    10: (64, 3),
}


V1_ATH_GEARBOX = {
    1: (4, 64),
    2: (8, 32),
    3: (16, 16),
    4: (24, 10),
    5: (32, 8),
    6: (40, 6),
    7: (48, 5),
    8: (56, 4),
    9: (64, 3),
}
