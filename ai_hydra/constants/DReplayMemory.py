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
    # Used by the ATHGearbox to calculate the seq_length/batch_size for the
    # different gears.
    MAX_TRAINING_FRAMES: Final[int] = 512
    # The highest gear
    MAX_GEAR: Final[int] = 26
    # ATH memory stores this maximum number of frames
    MAX_FRAMES: Final[int] = 150000
    # ATH uses this many memory buckets
    MAX_BUCKETS: Final[int] = 20
    # The ATHGearbox won't shift up or down before NUM_COOLDOWN_EPISODES
    # episodes have gone by.
    NUM_COOLDOWN_EPISODES: Final[int] = 100
    # If the last three buckets have a combined contents greater than
    # UPSHIFT_COUNT_THRESHOLD, the the ATHGearbox shifts UP.
    # If the last three buckets have a combined contents less than
    # DOWNSHIFT_COUNT_THRESHOLD, the the ATHGearbox shifts DOWN.
    UPSHIFT_COUNT_THRESHOLD: Final[int] = 150
    DOWNSHIFT_COUNT_THRESHOLD: Final[int] = 50
    # The number of episodes that the TrainMgr will tolerate with no new
    # highscore before calling `stagnation_warning()` on the ATHReplay.
    MAX_STAGNANT_EPISODES: Final[int] = 300
    # The number of episodes without a new highscore before the TrainMgr
    # calls `hard_reset()`.
    MAX_HARD_RESET_EPISODES: Final[int] = 2000


class DMemory:
    # Simple Replay Memory - Linear model
    MIN_FRAMES: Final[int] = 1500
    MAX_MEM_SIZE: Final[int] = 75000
