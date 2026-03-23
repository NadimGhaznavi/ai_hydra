# ai_hydra/nnet/ATH/ATHCommon.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


from dataclasses import dataclass, field

from ai_hydra.nnet.Transition import Transition

from ai_hydra.constants.DReplayMemory import DMemDef

MAX_TRAINING_FRAMES = DMemDef.MAX_TRAINING_FRAMES
MAX_GEAR = DMemDef.MAX_GEAR


@dataclass(slots=True, frozen=True)
class GearMeta:
    """
    This stores, for one episode and under one gear:

    - Ssequence length
    - Number of chunks
    - Slice ranges for each bucket/chunk
    """

    seq_length: int
    num_chunks: int
    ranges: tuple[tuple[int, int], ...]


@dataclass(slots=True, frozen=True)
class Episode:
    """
    This stores the frames for one game and additional metadata.
    """

    frames: tuple[Transition, ...]
    size: int
    gear_meta: dict[int, GearMeta] = field(default_factory=dict)


def get_gear_data(gear: int) -> tuple[int, int]:
    """
    - Accepts a gear.
    - Returns (sequence_length, batch_size)

    The sequence_length/batch_size should not exceed MAX_TRAINING_FRAMES.
    First gear (1), should have a sequence_length of 4.
    Every gear after one should increase the seq_length by 2.
    """

    if gear < 1 or gear > MAX_GEAR:
        raise ValueError(f"gear must be between 1 and {MAX_GEAR}")

    seq_len = (gear * 2) + 2
    batch_size = max(1, MAX_TRAINING_FRAMES // seq_len)
    return seq_len, batch_size


def get_valid_gears() -> range:
    return range(1, MAX_GEAR + 1)
