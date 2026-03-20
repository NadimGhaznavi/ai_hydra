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
