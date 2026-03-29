# ai_hydra/nnet/ATH/ATHCommon.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from __future__ import annotations

from dataclasses import dataclass, field

from ai_hydra.nnet.Transition import Transition


def assert_valid_gear(gear: int, max_gear: int) -> None:
    if not isinstance(gear, int):
        raise TypeError(f"gear must be int, got {type(gear).__name__}")
    if not isinstance(max_gear, int):
        raise TypeError(f"max_gear must be int, got {type(max_gear).__name__}")
    if max_gear < 1:
        raise ValueError(f"max_gear must be >= 1, got {max_gear}")
    if gear < 1 or gear > max_gear:
        raise ValueError(f"gear must be between 1 and {max_gear}, got {gear}")


@dataclass(slots=True, frozen=True)
class GearMeta:
    """
    Metadata for one episode under one gear.

    seq_length:
        Sequence length used for this gear.

    num_chunks:
        Number of full fixed-length chunks available in the episode.

    ranges:
        Slice ranges for each chunk as (start, end), where end is exclusive.
    """

    seq_length: int
    num_chunks: int
    ranges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.seq_length, int):
            raise TypeError(
                f"seq_length must be int, got {type(self.seq_length).__name__}"
            )
        if not isinstance(self.num_chunks, int):
            raise TypeError(
                f"num_chunks must be int, got {type(self.num_chunks).__name__}"
            )

        if self.seq_length < 1:
            raise ValueError(f"seq_length must be >= 1, got {self.seq_length}")
        if self.num_chunks < 0:
            raise ValueError(f"num_chunks must be >= 0, got {self.num_chunks}")

        if self.num_chunks != len(self.ranges):
            raise ValueError(
                "num_chunks must match len(ranges): "
                f"{self.num_chunks} != {len(self.ranges)}"
            )

        prev_end = -1
        for i, r in enumerate(self.ranges):
            if not isinstance(r, tuple) or len(r) != 2:
                raise TypeError(
                    f"ranges[{i}] must be a (start, end) tuple, got {r!r}"
                )

            start, end = r

            if not isinstance(start, int) or not isinstance(end, int):
                raise TypeError(f"ranges[{i}] values must be ints, got {r!r}")
            if start < 0:
                raise ValueError(
                    f"ranges[{i}].start must be >= 0, got {start}"
                )
            if end <= start:
                raise ValueError(
                    f"ranges[{i}] must satisfy end > start, got {r!r}"
                )

            # Require ascending, non-overlapping ranges.
            if start < prev_end:
                raise ValueError(
                    f"ranges must be sorted and non-overlapping; "
                    f"ranges[{i}]={r!r} overlaps prior end={prev_end}"
                )

            # ATH requires fixed-length full chunks only.
            if (end - start) != self.seq_length:
                raise ValueError(
                    f"ranges[{i}] length must equal seq_length "
                    f"({self.seq_length}), got {end - start}"
                )

            prev_end = end


@dataclass(slots=True, frozen=True)
class Episode:
    """
    One finalized episode and its per-gear metadata.
    """

    frames: tuple[Transition, ...]
    size: int
    gear_meta: dict[int, GearMeta] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.frames, tuple):
            raise TypeError(
                f"frames must be tuple, got {type(self.frames).__name__}"
            )
        if not isinstance(self.size, int):
            raise TypeError(
                f"size must be int, got {type(self.size).__name__}"
            )
        if self.size < 0:
            raise ValueError(f"size must be >= 0, got {self.size}")
        if self.size != len(self.frames):
            raise ValueError(
                f"size must equal len(frames): {self.size} != {len(self.frames)}"
            )

        if not isinstance(self.gear_meta, dict):
            raise TypeError(
                f"gear_meta must be dict, got {type(self.gear_meta).__name__}"
            )

        for gear, meta in self.gear_meta.items():
            if not isinstance(gear, int):
                raise TypeError(
                    f"gear_meta keys must be int, got {type(gear).__name__}"
                )
            if gear < 1:
                raise ValueError(f"gear_meta keys must be >= 1, got {gear}")
            if not isinstance(meta, GearMeta):
                raise TypeError(
                    f"gear_meta[{gear}] must be GearMeta, got "
                    f"{type(meta).__name__}"
                )


def build_chunk_ranges(
    ep_size: int, seq_length: int
) -> tuple[tuple[int, int], ...]:
    if not isinstance(ep_size, int):
        raise TypeError(f"ep_size must be int, got {type(ep_size).__name__}")
    if not isinstance(seq_length, int):
        raise TypeError(
            f"seq_length must be int, got {type(seq_length).__name__}"
        )

    if ep_size < 0:
        raise ValueError(f"ep_size must be >= 0, got {ep_size}")
    if seq_length < 1:
        raise ValueError(f"seq_length must be >= 1, got {seq_length}")

    num_chunks = ep_size // seq_length
    rem = ep_size % seq_length

    ranges: list[tuple[int, int]] = []
    for chunk_idx in range(num_chunks):
        start = rem + (chunk_idx * seq_length)
        end = start + seq_length
        ranges.append((start, end))

    return tuple(ranges)


def get_gear_data(
    gear: int,
    max_gear: int,
    max_training_frames: int,
) -> tuple[int, int]:
    """
    Return (sequence_length, batch_size) for a 1-based gear.

    Rules:
    - valid gears are 1..max_gear
    - gear 1 => seq_length 4
    - each higher gear adds 2 to sequence_length
    - batch_size is floor(max_training_frames / seq_length), minimum 1
    """
    assert_valid_gear(gear=gear, max_gear=max_gear)

    if not isinstance(max_training_frames, int):
        raise TypeError(
            "max_training_frames must be int, got "
            f"{type(max_training_frames).__name__}"
        )
    if max_training_frames < 4:
        raise ValueError(
            f"max_training_frames must be >= 4, got {max_training_frames}"
        )

    seq_len = (gear * 2) + 2
    batch_size = max(1, max_training_frames // seq_len)
    return seq_len, batch_size


def get_valid_gears(max_gear: int) -> range:
    if not isinstance(max_gear, int):
        raise TypeError(f"max_gear must be int, got {type(max_gear).__name__}")
    if max_gear < 1:
        raise ValueError(f"max_gear must be >= 1, got {max_gear}")
    return range(1, max_gear + 1)
