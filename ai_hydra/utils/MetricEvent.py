# ai_hydra/utils/MetricEvent.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class HighscoreEvent:
    epoch: int
    gear: int
    highscore: int
    epsilon: float | None
    elapsed_time: str | None


@dataclass(slots=True, frozen=True)
class LossEvent:
    epoch: int
    loss: float


@dataclass(slots=True, frozen=True)
class MemEvent:
    epoch: int
    gear: int
    bucket_counts: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class NiceEvent:
    window: str
    epoch: int
    calls: int
    triggered: int
    fatal_suggested: int
    overrides: int
    no_safe_alternative: int
    trigger_rate: float
    override_rate: float


@dataclass(slots=True, frozen=True)
class ScoreEvent:
    epoch: int
    score: int


@dataclass(slots=True, frozen=True)
class ShiftEvent:
    epoch: int
    gear: int
    seq_length: int
    batch_size: int
