# ai_hydra/utils/SimCfg.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from __future__ import annotations

from typing import Any, Callable, ClassVar
import traceback

from ai_hydra.constants.DNNet import (
    DNetField,
    DNetDef,
    DLinear,
    DRNN,
    DGRU,
)
from ai_hydra.constants.DHydraTui import DField
from ai_hydra.constants.DHydra import DHydra


class SimCfg:
    _DEFAULTS: ClassVar[dict[str, Any]] = {
        DNetField.BATCH_SIZE: DLinear.BATCH_SIZE,
        DNetField.DOWNSHIFT_COUNT_THRESHOLD: DGRU.DOWNSHIFT_COUNT_THRESHOLD,
        DNetField.DROPOUT_P: DLinear.DROPOUT_P,
        DNetField.EPSILON_DECAY: DLinear.EPSILON_DECAY_RATE,
        DNetField.GAMMA: DLinear.GAMMA,
        DNetField.HIDDEN_SIZE: DLinear.HIDDEN_SIZE,
        DNetField.INITIAL_EPSILON: DLinear.INITIAL_EPSILON,
        DNetField.LEARNING_RATE: DLinear.LEARNING_RATE,
        DNetField.MAX_BUCKETS: DGRU.MAX_BUCKETS,
        DNetField.MAX_FRAMES: DGRU.MAX_FRAMES,
        DNetField.MAX_GEAR: DGRU.MAX_GEAR,
        DNetField.MAX_HARD_RESET_EPISODES: DGRU.MAX_HARD_RESET_EPISODES,
        DNetField.MAX_STAGNANT_EPISODES: DGRU.MAX_STAGNANT_EPISODES,
        DNetField.MAX_TRAINING_FRAMES: DGRU.MAX_TRAINING_FRAMES,
        DNetField.MIN_EPSILON: DLinear.MINIMUM_EPSILON,
        DNetField.MODEL_TYPE: DField.LINEAR,
        DNetField.MOVE_DELAY: DNetDef.MOVE_DELAY,
        DNetField.NUM_COOLDOWN_EPISODES: DNetField.NUM_COOLDOWN_EPISODES,
        DNetField.PER_STEP: DNetDef.PER_STEP,
        DNetField.RANDOM_SEED: DHydra.RANDOM_SEED,
        DNetField.LAYERS: DRNN.RNN_LAYERS,
        DNetField.TAU: DRNN.TAU,
        DNetField.SEQ_LENGTH: DRNN.SEQ_LENGTH,
        DNetField.SIM_PAUSED: False,
        DNetField.UPSHIFT_COUNT_THRESHOLD: DGRU.UPSHIFT_COUNT_THRESHOLD,
    }

    _COERCE: ClassVar[dict[str, Callable[[Any], Any]]] = {
        DNetField.BATCH_SIZE: int,
        DNetField.DOWNSHIFT_COUNT_THRESHOLD: int,
        DNetField.DROPOUT_P: float,
        DNetField.EPSILON_DECAY: float,
        DNetField.GAMMA: float,
        DNetField.HIDDEN_SIZE: int,
        DNetField.INITIAL_EPSILON: float,
        DNetField.LEARNING_RATE: float,
        DNetField.MAX_BUCKETS: int,
        DNetField.MAX_FRAMES: int,
        DNetField.MAX_GEAR: int,
        DNetField.MAX_HARD_RESET_EPISODES: int,
        DNetField.MAX_STAGNANT_EPISODES: int,
        DNetField.MAX_TRAINING_FRAMES: int,
        DNetField.MIN_EPSILON: float,
        DNetField.MODEL_TYPE: str,
        DNetField.MOVE_DELAY: float,
        DNetField.NUM_COOLDOWN_EPISODES: int,
        DNetField.PER_STEP: bool,
        DNetField.RANDOM_SEED: int,
        DNetField.LAYERS: int,
        DNetField.TAU: float,
        DNetField.SEQ_LENGTH: int,
        DNetField.SIM_PAUSED: bool,
        DNetField.UPSHIFT_COUNT_THRESHOLD: int,
    }

    __slots__ = ("_values",)

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self._values: dict[str, Any] = dict(self._DEFAULTS)
        if values:
            self.apply(values)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SimCfg":
        return cls(payload)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def apply(self, payload: dict[str, Any]) -> None:
        for key, value in payload.items():
            self.set(key, value)

    def get(self, key: str) -> Any:
        if key not in self._DEFAULTS:
            raise KeyError(f"Unknown cfg key: {key}")
        return self._values[key]

    def set(self, key: str, value: Any) -> None:
        if key not in self._DEFAULTS:
            raise KeyError(f"Unknown cfg key: {key}")

        coerce = self._COERCE.get(key)
        try:
            self._values[key] = coerce(value) if coerce else value
        except Exception as e:
            print(f"ERROR: {e}")
            print(f"STACKTRACE: {traceback.format_exc()}")
