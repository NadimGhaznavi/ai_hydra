# ai_hydra/utils/SimCfg.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from __future__ import annotations

from typing import Any, Dict, Callable, ClassVar

from ai_hydra.constants.DSimCfg import Phase
from ai_hydra.constants.DNNet import DNetField, DNetDef, DEpsilonDef

_UNSET = object()


class SimCfg:
    # key -> phase (class-level: shared policy)
    _PHASE: ClassVar[Dict[str, Phase]] = {
        DNetField.INITIAL_EPSILON: Phase.PRE_START,
        DNetField.MOVE_DELAY: Phase.RUNTIME,
        DNetField.PER_STEP: Phase.RUNTIME,
    }

    # optional: key -> coercer (class-level: shared policy)
    _COERCE: ClassVar[Dict[str, Callable[[Any], Any]]] = {
        DNetField.INITIAL_EPSILON: float,
        DNetField.MOVE_DELAY: float,
        DNetField.PER_STEP: bool,
    }

    __slots__ = ("_values",)

    def __init__(self) -> None:
        # Only instance state
        self._values: Dict[str, Any] = {}

    @classmethod
    def init_client(cls) -> "SimCfg":
        cfg = cls()
        cfg._values = {
            DNetField.INITIAL_EPSILON: DEpsilonDef.INITIAL,
            DNetField.MOVE_DELAY: DNetDef.MOVE_DELAY,
            DNetField.PER_STEP: DNetDef.PER_STEP,
        }
        return cfg

    @classmethod
    def init_server(cls) -> "SimCfg":
        cfg = cls()
        cfg._values = {
            DNetField.INITIAL_EPSILON: _UNSET,
            DNetField.MOVE_DELAY: _UNSET,
            DNetField.PER_STEP: _UNSET,
        }
        return cfg

    def apply(self, payload: Dict[str, Any], *, phase: Phase) -> None:
        for k, v in payload.items():
            self.set(k, v, phase=phase)

    def get(self, key: str) -> Any:
        if key not in self._PHASE:
            raise KeyError(f"Unknown cfg key: {key}")
        v = self._values.get(key, _UNSET)
        if v is _UNSET:
            raise AssertionError(f"Cfg key is UNSET: {key}")
        return v

    def set(self, key: str, value: Any, *, phase: Phase) -> None:
        if key not in self._PHASE:
            raise KeyError(f"Unknown cfg key: {key}")
        if self._PHASE[key] == Phase.PRE_START and phase == Phase.RUNTIME:
            raise ValueError(f"{key} is pre_start only")

        coerce = self._COERCE.get(key)
        self._values[key] = coerce(value) if coerce else value

    def to_runtime_payload(self) -> Dict[str, Any]:
        return {
            k: self.get(k)
            for k, p in self._PHASE.items()
            if p == Phase.RUNTIME
        }

    def to_start_payload(self) -> Dict[str, Any]:
        return {k: self.get(k) for k in self._PHASE.keys()}
