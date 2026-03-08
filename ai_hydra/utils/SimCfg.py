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

from ai_hydra.constants.DNNet import DNetField, DNetDef, DEpsilonDef


class SimCfg:
    _DEFAULTS: ClassVar[dict[str, Any]] = {
        DNetField.INITIAL_EPSILON: DEpsilonDef.INITIAL,
        DNetField.MOVE_DELAY: DNetDef.MOVE_DELAY,
        DNetField.PER_STEP: DNetDef.PER_STEP,
    }

    _COERCE: ClassVar[dict[str, Callable[[Any], Any]]] = {
        DNetField.INITIAL_EPSILON: float,
        DNetField.MOVE_DELAY: float,
        DNetField.PER_STEP: bool,
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
        self._values[key] = coerce(value) if coerce else value
