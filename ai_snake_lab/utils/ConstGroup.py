"""
ai_snake_lab/utils/ConstGroup.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""


class MetaConst(type):
    """Metaclass that collects public attributes into a dictionary-like mapping."""

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        cls._constants = {
            k: v
            for k, v in namespace.items()
            if not k.startswith("_") and not callable(v)
        }
        return cls

    def __getitem__(cls, key):
        return cls._constants[key]

    def keys(cls):
        return cls._constants.keys()

    def values(cls):
        return cls._constants.values()

    def items(cls):
        return cls._constants.items()

    def __iter__(cls):
        return iter(cls._constants)

    def __contains__(cls, item):
        return item in cls._constants

    def __repr__(cls):
        return f"<ConstGroup {cls.__name__}: {cls._constants!r}>"


class ConstGroup(metaclass=MetaConst):
    """Base class for constant groups (dict + namespace)."""

    pass
