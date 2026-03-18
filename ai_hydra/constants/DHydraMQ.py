# ai_hydra/constants/DHydraMQ.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Final


class DEvent:
    DATA: Final[str] = "data"
    EV_TYPE: Final[str] = "ev_type"
    LEVEL: Final[str] = "level"
    MESSAGE: Final[str] = "message"
    PAYLOAD: Final[str] = "payload"
    SENDER: Final[str] = "sender"


class DHydraMQ:
    HEARTBEAT: Final[str] = "heartbeat"
    MSGS: Final[str] = "msgs"
    PER_EP: Final[str] = "per_ep"
    PER_STEP: Final[str] = "per_step"
    SCORES: Final[str] = "scores"
    TIMER: Final[str] = "timer"
    UTF_8: Final[str] = "utf-8"


class DHydraMQDef:
    TOPIC_PREFIX: Final[str] = "ai_hydra"
    PER_STEP_TOPIC: Final[str] = "per_step_topic"
    PER_EPISODE_TOPIC: Final[str] = "per_episode_topic"
    EVENTS_TOPIC: Final[str] = "events_topic"
    SCORES_TOPIC: Final[str] = "scores_topic"
    MAX_BATCH_TIME: Final[float] = 0.5
    MAX_BATCH_SIZE: Final[int] = 100
