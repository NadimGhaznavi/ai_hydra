# ai_hydra/utils/HydraEventMQ.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0
#

from typing import Callable, Awaitable, Any
from dataclasses import dataclass, field

import json
import zmq
import zmq.asyncio

from ai_hydra.constants.DHydra import DHydraLog
from ai_hydra.constants.DHydraMQ import DEvent, DHydraMQ, DHydraMQDef


@dataclass(slots=True, frozen=True)
class EventMsg:
    level: DHydraLog
    message: str | None = None
    ev_type: str | None = None
    payload: dict[Any, Any] = field(default_factory=dict)


class HydraEventMQ:

    def __init__(self, client_id: str, pub_func):
        self._client_id = client_id
        self._pub_event = pub_func

    async def publish(self, event: EventMsg):
        msg_dict = {
            DEvent.SENDER: self._client_id,
            DEvent.LEVEL: event.level,
            DEvent.MESSAGE: event.message,
            DEvent.EV_TYPE: event.ev_type,
            DEvent.PAYLOAD: event.payload,
        }
        await self._pub_event(msg_dict)
