# ai_hydra/utils/HydraServerMQ.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0
#

from typing import Any
from collections.abc import Callable, Awaitable
import inspect

import zmq
import zmq.asyncio
import asyncio
import json

from ai_hydra.constants.DHydra import (
    DHydra,
    DHydraMQ,
    DHydraRouterDef,
    DModule,
    DHydraServerDef,
    DHydraMQDef,
)
from ai_hydra.utils.HydraMsg import HydraMsg
from ai_hydra.zmq.HydraBaseMQ import HydraBaseMQ

MsgHandler = Callable[[HydraMsg], Any | Awaitable[Any]]


class HydraServerMQ(HydraBaseMQ):
    def __init__(
        self,
        *,
        router_address: str = DHydraRouterDef.HOSTNAME,
        router_port: int = DHydraRouterDef.PORT,
        router_hb_port: int = DHydraRouterDef.HEARTBEAT_PORT,
        identity: str = DModule.HYDRA_MQ,
        srv_methods: dict[str, MsgHandler] | None = None,
        pub_port: int = DHydraServerDef.PUB_PORT,
        topic_prefix: str = DHydraMQDef.TOPIC_PREFIX,
    ) -> None:

        super().__init__(
            router_address=router_address,
            router_port=router_port,
            router_hb_port=router_hb_port,
            identity=identity,
            topic_prefix=topic_prefix,
        )
        self.pub_port = pub_port
        self.srv_methods = srv_methods or {}

        self.pub_socket: zmq.asyncio.Socket | None = None
        self.pub_addr = f"tcp://*:{pub_port}"
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind(self.pub_addr)

        self.listen_task: asyncio.Task[None] | None = None
        self.listen_stop_event = asyncio.Event()

    async def bg_listen(self) -> None:
        try:
            while not self.listen_stop_event.is_set():

                try:
                    message_data = await asyncio.wait_for(
                        self.socket.recv(copy=True),
                        timeout=DHydra.NETWORK_TIMEOUT,
                    )
                    hydra_msg = HydraMsg.from_json(
                        self._ensure_bytes(message_data)
                    )
                    method = hydra_msg.method
                    handler = self.srv_methods.get(method)
                    if handler is not None:
                        result = handler(hydra_msg)
                        if inspect.isawaitable(result):
                            await result
                    else:
                        print(f"ERROR: Unhandled method {method}")

                except asyncio.TimeoutError:
                    # No message was received, continue...
                    pass
                except Exception as e:
                    print(f"ERROR: {e}")

        except asyncio.CancelledError:
            # normal during shutdown
            raise
        except Exception as e:
            print(f"ERROR: {e}")
            # let the task end; caller can decide what to do
            return

    async def _publish(self, topic_suffix: str, payload: dict) -> None:
        """
        Publish telemetry as multipart [topic, json_bytes].
        Topic is f"{topic_prefix}.{topic_suffix}"
        """
        if self.pub_socket is None:
            raise RuntimeError("publish() called but PUB not configured")

        topic = self.topic(topic_suffix).encode(DHydraMQ.UTF_8)
        data = json.dumps(payload, separators=(",", ":")).encode(
            DHydraMQ.UTF_8
        )
        await self.pub_socket.send_multipart([topic, data])

    async def publish_per_step(self, payload: dict) -> None:
        await self._publish(DHydraMQDef.PER_STEP_TOPIC, payload=payload)

    async def publish_per_episode(self, payload: dict) -> None:
        await self._publish(DHydraMQDef.PER_EPISODE_TOPIC, payload=payload)

    async def publish_scores(self, payload: dict) -> None:
        await self._publish(DHydraMQDef.SCORES_TOPIC, payload=payload)

    def start(self) -> None:
        super().start()

        if self.listen_task is None:
            self.listen_task = asyncio.create_task(
                self.bg_listen(), name="listen"
            )
