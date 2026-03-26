# ai_hydra/utils/HydraClientMQ.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0
#
"""
_init_sub_socket()
connect_sub()
subscribe()
unsubscribe()
bg_sub_listen()
"""

from typing import Any
from collections.abc import Awaitable, Callable
import asyncio
import zmq, zmq.asyncio
import json
import traceback

from ai_hydra.constants.DHydra import (
    DHydra,
    DHydraRouterDef,
    DModule,
    DHydraServerDef,
)
from ai_hydra.constants.DHydraMQ import DHydraMQDef, DHydraMQ
from ai_hydra.zmq.HydraBaseMQ import HydraBaseMQ
from ai_hydra.zmq.HydraMsg import HydraMsg

SubHandler = Callable[[str, dict], Any | Awaitable[Any]]


class HydraClientMQ(HydraBaseMQ):
    def __init__(
        self,
        *,
        router_address: str = DHydraRouterDef.HOSTNAME,
        router_port: int = DHydraRouterDef.PORT,
        router_hb_port: int = DHydraRouterDef.HEARTBEAT_PORT,
        identity: str = DModule.HYDRA_MQ,
        topic_prefix: str = DHydraMQDef.TOPIC_PREFIX,
        server_address: str = DHydraServerDef.HOSTNAME,
        server_pub_port: int = DHydraServerDef.PUB_PORT,
        sub_methods: dict[str, SubHandler] | None = None,
    ) -> None:

        super().__init__(
            router_address=router_address,
            router_port=router_port,
            router_hb_port=router_hb_port,
            identity=identity,
            topic_prefix=topic_prefix,
        )
        self.srv_host = server_address
        self.sub_port = server_pub_port
        self.sub_methods = sub_methods or {}

        self.sub_socket: zmq.asyncio.Socket | None = None
        self.sub_task: asyncio.Task[None] | None = None
        self.sub_addr = f"tcp://{server_address}:{server_pub_port}"
        self.sub_socket = self.ctx.socket(zmq.SUB)

        self.sub_socket.connect(self.sub_addr)
        self.sub_stop_event = asyncio.Event()

    async def bg_sub_listen(self) -> None:
        """
        Background subscriber loop for telemetry.
        """
        if self.sub_socket is None:
            return

        try:
            while not self.sub_stop_event.is_set():
                try:
                    frames = await asyncio.wait_for(
                        self.sub_socket.recv_multipart(copy=True),
                        timeout=DHydra.NETWORK_TIMEOUT,
                    )

                    if len(frames) != 2:
                        print(
                            f"ERROR: telemetry expected 2 frames, got {len(frames)}"
                        )
                        continue

                    topic = self._ensure_bytes(frames[0]).decode(
                        DHydraMQ.UTF_8, errors="replace"
                    )
                    payload_bytes = self._ensure_bytes(frames[1])

                    try:
                        payload = json.loads(payload_bytes.decode("utf-8"))
                    except Exception as e:
                        print(f"ERROR: SUB JSON decode error: {e}")
                        continue

                    handler = self.sub_methods.get(topic)

                    if handler is not None:
                        result = handler(topic, payload)
                        if asyncio.iscoroutine(result):
                            await result

                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    print(f"ERROR: telemetry listen error: {e}")
                    print(f"STACKTRACK: {traceback.format_exc()}")

        except asyncio.CancelledError:
            raise

    def disable_events_sub(self) -> None:
        self._sub_set(False, self.topic(DHydraMQDef.EVENTS_TOPIC))

    def disable_per_step_sub(self) -> None:
        self._sub_set(False, self.topic(DHydraMQDef.PER_STEP_TOPIC))

    def disable_per_episode_sub(self) -> None:
        self._sub_set(False, self.topic(DHydraMQDef.PER_EPISODE_TOPIC))

    def disable_scores_sub(self) -> None:
        self._sub_set(False, self.topic(DHydraMQDef.SCORES_TOPIC))

    def enable_events_sub(self) -> None:
        self._sub_set(True, self.topic(DHydraMQDef.EVENTS_TOPIC))

    def enable_per_step_sub(self) -> None:
        self._sub_set(True, self.topic(DHydraMQDef.PER_STEP_TOPIC))

    def enable_per_episode_sub(self) -> None:
        self._sub_set(True, self.topic(DHydraMQDef.PER_EPISODE_TOPIC))

    def enable_scores_sub(self) -> None:
        self._sub_set(True, self.topic(DHydraMQDef.SCORES_TOPIC))

    async def quit(self) -> None:
        await super().quit()
        if self._stopped:
            return
        self._stopped = True
        self._started = False

        # 1. stop all tasks first
        self.hb_stop_event.set()
        self.sub_stop_event.set()

        if self.hb_task is not None:
            self.hb_task.cancel()
            try:
                await self.hb_task
            except asyncio.CancelledError:
                pass
            finally:
                self.hb_task = None

        if self.sub_task is not None:
            self.sub_task.cancel()
            try:
                await self.sub_task
            except asyncio.CancelledError:
                pass
            finally:
                self.sub_task = None

        # 2. close sockets
        self._ignore_zmq_teardown(
            lambda: self.socket.disconnect(self.router_addr),
            f"socket.disconnect({self.router_addr})",
        )
        self._ignore_zmq_teardown(
            lambda: self.socket.close(linger=0),
            "socket.close(linger=0)",
        )

        self._ignore_zmq_teardown(
            lambda: self.hb_socket.disconnect(self.router_hb_addr),
            f"hb_socket.disconnect({self.router_hb_addr})",
        )
        self._ignore_zmq_teardown(
            lambda: self.hb_socket.close(linger=0),
            "hb_socket.close(linger=0)",
        )

        if self.sub_socket is not None:
            self._ignore_zmq_teardown(
                lambda: self.sub_socket.close(linger=0),
                "sub_socket.close(linger=0)",
            )
            self.sub_socket = None

        # 3. terminate context once
        try:
            self.ctx.term()
        except zmq.ZMQError as e:
            print(f"DEBUG: ignoring ctx.term() during shutdown: {e}")

    def start(self) -> None:
        super().start()

        if self.sub_task is None:
            self.sub_task = asyncio.create_task(
                self.bg_sub_listen(), name="sub_listen"
            )

    def _sub_set(self, subscribe: bool, prefix: str) -> None:
        if self.sub_socket is None:
            raise RuntimeError("SUB not configured")
        opt = zmq.SUBSCRIBE if subscribe else zmq.UNSUBSCRIBE
        self.sub_socket.setsockopt(opt, prefix.encode(DHydraMQ.UTF_8))
