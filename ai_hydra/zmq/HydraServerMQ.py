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
import traceback

import zmq
import zmq.asyncio
import asyncio
import json
import time

from ai_hydra.constants.DHydra import (
    DHydra,
    DHydraMQ,
    DHydraRouterDef,
    DModule,
    DHydraServerDef,
    DHydraMQDef,
    DMethod,
    DHydraMsg,
    DHydraLog,
)
from ai_hydra.constants.DHydraTui import DField

from ai_hydra.zmq.HydraMsg import HydraMsg
from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.zmq.HydraBaseMQ import HydraBaseMQ
from ai_hydra.zmq.HydraMsgBatch import HydraMsgBatch

MsgHandler = Callable[[HydraMsg], Any | Awaitable[Any]]


class HydraServerMQ(HydraBaseMQ):
    def __init__(
        self,
        log_level: DHydraLog,
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
        self.log = HydraLog(
            client_id="HydraServerMQ",
            log_level=log_level,
            to_console=True,
        )
        self.pub_port = pub_port
        self.srv_methods = srv_methods or {}

        # Per topic batch storage
        self._per_ep_batch = HydraMsgBatch()
        self._scores_batch = HydraMsgBatch()

        # Per-Step topic doesn't need batching, it's slow, because of the
        # user defined move_delay

        self.pub_socket: zmq.asyncio.Socket | None = None
        self.pub_addr = f"tcp://*:{pub_port}"
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind(self.pub_addr)

        self.listen_task: asyncio.Task[None] | None = None
        self.listen_stop_event = asyncio.Event()

        self.check_batches_task: asyncio.Task[None] | None = None
        self.check_batches_stop_event = asyncio.Event()

    async def bg_check_batches(self) -> None:
        try:
            while not self.check_batches_stop_event.is_set():
                await self._flush_timed_out_batch(
                    batch=self._per_ep_batch,
                    topic=DHydraMQDef.PER_EPISODE_TOPIC,
                    method=DMethod.PER_EP_BATCH,
                )
                await self._flush_timed_out_batch(
                    batch=self._scores_batch,
                    topic=DHydraMQDef.SCORES_TOPIC,
                    method=DMethod.SCORES_BATCH,
                )

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.log.critical(f"ERROR: {e}")
            self.log.critical(f"STACKTRACE: {traceback.format_exc()}")

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

    async def _bg_publish(
        self,
        *,
        topic: str,
        method: str,
        payload: list[dict],
    ) -> None:
        envelope = {
            DHydraMsg.SENDER: self.identity,
            DHydraMsg.METHOD: method,
            DHydraMsg.PAYLOAD: {DHydraMQ.MSGS: payload},
            DHydraMsg.PROTOCOL_VERSION: DHydra.PROTOCOL_VERSION,
        }
        topic_b = self.topic(topic).encode(DHydraMQ.UTF_8)
        data = json.dumps(envelope, separators=(",", ":")).encode(
            DHydraMQ.UTF_8
        )
        await self.pub_socket.send_multipart([topic_b, data])

    async def _enqueue_and_maybe_flush(
        self,
        lock: asyncio.Lock,
        msgs: list,
        timer_name: str,
        count_name: str,
        payload: dict,
        topic: str,
        method: str,
    ) -> None:

        await lock.acquire()

        timer = getattr(self, timer_name)
        count = getattr(self, count_name)

        # Start batch
        if len(msgs) == 0:
            setattr(self, timer_name, time.monotonic())

            count_msg = {
                DHydraMsg.SENDER: self.identity,
                DHydraMsg.METHOD: DMethod.COUNTER,
                DHydraMsg.PAYLOAD: {DField.COUNT: count},
                DHydraMsg.PROTOCOL_VERSION: DHydra.PROTOCOL_VERSION,
            }

            msgs.append(count_msg)
            msgs.append(payload)

            setattr(self, count_name, count + 1)
            lock.release()
            return

        # Normal append
        msgs.append(payload)

        # Size flush
        if len(msgs) >= DHydraMQDef.MAX_BATCH_SIZE:
            local_storage = msgs.copy()
            msgs.clear()
            setattr(self, timer_name, None)

            lock.release()

            await self._publish(
                topic=topic,
                method=method,
                payload=local_storage,
            )
            return

        lock.release()

    async def _flush_timed_out_batch(
        self,
        *,
        batch: HydraMsgBatch,
        topic: str,
        method: str,
    ) -> None:
        local_storage = None
        now = time.monotonic()

        await batch.acquire_lock()
        try:
            if batch.has_timed_out(now):
                local_storage = batch.pop_batch()
        finally:
            batch.release_lock()

        if local_storage is not None:
            await self._bg_publish(
                topic=topic,
                method=method,
                payload=local_storage,
            )

    async def publish_per_episode(self, payload: dict) -> None:
        """
        Received a new message to publish.
        """
        await self._batched_publish(
            batch=self._per_ep_batch,
            payload=payload,
            topic=DHydraMQDef.PER_EPISODE_TOPIC,
            method=DMethod.PER_EP_BATCH,
        )

    async def publish_per_step(self, payload: dict) -> None:
        topic = self.topic(DHydraMQDef.PER_STEP_TOPIC).encode(DHydraMQ.UTF_8)
        data = json.dumps(payload, separators=(",", ":")).encode(
            DHydraMQ.UTF_8
        )
        await self.pub_socket.send_multipart([topic, data])

    async def publish_scores(self, payload: dict) -> None:
        await self._batched_publish(
            batch=self._scores_batch,
            payload=payload,
            topic=DHydraMQDef.SCORES_TOPIC,
            method=DMethod.SCORES_BATCH,
        )

    async def _batched_publish(
        self, batch: HydraMsgBatch, payload: dict, topic: str, method: str
    ) -> None:
        """
        A better name for this function is _enqueue_and_maybe_publish(), but
        that's too verbose, even for me! :)
        """
        local_storage = None
        await batch.acquire_lock()
        try:
            if batch.is_empty():
                count_msg = HydraMsg(
                    sender=self.identity,
                    method=DMethod.COUNTER,
                    payload={DField.COUNT: batch.batch_num()},
                )
                batch.append(count_msg.to_dict())
                batch.append(payload)

            elif batch.batch_size() >= (DHydraMQDef.MAX_BATCH_SIZE - 1):
                batch.append(payload)
                local_storage = batch.pop_batch()

            else:
                batch.append(payload)

        except Exception as e:
            self.log.critical(f"ERROR: {e}")
            self.log.critical(f"STACKTRACE: {traceback.format_exc()}")

        finally:
            batch.release_lock()

        if local_storage is not None:
            envelope = {
                DHydraMsg.SENDER: DModule.HYDRA_MGR,
                DHydraMsg.METHOD: method,
                DHydraMsg.PAYLOAD: {DHydraMQ.MSGS: local_storage},
                DHydraMsg.PROTOCOL_VERSION: DHydra.PROTOCOL_VERSION,
            }
            topic = self.topic(topic).encode(DHydraMQ.UTF_8)
            data = json.dumps(envelope, separators=(",", ":")).encode(
                DHydraMQ.UTF_8
            )
            await self.pub_socket.send_multipart([topic, data])

    async def UNUSED_publish(
        self, topic: str, method: str, payload: dict
    ) -> None:
        payload = {
            DHydraMsg.SENDER: DModule.HYDRA_MGR,
            DHydraMsg.METHOD: method,
            DHydraMsg.PAYLOAD: payload,
            DHydraMsg.PROTOCOL_VERSION: DHydra.PROTOCOL_VERSION,
        }
        topic = self.topic(topic).encode(DHydraMQ.UTF_8)
        data = json.dumps(payload, separators=(",", ":")).encode(
            DHydraMQ.UTF_8
        )
        await self.pub_socket.send_multipart([topic, data])

    def start(self) -> None:
        super().start()

        if self.listen_task is None:
            self.listen_task = asyncio.create_task(
                self.bg_listen(), name="listen"
            )

        if self.check_batches_task is None:
            self.check_batches_task = asyncio.create_task(
                self.bg_check_batches(), name="check-batches"
            )
