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
)
from ai_hydra.constants.DHydraTui import DField

from ai_hydra.utils.HydraMsg import HydraMsg
from ai_hydra.zmq.HydraBaseMQ import HydraBaseMQ
from ai_hydra.zmq.HydraMsgBatch import HydraMsgBatch

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

        # Per topic batch storage
        self._per_ep_batch = HydraMsgBatch()
        self._per_step_batch = HydraMsgBatch()
        self._scores_batch = HydraMsgBatch()

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

                # Per epsiode batch
                now = time.monotonic()
                local_storage = None
                await self._per_ep_lock.acquire()
                if (
                    self._per_ep_msgs
                    and self._per_ep_timer
                    and (
                        now - self._per_ep_timer >= DHydraMQDef.MAX_BATCH_TIME
                    )
                ):
                    local_storage = self._per_ep_msgs
                    self._per_ep_msgs = []
                    self._per_ep_timer = None
                self._per_ep_lock.release()
                if local_storage is not None:
                    await self._publish(
                        topic=DHydraMQDef.PER_EPISODE_TOPIC,
                        method=DMethod.PER_EP_BATCH,
                        payload=local_storage,
                    )

                # Per step batch
                now = time.monotonic()
                local_storage = None
                await self._per_step_lock.acquire()
                if (
                    self._per_step_msgs
                    and self._per_step_timer
                    and (
                        now - self._per_step_timer
                        >= DHydraMQDef.MAX_BATCH_TIME
                    )
                ):
                    local_storage = self._per_step_msgs
                    self._per_step_msgs = []
                    self._per_step_timer = None
                self._per_step_lock.release()
                if local_storage is not None:
                    await self._publish(
                        topic=DHydraMQDef.PER_STEP_TOPIC,
                        method=DMethod.PER_STEP_BATCH,
                        payload=local_storage,
                    )

                # Scores batch
                now = time.monotonic()
                local_storage = None
                await self._scores_lock.acquire()
                if (
                    self._scores_msgs
                    and self._scores_timer
                    and (
                        now - self._scores_timer >= DHydraMQDef.MAX_BATCH_TIME
                    )
                ):
                    local_storage = self._scores_msgs
                    self._scores_msgs = []
                    self._scores_timer = None
                self._scores_lock.release()
                if local_storage is not None:
                    await self._publish(
                        topic=DHydraMQDef.SCORES_TOPIC,
                        method=DMethod.SCORES_BATCH,
                        payload=local_storage,
                    )

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"ERROR: {e}")

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

    async def publish_per_episode(self, payload: dict) -> None:
        """
        Received a new message to publish.
        """
        local_storage = None
        await self._per_ep_batch.acquire_lock()
        try:
            if self._per_ep_batch.is_empty():
                count_msg = HydraMsg(
                    sender=self.identity,
                    method=DMethod.COUNTER,
                    payload={DField.COUNT: self._per_ep_batch.batch_num()},
                )
                self._per_ep_batch.append(count_msg)
                self._per_ep_batch.append(payload)

            elif self._per_ep_batch.batch_size() >= DHydraMQDef.MAX_BATCH_SIZE:
                self._per_ep_batch.append(payload)
                local_storage = self._per_ep_batch.pop_batch()

            else:
                self._per_ep_batch.append(payload)
        finally:
            self._per_ep_batch.release_lock()

        if local_storage is not None:
            await self._publish(
                topic=DHydraMQDef.PER_EPISODE_TOPIC,
                method=DMethod.PER_EP_BATCH,
                payload=local_storage,
            )

    async def publish_per_step(self, payload: dict) -> None:
        await self._publish(
            batch=self._per_step_batch,
            payload=payload,
            topic=DHydraMQDef.PER_STEP_TOPIC,
            method=DMethod.PER_STEP_BATCH,
        )

    async def _publish(
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
        finally:
            batch.release_lock()

        if local_storage is not None:
            envelope = {
                DHydraMsg.SENDER: DModule.HYDRA_MGR,
                DHydraMsg.METHOD: method,
                DHydraMsg.PAYLOAD: local_storage,
                DHydraMsg.PROTOCOL_VERSION: DHydra.PROTOCOL_VERSION,
            }
            topic = self.topic(topic).encode(DHydraMQ.UTF_8)
            data = json.dumps(envelope, separators=(",", ":")).encode(
                DHydraMQ.UTF_8
            )
            await self.pub_socket.send_multipart([topic, data])

    async def publish_scores(self, payload: dict) -> None:
        await self._scores_lock.acquire()

        if len(self._scores_msgs) == 0:
            self._scores_timer = time.monotonic()
            count_msg = {
                DHydraMsg.SENDER: self.identity,
                DHydraMsg.METHOD: DMethod.COUNTER,
                DHydraMsg.PAYLOAD: {DField.COUNT: self._scores_count},
                DHydraMsg.PROTOCOL_VERSION: DHydra.PROTOCOL_VERSION,
            }
            self._scores_msgs.append(count_msg)
            self._scores_msgs.append(payload)
            self._scores_lock.release()
            self._scores_count += 1

        elif len(self._scores_msgs) >= DHydraMQDef.MAX_BATCH_SIZE:
            self._scores_msgs.append(payload)
            local_storage = self._scores_msgs
            self._scores_msgs = []
            self._scores_timer = None
            self._scores_lock.release()
            await self._publish(
                topic=DHydraMQDef.SCORES_TOPIC,
                method=DMethod.SCORES_BATCH,
                payload=local_storage,
            )

        else:
            self._scores_msgs.append(payload)
            self._scores_lock.release()

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
