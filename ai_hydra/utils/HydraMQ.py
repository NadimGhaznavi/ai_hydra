# ai_hydra/utils/HydraMQ.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0
#

import asyncio
import time
from collections.abc import Awaitable, Callable

import zmq
import zmq.asyncio
from zmq.sugar.frame import Frame

from ai_hydra.constants.DHydra import (
    DHydra,
    DHydraRouterDef,
    DMethod,
    DModule,
)
from ai_hydra.constants.DHydraTui import DLabel
from ai_hydra.utils.HydraMsg import HydraMsg


def _ensure_bytes(data: bytes | Frame) -> bytes:
    return data.bytes if isinstance(data, Frame) else data


class HydraMQ:
    """
    Async ZeroMQ client for HydraRouter communication.

    HydraMQ provides an async DEALER socket client that connects to a
    HydraRouter instance. It handles message serialization, heartbeats,
    and connection lifecycle.

    The client uses a DEALER socket which allows asynchronous bidirectional
    communication through a ROUTER-based message broker.

    Example:
        mq = HydraMQ(
            router_address="localhost",
            router_port=5757,
            id="my-service"
        )

        # Send message
        msg = HydraMsg(
            sender=mq.identity,
            target="other-service",
            method="ping",
            payload={"data": "test"}
        )
        await mq.send(msg)

        # Receive message
        response = await mq.recv()
        print(response.payload)

        # Cleanup
        await mq.quit()
    """

    def __init__(
        self,
        router_address: str = DHydraRouterDef.HOSTNAME,
        router_port: int = DHydraRouterDef.PORT,
        router_hb_port: int = DHydraRouterDef.HEARTBEAT_PORT,
        identity: str = DModule.HYDRA_MQ,
        srv_methods: (
            dict[str, Callable[[HydraMsg], object | Awaitable[object]]] | None
        ) = None,
    ) -> None:
        """
        Initialize HydraMQ client.

        Args:
            router_address: Hostname/IP of the HydraRouter
            router_port: Port number of the HydraRouter
            id: Base identifier for this client (random suffix added)
            heartbeat_enabled: Whether to send periodic heartbeats

        Returns:
            None
        """
        self.router = router_address
        self.port = router_port
        self.hb_port = router_hb_port
        # Legacy parameters retained for compatibility with existing callers.
        self.srv_methods = srv_methods or {}

        # Create async ZeroMQ context and DEALER socket
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.hb_socket = self.ctx.socket(zmq.DEALER)

        # Generate unique identity: base-id + random 4-char suffix
        self.identity = identity

        # Set ZeroMQ socket identity (must be bytes)
        self.socket.setsockopt(zmq.IDENTITY, self.identity.encode("utf-8"))
        self.hb_socket.setsockopt(zmq.IDENTITY, self.identity.encode("utf-8"))

        # Build router address
        self.router_addr = f"tcp://{self.router}:{self.port}"
        self.router_hb_addr = f"tcp://{self.router}:{self.hb_port}"

        # Asyncio control events
        self.stop_event = asyncio.Event()
        self.heartbeat_stop_event = asyncio.Event()
        self._stopped = False

        # Connect to router
        self.socket.connect(self.router_addr)
        self.hb_socket.connect(self.router_hb_addr)

        # Placeholder for heartbeat task
        self.heartbeat_task: asyncio.Task[None] | None = None
        self.srv_task: asyncio.Task[None] | None = None

        if self.srv_methods:
            self.srv_stop_event = asyncio.Event()
            self.srv_pause_event = asyncio.Event()

        # A float holding time.time() for when the last heartbeat reply was
        # received
        self._last_heartbeat: float = 0.0

        # Flag to determine if start() has been called
        self._started = False

    async def bg_listen(self) -> None:
        try:
            while not self.srv_stop_event.is_set():
                # Handle pause
                if self.srv_pause_event.is_set():
                    await asyncio.sleep(0.1)
                    continue

                try:
                    message_data = await asyncio.wait_for(
                        self.socket.recv(copy=True),
                        timeout=DHydra.NETWORK_TIMEOUT,
                    )
                    hydra_msg = HydraMsg.from_json(_ensure_bytes(message_data))
                    method = hydra_msg.method
                    handler = self.srv_methods.get(method)
                    if handler is not None:
                        result = handler(hydra_msg)
                        if asyncio.iscoroutine(result):
                            await result
                    else:
                        print(f"{DLabel.ERROR}: Unhandled method {method}")

                except asyncio.TimeoutError:
                    # No message was received, continue...
                    pass
                except Exception as e:
                    print(f"{DLabel.ERROR}: {e}")

        except asyncio.CancelledError:
            # normal during shutdown
            raise
        except Exception as e:
            print(f"{DLabel.ERROR}: {e}")
            # let the task end; caller can decide what to do
            return

    def connected(self) -> bool:
        if self._last_heartbeat == 0:
            return False

        interval = time.time() - self._last_heartbeat
        if interval > (2 * DHydra.HEARTBEAT_INTERVAL):
            return False

        return True

    def _ignore_zmq_teardown(
        self, action: Callable[[], None], what: str
    ) -> None:
        try:
            action()
        except zmq.ZMQError as e:
            # expected during shutdown races / already-closed sockets
            print(f"{DLabel.DEBUG}: ignoring {what} during shutdown: {e}")

    async def quit(self) -> None:
        """
        Cleanly shutdown the HydraMQ client.

        Stops heartbeat task, disconnects from router, and cleans up
        ZeroMQ resources.

        Returns:
            None
        """
        if self._stopped:
            return
        self._stopped = True

        # Stop heartbeat task
        if self.heartbeat_task is not None:
            self.heartbeat_stop_event.set()
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            finally:
                self.heartbeat_task = None

        if self.srv_task is not None:
            self.srv_stop_event.set()
            self.srv_task.cancel()
            try:
                await self.srv_task
            except asyncio.CancelledError:
                pass

        # Disconnect and cleanup
        # Disconnect and cleanup (guarded: ignore errors during teardown)
        # Disconnect and cleanup
        try:
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
        finally:
            try:
                self.ctx.term()
            except zmq.ZMQError as e:
                print(
                    f"{DLabel.DEBUG}: ignoring ctx.term() during shutdown: {e}"
                )

    async def recv(self) -> HydraMsg:
        """
        Receive a HydraMsg from the router.

        Waits for an incoming message, deserializes it, and returns
        a HydraMsg instance.

        Args:
            timeout: Maximum time to wait for a message in seconds

        Returns:
            HydraMsg instance

        Raises:
            asyncio.TimeoutError: If no message received within timeout
            zmq.ZMQError: If receive operation fails
            json.JSONDecodeError: If message is not valid JSON
        """
        # DEALER socket receives single frame from ROUTER
        # ROUTER sends [client_identity, message], but DEALER
        # automatically strips the identity, leaving just [message]
        message_data = None
        message_data = await asyncio.wait_for(
            self.socket.recv(copy=True), timeout=DHydra.NETWORK_TIMEOUT
        )
        if message_data is not None:
            return HydraMsg.from_json(_ensure_bytes(message_data))

    async def send(self, msg: HydraMsg) -> None:
        """
        Send a HydraMsg through the router.

        Serializes the message to JSON and sends it through the
        DEALER socket to the connected ROUTER.

        Args:
            msg: HydraMsg instance to send

        Returns:
            None

        Raises:
            zmq.ZMQError: If send operation fails
        """
        # DEALER socket automatically prepends identity when sending to ROUTER
        await self.socket.send(msg.to_json())

    def _split_router_frames(
        self, frames: list[bytes]
    ) -> tuple[bytes, bytes, list[bytes]]:
        """
        Returns (sender, payload, routing_prefix)
        routing_prefix is what you should echo back before payload.
        """
        if len(frames) < 2:
            raise ValueError(f"Expected >=2 frames, got {len(frames)}")

        sender = frames[0]

        if len(frames) >= 3 and frames[1] == b"":
            return sender, frames[-1], [sender, b""]
        else:
            return sender, frames[-1], [sender]

    def start(self) -> None:
        if self._started:
            return
        if self.srv_methods and self.srv_task is None:
            self.srv_task = asyncio.create_task(
                self.bg_listen(), name="hydra-mq-listen"
            )
        # Start heartbeat task
        self.heartbeat_task = asyncio.create_task(
            self.start_heartbeat_bg(), name="hydra-mq-heartbeat"
        )
        self._started = True

    def started(self) -> bool:
        return self._started

    async def stop(self) -> None:
        """Alias for quit(), but safe to call multiple times."""
        await self.quit()

    async def start_heartbeat_bg(self) -> None:
        """
        Periodic heartbeat loop to keep connection alive.

        Sends heartbeat messages to the router at regular intervals
        to indicate this client is still active.

        Returns:
            None
        """
        while not self.heartbeat_stop_event.is_set():
            msg = HydraMsg(
                sender=self.identity,
                target=DModule.HYDRA_ROUTER,
                method=DMethod.HEARTBEAT,
            )
            await self.hb_socket.send(msg.to_json())

            try:
                message_data = await asyncio.wait_for(
                    self.hb_socket.recv(copy=True),
                    timeout=DHydra.NETWORK_TIMEOUT,
                )
                reply = HydraMsg.from_json(_ensure_bytes(message_data))

                if reply.method == DMethod.HEARTBEAT_REPLY:
                    self._last_heartbeat = time.time()

            except asyncio.TimeoutError:
                # Just continue and try again
                pass

            await asyncio.sleep(DHydra.HEARTBEAT_INTERVAL)
