# ai_hydra/server/HydraServer.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

import argparse
import asyncio
import signal
from typing import Callable, Optional

from ai_hydra.constants.DHydra import (
    DHydraLog,
    DHydraLogDef,
    DHydraRouterDef,
    DHydraServerDef,
    DMethod,
    DModule,
)
from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.utils.HydraMQ import HydraMQ, HydraMsg


class HydraServer:
    """
    Abstract base class for HydraServer implementations.

    Provides common ZeroMQ-based server functionality that binds to a port
    and handles client requests using the REQ/REP pattern. Subclasses must
    implement application-specific message handling logic.
    """

    def __init__(
        self,
        address: str = "*",
        port: int = DHydraServerDef.PORT,
        router_address: str = DHydraRouterDef.HOSTNAME,
        router_port: int = DHydraRouterDef.PORT,
        identity: str = DModule.HYDRA_SERVER,
        log_level: DHydraLog = DHydraLogDef.DEFAULT_LOG_LEVEL,
    ):
        """
        Initialize the HydraServer with binding parameters.

        Args:
            address (str): The address to bind to (default: "*" for all
                interfaces)
            port (int): The port to bind to
            server_id (str): Identifier for logging purposes
        """

        self.address = address
        self.port = port
        self.router_address = router_address
        self.router_port = router_port
        self.identity = identity

        self._methods: dict[str, Callable[[HydraMsg], object]] = {
            str(DMethod.PING): self.ping,
        }

        # Messaging stub
        self.mq: Optional[HydraMQ] = None

        # Structured console logs
        self.log = HydraLog(
            client_id=self.identity, log_level=log_level, to_console=True
        )

        # Shutdown coordination
        self._stop_event: asyncio.Event | None = None
        self._main_task: asyncio.Task[None] | None = None

    def _install_signal_handlers(self, stop_event: asyncio.Event) -> None:
        """
        Prefer signal handlers when available (Unix). On Windows,
        add_signal_handler may not be implemented; Ctrl-C will still raise
        KeyboardInterrupt.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        def _request_stop() -> None:
            if not stop_event.is_set():
                self.log.info("Shutdown requested (signal)")
                stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
            except (NotImplementedError, RuntimeError):
                # Not supported on some platforms/event loops
                pass

    def loglevel(self, log_level: DHydraLog) -> None:
        """
        Initialize console logging for the server instance.
        """
        self.log = HydraLog(
            client_id=self.identity, log_level=log_level, to_console=True
        )

    async def _main_loop(self, stop_event: asyncio.Event) -> None:
        self.mq = HydraMQ(
            router_address=self.router_address,
            router_port=self.router_port,
            identity=self.identity,
            srv_methods=self._methods,
        )
        self.mq.start()

        try:
            while not stop_event.is_set():
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            # Normal during shutdown
            raise

    async def ping(self, msg: HydraMsg) -> None:
        self.log.info(f"Received ping from {msg.sender}")
        reply_msg = HydraMsg(
            sender=DModule.HYDRA_SERVER,
            target=msg.sender,
            method=DMethod.PONG,
        )
        if self.mq is not None:
            await self.mq.send(reply_msg)
            self.log.info(f"Sent pong to {msg.sender}")

    async def run(self) -> None:
        """Entrypoint for running the server inside an event loop."""
        self._stop_event = asyncio.Event()
        stop_event = self._stop_event  # local alias for closure safety
        self._install_signal_handlers(stop_event)

        self.log.info("Initialized, starting main loop")
        self._main_task = asyncio.create_task(
            self._main_loop(stop_event), name="hydra-server-main"
        )

        try:
            await self._stop_event.wait()
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        """Graceful shutdown: cancel main task, close MQ, cancel stragglers."""
        self.log.info("Shutting down...")

        # Cancel main loop task if still running
        if self._main_task is not None and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass

        # Stop/close MQ if available
        if self.mq is not None:
            stop = getattr(self.mq, "stop", None)
            if callable(stop):
                try:
                    maybe_awaitable = stop()
                    if asyncio.iscoroutine(maybe_awaitable):
                        await maybe_awaitable
                except Exception as e:
                    self.log.error(f"Error stopping HydraMQ: {e}")
            self.mq = None

        # Cancel any other pending tasks (keeps asyncio.run() happy)
        current = asyncio.current_task()
        pending = [
            t for t in asyncio.all_tasks() if t is not current and not t.done()
        ]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        self.log.info("Shutdown complete.")


async def amain() -> None:
    parser = argparse.ArgumentParser(description="Hydra ZeroMQ server")
    parser.add_argument("--address", default="*", help="Address to bind to")
    parser.add_argument(
        "--log_level",
        default=DHydraLogDef.DEFAULT_LOG_LEVEL,
        help="Log level [DEBUG|INFO|WARNING|ERROR|CRITICAL]",
    )
    parser.add_argument(
        "--port", type=int, default=DHydraServerDef.PORT, help="Server port"
    )
    parser.add_argument(
        "--router-address",
        default=DHydraRouterDef.HOSTNAME,
        help="HydraRouter hostname",
    )
    parser.add_argument(
        "--router-port",
        type=int,
        default=DHydraRouterDef.PORT,
        help="HydraRouter main port",
    )
    parser.add_argument(
        "--identity", default=DModule.HYDRA_SERVER, help="Server identity"
    )
    args = parser.parse_args()

    server = HydraServer(
        address=args.address,
        port=args.port,
        router_address=args.router_address,
        router_port=args.router_port,
        identity=args.identity,
        log_level=args.log_level,
    )

    await server.run()


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        # If signal handlers aren't supported, Ctrl-C lands here.
        # asyncio.run will already have cancelled tasks; keep this quiet and
        # clean.
        pass


if __name__ == "__main__":
    main()
