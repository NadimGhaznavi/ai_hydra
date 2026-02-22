"""
AI Hydra Router

Enhanced ZeroMQ router between TUI clients and the AI Hydra server.
Provides centralized message routing with comprehensive validation,
client management, and error handling.
"""

import asyncio
import logging
import time
import argparse
from copy import deepcopy
from typing import Dict, Any, List, Optional, Set

import zmq
import zmq.asyncio

from ai_hydra.router_constants import RouterConstants, RouterLabels
from ai_hydra.validation import MessageValidator, validate_message
from ai_hydra.exceptions import (
    HydraRouterError,
    MessageValidationError,
    ConnectionError as RouterConnectionError,
    ClientRegistrationError,
    HeartbeatError,
    RoutingError,
)


class ClientRegistry:
    """Thread-safe client registry for tracking connected clients and servers."""

    def __init__(self):
        """Initialize client registry."""
        self.clients: Dict[str, tuple] = (
            {}
        )  # client_id -> (client_type, last_heartbeat)
        self.lock = asyncio.Lock()

    async def register_client(self, client_id: str, client_type: str) -> None:
        """
        Register a new client.

        Args:
            client_id: Unique client identifier
            client_type: Type of client (HydraClient, HydraServer, etc.)

        Raises:
            ClientRegistrationError: If registration fails
        """
        if not client_id or not isinstance(client_id, str):
            raise ClientRegistrationError(
                "Client ID must be non-empty string",
                client_id=client_id,
                operation="register",
            )

        if not client_type or client_type not in MessageValidator.VALID_SENDERS:
            raise ClientRegistrationError(
                f"Invalid client type: {client_type}",
                client_id=client_id,
                client_type=client_type,
                operation="register",
            )

        async with self.lock:
            self.clients[client_id] = (client_type, time.time())

    async def update_heartbeat(self, client_id: str) -> None:
        """
        Update client heartbeat timestamp.

        Args:
            client_id: Client identifier

        Raises:
            ClientRegistrationError: If client not found
        """
        async with self.lock:
            if client_id not in self.clients:
                raise ClientRegistrationError(
                    f"Client not registered: {client_id}",
                    client_id=client_id,
                    operation="update_heartbeat",
                )

            client_type, _ = self.clients[client_id]
            self.clients[client_id] = (client_type, time.time())

    async def remove_client(self, client_id: str) -> None:
        """
        Remove client from registry.

        Args:
            client_id: Client identifier
        """
        async with self.lock:
            if client_id in self.clients:
                del self.clients[client_id]

    async def get_clients_by_type(self, client_type: str) -> List[str]:
        """
        Get all clients of specified type.

        Args:
            client_type: Type of clients to retrieve

        Returns:
            List of client IDs
        """
        async with self.lock:
            return [
                client_id
                for client_id, (ctype, _) in self.clients.items()
                if ctype == client_type
            ]

    async def prune_inactive_clients(self, timeout: float) -> List[str]:
        """
        Remove clients that haven't sent heartbeats within timeout.

        Args:
            timeout: Timeout threshold in seconds

        Returns:
            List of removed client IDs
        """
        now = time.time()
        removed_clients = []

        async with self.lock:
            clients_to_remove = []
            for client_id, (client_type, last_heartbeat) in self.clients.items():
                if now - last_heartbeat > timeout:
                    clients_to_remove.append(client_id)
                    removed_clients.append(client_id)

            for client_id in clients_to_remove:
                del self.clients[client_id]

        return removed_clients

    async def get_client_count(self) -> Dict[str, int]:
        """
        Get count of clients by type.

        Returns:
            Dictionary mapping client type to count
        """
        async with self.lock:
            counts = {}
            for client_type, _ in self.clients.values():
                counts[client_type] = counts.get(client_type, 0) + 1
            return counts

    async def get_all_clients(self) -> Dict[str, tuple]:
        """
        Get copy of all registered clients.

        Returns:
            Dictionary mapping client_id to (client_type, last_heartbeat)
        """
        async with self.lock:
            return deepcopy(self.clients)


class MessageRouter:
    """Handles message routing logic between clients and servers."""

    def __init__(self, client_registry: ClientRegistry, logger: logging.Logger):
        """
        Initialize message router.

        Args:
            client_registry: Client registry instance
            logger: Logger instance
        """
        self.client_registry = client_registry
        self.logger = logger

    async def route_message(
        self,
        sender_id: str,
        sender_type: str,
        elem: str,
        data: Any,
        socket: zmq.asyncio.Socket,
    ) -> None:
        """
        Route message based on sender type and routing rules.

        Args:
            sender_id: ID of message sender
            sender_type: Type of sender (HydraClient, HydraServer, etc.)
            elem: Message element/type
            data: Message data
            socket: ZMQ socket for sending responses

        Raises:
            RoutingError: If routing fails
        """
        try:
            if sender_type == RouterConstants.HYDRA_CLIENT:
                await self._route_client_message(sender_id, elem, data, socket)
            elif sender_type == RouterConstants.HYDRA_SERVER:
                await self._route_server_message(sender_id, elem, data, socket)
            else:
                raise RoutingError(
                    f"Unknown sender type: {sender_type}",
                    message_type=elem,
                    sender_id=sender_id,
                    routing_rule="sender_type_dispatch",
                )
        except Exception as e:
            self.logger.error(f"Message routing failed: {e}")
            raise RoutingError(
                f"Failed to route message: {str(e)}",
                message_type=elem,
                sender_id=sender_id,
            )

    async def _route_client_message(
        self, sender_id: str, elem: str, data: Any, socket: zmq.asyncio.Socket
    ) -> None:
        """Route message from client to server."""
        # Get all connected servers
        servers = await self.client_registry.get_clients_by_type(
            RouterConstants.HYDRA_SERVER
        )

        if not servers:
            # No server connected - inform the client
            error_msg = {RouterConstants.ERROR: "No AI Hydra server connected"}
            await socket.send_multipart(
                [sender_id.encode(), zmq.utils.jsonapi.dumps(error_msg)]
            )
            self.logger.warning(
                f"No server connected for client {sender_id} request: {elem}"
            )
            return

        # Construct message for server
        msg = {
            RouterConstants.SENDER: RouterConstants.HYDRA_CLIENT,
            RouterConstants.ELEM: elem,
            RouterConstants.DATA: data,
        }
        msg_bytes = zmq.utils.jsonapi.dumps(msg)

        # Send to all connected servers (usually just one)
        for server_id in servers:
            try:
                await socket.send_multipart([server_id.encode(), msg_bytes])
                self.logger.debug(
                    f"Forwarded {elem} from client {sender_id} to server {server_id}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to forward message to server {server_id}: {e}"
                )

        # Acknowledge the client
        ack_msg = {RouterConstants.STATUS: RouterConstants.OK}
        await socket.send_multipart(
            [sender_id.encode(), zmq.utils.jsonapi.dumps(ack_msg)]
        )

    async def _route_server_message(
        self, sender_id: str, elem: str, data: Any, socket: zmq.asyncio.Socket
    ) -> None:
        """Route message from server to clients."""
        # Drop status/error messages (they're handled locally)
        if elem in [RouterConstants.STATUS, RouterConstants.ERROR]:
            return

        # Get all connected clients
        clients = await self.client_registry.get_clients_by_type(
            RouterConstants.HYDRA_CLIENT
        )

        if not clients:
            self.logger.debug(f"No clients connected to receive server message: {elem}")
            return

        # Construct message for clients
        msg = {
            RouterConstants.SENDER: RouterConstants.HYDRA_SERVER,
            RouterConstants.ELEM: elem,
            RouterConstants.DATA: data,
        }
        msg_bytes = zmq.utils.jsonapi.dumps(msg)

        # Broadcast to all connected clients
        for client_id in clients:
            if client_id != sender_id:  # Don't send back to sender
                try:
                    await socket.send_multipart([client_id.encode(), msg_bytes])
                    self.logger.debug(
                        f"Broadcast {elem} from server {sender_id} to client {client_id}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to send message to client {client_id}: {e}"
                    )


class BackgroundTaskManager:
    """Manages background tasks for router operations."""

    def __init__(self, client_registry: ClientRegistry, logger: logging.Logger):
        """
        Initialize background task manager.

        Args:
            client_registry: Client registry instance
            logger: Logger instance
        """
        self.client_registry = client_registry
        self.logger = logger
        self.tasks: Set[asyncio.Task] = set()
        self.shutdown_event = asyncio.Event()

    async def start_tasks(self) -> None:
        """Start all background tasks."""
        # Client pruning task
        prune_task = asyncio.create_task(self._client_pruning_task())
        self.tasks.add(prune_task)

        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitoring_task())
        self.tasks.add(health_task)

        self.logger.info(f"Started {len(self.tasks)} background tasks")

    async def stop_tasks(self) -> None:
        """Stop all background tasks."""
        self.shutdown_event.set()

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        self.tasks.clear()
        self.logger.info("Stopped all background tasks")

    async def _client_pruning_task(self) -> None:
        """Background task to prune inactive clients."""
        while not self.shutdown_event.is_set():
            try:
                timeout = RouterConstants.HEARTBEAT_INTERVAL * 3
                removed_clients = await self.client_registry.prune_inactive_clients(
                    timeout
                )

                if removed_clients:
                    self.logger.info(f"Pruned inactive clients: {removed_clients}")

                # Wait for next pruning cycle
                await asyncio.wait_for(
                    self.shutdown_event.wait(),
                    timeout=RouterConstants.HEARTBEAT_INTERVAL * 2,
                )

            except asyncio.TimeoutError:
                continue  # Normal timeout, continue pruning
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Client pruning task error: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

    async def _health_monitoring_task(self) -> None:
        """Background task for health monitoring and metrics."""
        while not self.shutdown_event.is_set():
            try:
                # Log client counts periodically
                client_counts = await self.client_registry.get_client_count()
                if client_counts:
                    count_str = ", ".join(
                        f"{ctype}: {count}" for ctype, count in client_counts.items()
                    )
                    self.logger.info(f"Connected clients - {count_str}")

                # Wait for next monitoring cycle
                await asyncio.wait_for(
                    self.shutdown_event.wait(), timeout=30  # Monitor every 30 seconds
                )

            except asyncio.TimeoutError:
                continue  # Normal timeout, continue monitoring
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring task error: {e}")
                await asyncio.sleep(1)  # Brief pause before retry


class HydraRouter:
    """Enhanced ZeroMQ router between TUI clients and the AI Hydra server."""

    def __init__(
        self,
        router_address: str = "0.0.0.0",
        router_port: int = 5556,
        log_level: str = "INFO",
    ):
        """
        Initialize the AI Hydra router with enhanced capabilities.

        Args:
            router_address: Address to bind the router to
            router_port: Port to bind the router to
            log_level: Logging level
        """
        # Setup logging
        self.logger = logging.getLogger("HydraRouter")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Initialize components
        self.validator = MessageValidator()
        self.client_registry = ClientRegistry()
        self.message_router = MessageRouter(self.client_registry, self.logger)
        self.task_manager = BackgroundTaskManager(self.client_registry, self.logger)

        # Initialize ZMQ context
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.ROUTER)

        # Bind to the router service
        router_service = f"tcp://{router_address}:{router_port}"
        try:
            self.socket.bind(router_service)
            self.logger.info(f"Enhanced Hydra Router started on {router_service}")
        except zmq.error.ZMQError as e:
            self.logger.critical(f"Failed to bind router to {router_service}: {e}")
            raise RouterConnectionError(
                f"Failed to bind router to {router_service}",
                address=router_address,
                port=router_port,
            ) from e

        # Router state
        self.is_running = False

    async def start(self) -> None:
        """Start the router and all background tasks."""
        try:
            self.is_running = True

            # Start background tasks
            await self.task_manager.start_tasks()

            self.logger.info("Hydra Router fully initialized and running")

        except Exception as e:
            self.logger.error(f"Failed to start router: {e}")
            await self.shutdown()
            raise

    async def handle_requests(self) -> None:
        """Continuously route messages between clients and servers with enhanced error handling."""
        self.logger.info("Enhanced router message handling started")

        while self.is_running:
            try:
                # ROUTER sockets prepend an identity frame
                frames = await self.socket.recv_multipart()

                if len(frames) != 2:
                    identity_str = frames[0].decode() if frames else "unknown"
                    self._log_frame_error(frames, identity_str)
                    continue

                identity = frames[0]
                identity_str = identity.decode()
                msg_bytes = frames[1]

                # Parse JSON message
                try:
                    msg = zmq.utils.jsonapi.loads(msg_bytes)
                except (ValueError, TypeError) as e:
                    self._log_json_parse_error(msg_bytes, e, identity_str)
                    continue

                # Validate message format
                is_valid, error_msg = self.validator.validate_router_message(msg)
                if not is_valid:
                    self._log_malformed_message(msg, error_msg, identity_str)
                    continue

                # Extract message components
                sender_type = msg[RouterConstants.SENDER]
                elem = msg[RouterConstants.ELEM]
                data = msg.get(RouterConstants.DATA, {})

                # Debug logging
                self.logger.debug(
                    f"Valid message from {sender_type}({identity_str}): {elem}"
                )

                # Handle heartbeat messages
                if elem == RouterConstants.HEARTBEAT:
                    try:
                        await self.client_registry.register_client(
                            identity_str, sender_type
                        )
                        self.logger.debug(
                            f"Heartbeat processed for {sender_type}({identity_str})"
                        )
                    except ClientRegistrationError as e:
                        self.logger.error(f"Heartbeat registration failed: {e}")
                    continue

                # Log important commands
                if elem in [
                    RouterConstants.START_SIMULATION,
                    RouterConstants.STOP_SIMULATION,
                    RouterConstants.PAUSE_SIMULATION,
                    RouterConstants.RESUME_SIMULATION,
                    RouterConstants.RESET_SIMULATION,
                ]:
                    self.logger.info(
                        f"Command {elem} from {sender_type}/{identity_str}"
                    )

                # Route message using enhanced router
                try:
                    await self.message_router.route_message(
                        sender_id=identity_str,
                        sender_type=sender_type,
                        elem=elem,
                        data=data,
                        socket=self.socket,
                    )
                except RoutingError as e:
                    self.logger.error(f"Message routing failed: {e}")
                    # Send error response to sender
                    error_msg = {RouterConstants.ERROR: f"Routing failed: {str(e)}"}
                    await self.socket.send_multipart(
                        [identity, zmq.utils.jsonapi.dumps(error_msg)]
                    )

            except asyncio.CancelledError:
                self.logger.info("Router shutting down...")
                break
            except KeyboardInterrupt:
                self.logger.info("Router shutdown requested")
                break
            except zmq.ZMQError as e:
                self.logger.error(f"ZMQ error in router: {e}")
                await asyncio.sleep(0.1)
                continue
            except Exception as e:
                self.logger.error(f"Unexpected router error: {e}", exc_info=True)
                await asyncio.sleep(0.1)
                continue

    def _log_malformed_message(
        self, message: Dict[str, Any], error: str, client_identity: str = "unknown"
    ) -> None:
        """
        Log detailed information about malformed messages for debugging.

        Args:
            message: The malformed message
            error: Specific error description
            client_identity: Identity of the client that sent the message
        """
        # Log the main error
        self.logger.error(f"Malformed message from client {client_identity}: {error}")

        # Get detailed validation error information
        detailed_error = self.validator.get_validation_error_details(message)
        self.logger.error(f"Validation details: {detailed_error}")

        # Log expected vs actual format
        expected_format = {
            RouterConstants.SENDER: "string (HydraClient|HydraServer|HydraRouter)",
            RouterConstants.ELEM: "string (message type)",
            RouterConstants.DATA: "dict (optional)",
            RouterConstants.CLIENT_ID: "string (optional)",
            RouterConstants.TIMESTAMP: "float (optional)",
            RouterConstants.REQUEST_ID: "string (optional)",
        }

        self.logger.error(f"Expected message format: {expected_format}")

        # Log actual message content (truncated for safety)
        actual_message = str(message)
        if len(actual_message) > 500:
            actual_message = actual_message[:500] + "... (truncated)"

        self.logger.error(f"Actual message received: {actual_message}")

        # Log debugging hints
        self.logger.error(
            "Debugging hints: "
            "1. Ensure client sends RouterConstants format with 'sender' and 'elem' fields. "
            "2. Check MQClient format conversion is working correctly. "
            "3. Verify message serialization/deserialization process."
        )

    def _log_frame_error(
        self, frames: List[bytes], client_identity: str = "unknown"
    ) -> None:
        """
        Log detailed information about malformed ZMQ frames.

        Args:
            frames: The received ZMQ frames
            client_identity: Identity of the client that sent the frames
        """
        self.logger.error(
            f"Malformed ZMQ frames from client {client_identity}: incorrect frame count"
        )

        # Log expected vs actual frame structure
        self.logger.error(
            "Expected frame structure: [identity_frame, message_frame] (2 frames total)"
        )
        self.logger.error(f"Actual frame count: {len(frames)}")

        # Log frame details (truncated for safety)
        for i, frame in enumerate(frames):
            frame_str = str(frame)
            if len(frame_str) > 200:
                frame_str = frame_str[:200] + "... (truncated)"
            self.logger.error(f"Frame {i}: {frame_str}")

        # Log debugging hints
        self.logger.error(
            "Debugging hints: "
            "1. Check ZMQ socket type compatibility (ROUTER expects identity + message frames). "
            "2. Verify client is using correct ZMQ socket type (DEALER or REQ). "
            "3. Ensure client is not sending multipart messages incorrectly."
        )

    def _log_json_parse_error(
        self, msg_bytes: bytes, error: Exception, client_identity: str = "unknown"
    ) -> None:
        """
        Log detailed information about JSON parsing errors.

        Args:
            msg_bytes: The raw message bytes that failed to parse
            error: The JSON parsing exception
            client_identity: Identity of the client that sent the message
        """
        self.logger.error(f"JSON parsing error from client {client_identity}: {error}")

        # Log expected vs actual format
        self.logger.error(
            "Expected: Valid JSON string representing a message dictionary"
        )

        # Log actual message content (truncated and safely decoded)
        try:
            msg_str = msg_bytes.decode("utf-8", errors="replace")
        except Exception:
            msg_str = str(msg_bytes)

        if len(msg_str) > 300:
            msg_str = msg_str[:300] + "... (truncated)"

        self.logger.error(f"Actual message bytes: {msg_str}")
        self.logger.error(f"Message length: {len(msg_bytes)} bytes")

        # Log debugging hints
        self.logger.error(
            "Debugging hints: "
            "1. Verify client is sending valid JSON format. "
            "2. Check for encoding issues (should be UTF-8). "
            "3. Ensure message is properly serialized before sending. "
            "4. Check for truncated or corrupted messages."
        )

    async def shutdown(self) -> None:
        """Gracefully shutdown the router and all components."""
        self.logger.info("Shutting down Enhanced Hydra Router...")

        self.is_running = False

        # Stop background tasks
        await self.task_manager.stop_tasks()

        # Close socket
        if self.socket:
            self.socket.close(linger=0)

        # Terminate context
        if self.ctx:
            self.ctx.term()

        self.logger.info("Enhanced Hydra Router shutdown complete")

    # Legacy methods for backward compatibility
    def _validate_message_format(self, message: Dict[str, Any]) -> tuple[bool, str]:
        """Legacy validation method for backward compatibility."""
        return self.validator.validate_router_message(message)

    async def start_background_tasks(self) -> None:
        """Legacy method for backward compatibility."""
        await self.task_manager.start_tasks()

    async def broadcast_to_clients(self, elem: str, data: Any, sender_id: str) -> None:
        """Legacy method for backward compatibility."""
        await self.message_router._route_server_message(
            sender_id, elem, data, self.socket
        )

    async def forward_to_server(self, elem: str, data: Any, sender: bytes) -> None:
        """Legacy method for backward compatibility."""
        sender_id = sender.decode()
        await self.message_router._route_client_message(
            sender_id, elem, data, self.socket
        )

    async def prune_dead_clients_bg(self) -> None:
        """Legacy method - now handled by BackgroundTaskManager."""
        pass

    async def prune_dead_clients(self) -> None:
        """Legacy method - now handled by BackgroundTaskManager."""
        timeout = RouterConstants.HEARTBEAT_INTERVAL * 3
        removed_clients = await self.client_registry.prune_inactive_clients(timeout)
        if removed_clients:
            self.logger.info(f"Pruned inactive clients: {removed_clients}")


async def main_async(router_address: str, router_port: int, log_level: str) -> None:
    """Main async function for the enhanced router."""
    router = HydraRouter(
        router_address=router_address, router_port=router_port, log_level=log_level
    )

    try:
        # Start the enhanced router
        await router.start()

        # Start handling requests
        await router.handle_requests()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.getLogger("HydraRouter").error(f"Router failed: {e}", exc_info=True)
        raise
    finally:
        await router.shutdown()


def main():
    """Main entry point for the AI Hydra router."""
    parser = argparse.ArgumentParser(
        description="AI Hydra Router - Routes messages between clients and servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start router on default port
  ai-hydra-router
  
  # Start router on custom port with debug logging
  ai-hydra-router --port 6666 --log-level DEBUG
  
  # Start router bound to specific interface
  ai-hydra-router --address 192.168.1.100 --port 5556
        """,
    )

    parser.add_argument(
        "-a",
        "--address",
        default="0.0.0.0",
        help="IP address to bind router to (default: 0.0.0.0)",
    )

    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=5556,
        help="Port to bind router to (default: 5556)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print(f"🚀 Starting AI Hydra Router on {args.address}:{args.port}")
    print(f"📊 Log level: {args.log_level}")
    print("📡 Press Ctrl+C to stop")

    try:
        asyncio.run(main_async(args.address, args.port, args.log_level))
    except KeyboardInterrupt:
        print("\n🛑 Router stopped by user")
    except Exception as e:
        print(f"❌ Router failed: {e}")
        raise


if __name__ == "__main__":
    main()
