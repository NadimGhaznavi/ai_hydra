"""
AI Hydra Router

Pure MQ router between TUI clients and the AI Hydra server.
Based on the ai_snake_lab SimRouter pattern.
"""

import asyncio
import logging
import time
import argparse
from copy import deepcopy
from typing import Dict, Any, List

import zmq
import zmq.asyncio

from ai_hydra.router_constants import RouterConstants, RouterLabels


class HydraRouter:
    """Pure MQ router between TUI clients and the AI Hydra server."""

    def __init__(
        self,
        router_address: str = "0.0.0.0",
        router_port: int = 5556,
        log_level: str = "INFO",
    ):
        """
        Initialize the AI Hydra router.

        Args:
            router_address: Address to bind the router to
            router_port: Port to bind the router to
            log_level: Logging level
        """
        # Setup logging
        self.logger = logging.getLogger("HydraRouter")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Initialize ZMQ context
        self.ctx = zmq.asyncio.Context()

        # Create a ROUTER socket to manage multiple clients
        self.socket = self.ctx.socket(zmq.ROUTER)

        # Bind to the router service
        router_service = f"tcp://{router_address}:{router_port}"
        try:
            self.socket.bind(router_service)
            self.logger.info(f"Router started on {router_service}")
        except zmq.error.ZMQError as e:
            self.logger.critical(f"Failed to bind router to {router_service}: {e}")
            raise

        # Client tracking
        self.clients: Dict[str, tuple] = (
            {}
        )  # client_id -> (client_type, last_heartbeat)
        self.client_count = 0
        self.server_count = 0

        # Lock for concurrent client dictionary access
        self.clients_lock = asyncio.Lock()

        # Background task reference (will be started in start_background_tasks)
        self.prune_task = None

    def _validate_message_format(self, message: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate message format compliance with RouterConstants format.

        Args:
            message: The message dictionary to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(message, dict):
            return False, f"Message must be a dictionary, got {type(message).__name__}"

        # Check required fields
        required_fields = [RouterConstants.SENDER, RouterConstants.ELEM]
        missing_fields = []

        for field in required_fields:
            if field not in message:
                missing_fields.append(field)

        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"

        # Validate field types and values
        sender = message.get(RouterConstants.SENDER)
        elem = message.get(RouterConstants.ELEM)

        if not isinstance(sender, str) or not sender.strip():
            return (
                False,
                f"Field '{RouterConstants.SENDER}' must be a non-empty string, got: {repr(sender)}",
            )

        if not isinstance(elem, str) or not elem.strip():
            return (
                False,
                f"Field '{RouterConstants.ELEM}' must be a non-empty string, got: {repr(elem)}",
            )

        # Validate sender type
        valid_senders = [
            RouterConstants.HYDRA_CLIENT,
            RouterConstants.HYDRA_SERVER,
            RouterConstants.HYDRA_ROUTER,
        ]
        if sender not in valid_senders:
            return (
                False,
                f"Invalid sender type '{sender}', expected one of: {', '.join(valid_senders)}",
            )

        # Validate data field if present
        if RouterConstants.DATA in message:
            data = message[RouterConstants.DATA]
            if not isinstance(data, (dict, type(None))):
                return (
                    False,
                    f"Field '{RouterConstants.DATA}' must be a dictionary or None, got {type(data).__name__}",
                )

        return True, ""

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

        # Log message analysis
        if isinstance(message, dict):
            present_fields = list(message.keys())
            self.logger.error(f"Present fields: {present_fields}")

            # Check for common issues
            if (
                RouterConstants.MESSAGE_TYPE in message
                and RouterConstants.ELEM not in message
            ):
                self.logger.error(
                    f"Detected ZMQMessage format: found '{RouterConstants.MESSAGE_TYPE}' field "
                    f"but missing '{RouterConstants.ELEM}' field. "
                    f"This suggests the client is sending ZMQMessage format instead of RouterConstants format."
                )

            # Log field types for debugging
            field_types = {k: type(v).__name__ for k, v in message.items()}
            self.logger.error(f"Field types: {field_types}")
        else:
            self.logger.error(
                f"Message is not a dictionary: type={type(message).__name__}"
            )

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

    def _log_unknown_sender_error(
        self, sender_type: str, elem: str, data: Any, client_identity: str = "unknown"
    ) -> None:
        """
        Log detailed information about unknown sender type errors.

        Args:
            sender_type: The unknown sender type received
            elem: The message element/type
            data: The message data
            client_identity: Identity of the client that sent the message
        """
        self.logger.error(
            f"Unknown sender type from client {client_identity}: '{sender_type}'"
        )

        # Log expected vs actual sender types
        valid_senders = [
            RouterConstants.HYDRA_CLIENT,
            RouterConstants.HYDRA_SERVER,
            RouterConstants.HYDRA_ROUTER,
        ]
        self.logger.error(f"Expected sender types: {', '.join(valid_senders)}")
        self.logger.error(
            f"Actual sender type: '{sender_type}' (type: {type(sender_type).__name__})"
        )

        # Log additional message context
        self.logger.error(f"Message element: '{elem}'")

        # Log data summary (truncated for safety)
        data_str = str(data)
        if len(data_str) > 200:
            data_str = data_str[:200] + "... (truncated)"
        self.logger.error(f"Message data: {data_str}")

        # Log debugging hints
        self.logger.error(
            "Debugging hints: "
            "1. Check client configuration for correct sender type. "
            "2. Verify MQClient is setting sender field correctly. "
            "3. Check for typos in sender type string. "
            f"4. Ensure sender is one of: {', '.join(valid_senders)}"
        )

    async def start_background_tasks(self) -> None:
        """Start background tasks like client pruning."""
        if self.prune_task is None:
            self.prune_task = asyncio.create_task(self.prune_dead_clients_bg())

    async def broadcast_to_clients(self, elem: str, data: Any, sender_id: str) -> None:
        """Broadcast messages from server to all connected clients."""
        client_ids = []
        clients = deepcopy(self.clients)

        for client_id in clients.keys():
            if clients[client_id][0] == RouterConstants.HYDRA_CLIENT:
                client_ids.append(client_id)

        # Nothing to do if no clients
        if not client_ids:
            return

        msg = {
            RouterConstants.SENDER: RouterConstants.HYDRA_SERVER,
            RouterConstants.ELEM: elem,
            RouterConstants.DATA: data,
        }
        msg_bytes = zmq.utils.jsonapi.dumps(msg)

        for client_id in client_ids:
            if client_id != sender_id:
                try:
                    await self.socket.send_multipart([client_id.encode(), msg_bytes])
                    self.logger.debug(
                        f"Broadcast message to client {client_id}: {elem}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to send message to client {client_id}: {e}"
                    )

    async def handle_requests(self) -> None:
        """Continuously route messages between clients and servers."""
        self.logger.info("Router message handling started")

        while True:
            try:
                # ROUTER sockets prepend an identity frame
                frames = await self.socket.recv_multipart()
                identity = frames[0]
                identity_str = identity.decode()
                msg_bytes = frames[1]

                if len(frames) != 2:
                    self._log_frame_error(frames, identity_str)
                    continue

                try:
                    msg = zmq.utils.jsonapi.loads(msg_bytes)
                except (ValueError, TypeError) as e:
                    self._log_json_parse_error(msg_bytes, e, identity_str)
                    continue

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
                self.logger.error(f"Router error: {e}")
                continue

            # Parse message
            sender_type = msg.get(RouterConstants.SENDER)
            elem = msg.get(RouterConstants.ELEM)
            data = msg.get(RouterConstants.DATA, {})

            # Debug logging
            self.logger.debug(f"Message from {sender_type}({identity_str}): {elem}")

            # Validate message format
            validation_result = self._validate_message_format(msg)
            if not validation_result[0]:
                self._log_malformed_message(msg, validation_result[1], identity_str)
                continue

            # Handle heartbeat messages
            if elem == RouterConstants.HEARTBEAT:
                async with self.clients_lock:
                    self.clients[identity_str] = (sender_type, time.time())
                self.logger.debug(f"Heartbeat from {sender_type}({identity_str})")
                continue

            # Log important commands
            if elem in [
                RouterConstants.START_SIMULATION,
                RouterConstants.STOP_SIMULATION,
                RouterConstants.PAUSE_SIMULATION,
                RouterConstants.RESUME_SIMULATION,
                RouterConstants.RESET_SIMULATION,
            ]:
                self.logger.info(f"Command {elem} from {sender_type}/{identity_str}")

            ### Routing Logic ###

            # Forward client commands to server
            if sender_type == RouterConstants.HYDRA_CLIENT:
                await self.forward_to_server(elem=elem, data=data, sender=identity)

            # Handle server messages
            elif sender_type == RouterConstants.HYDRA_SERVER:
                # Drop status/error messages (they're handled locally)
                if elem in [RouterConstants.STATUS, RouterConstants.ERROR]:
                    continue

                # Broadcast all other messages to clients
                await self.broadcast_to_clients(
                    elem=elem, data=data, sender_id=identity_str
                )

            else:
                self._log_unknown_sender_error(sender_type, elem, data, identity_str)

    async def forward_to_server(self, elem: str, data: Any, sender: bytes) -> None:
        """Forward client command to the AI Hydra server."""
        # Find all connected servers
        servers = []
        clients = deepcopy(self.clients)

        for identity in clients.keys():
            if clients[identity][0] == RouterConstants.HYDRA_SERVER:
                servers.append(identity)

        # No server connected - inform the client
        if not servers:
            error_msg = {RouterConstants.ERROR: "No AI Hydra server connected"}
            await self.socket.send_multipart(
                [sender, zmq.utils.jsonapi.dumps(error_msg)]
            )
            self.logger.warning("No server connected for client request")
            return

        # Construct message
        msg = {
            RouterConstants.SENDER: RouterConstants.HYDRA_CLIENT,
            RouterConstants.ELEM: elem,
            RouterConstants.DATA: data,
        }
        msg_bytes = zmq.utils.jsonapi.dumps(msg)

        # Send to all connected servers (usually just one)
        for server_id in servers:
            try:
                await self.socket.send_multipart([server_id.encode(), msg_bytes])
                self.logger.debug(f"Forwarded {elem} to server {server_id}")
            except Exception as e:
                self.logger.error(
                    f"Failed to forward message to server {server_id}: {e}"
                )

        # Acknowledge the client
        ack_msg = {RouterConstants.STATUS: RouterConstants.OK}
        await self.socket.send_multipart([sender, zmq.utils.jsonapi.dumps(ack_msg)])

    async def prune_dead_clients_bg(self) -> None:
        """Background task to prune dead clients."""
        while True:
            await self.prune_dead_clients()
            await asyncio.sleep(RouterConstants.HEARTBEAT_INTERVAL * 4)

    async def prune_dead_clients(self) -> None:
        """Remove clients that haven't sent heartbeats recently."""
        async with self.clients_lock:
            now = time.time()
            client_count = 0
            server_count = 0

            clients_copy = deepcopy(self.clients)
            for identity in clients_copy.keys():
                sender_type, last_heartbeat = self.clients[identity]

                # Remove clients that haven't sent heartbeat in 3x the interval
                if now - last_heartbeat > (RouterConstants.HEARTBEAT_INTERVAL * 3):
                    self.logger.info(f"Removing inactive client: {identity}")
                    del self.clients[identity]
                else:
                    if sender_type == RouterConstants.HYDRA_SERVER:
                        server_count += 1
                    elif sender_type == RouterConstants.HYDRA_CLIENT:
                        client_count += 1

            # Update counts if changed
            if client_count != self.client_count or server_count != self.server_count:
                self.client_count = client_count
                self.server_count = server_count
                self.logger.info(
                    f"Connected clients: {client_count}, servers: {server_count}"
                )

    async def shutdown(self) -> None:
        """Gracefully shutdown the router."""
        self.logger.info("Shutting down router...")

        # Cancel background tasks
        if self.prune_task:
            self.prune_task.cancel()
            try:
                await self.prune_task
            except asyncio.CancelledError:
                pass

        # Close socket
        if self.socket:
            self.socket.close(linger=0)

        # Terminate context
        if self.ctx:
            self.ctx.term()

        self.logger.info("Router shutdown complete")


async def main_async(router_address: str, router_port: int, log_level: str) -> None:
    """Main async function for the router."""
    router = HydraRouter(
        router_address=router_address, router_port=router_port, log_level=log_level
    )

    try:
        # Start background tasks
        await router.start_background_tasks()

        # Start handling requests
        await router.handle_requests()
    except KeyboardInterrupt:
        pass
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
