"""
AI Hydra MQ Client

Generic ZeroMQ client for connecting to the AI Hydra router.
Based on the ai_snake_lab MQClient pattern.
"""

import asyncio
import logging
import random
import time
import uuid
from typing import Optional, Dict, Any, Tuple

import zmq
import zmq.asyncio

from ai_hydra.zmq_protocol import ZMQMessage, MessageType
from ai_hydra.router_constants import RouterConstants


class MQClient:
    """Generic ZeroMQ client for AI Hydra router communication."""

    # Message type to RouterConstants elem mapping
    MESSAGE_TYPE_MAPPING = {
        MessageType.HEARTBEAT.value: RouterConstants.HEARTBEAT,
        MessageType.START_SIMULATION.value: RouterConstants.START_SIMULATION,
        MessageType.STOP_SIMULATION.value: RouterConstants.STOP_SIMULATION,
        MessageType.PAUSE_SIMULATION.value: RouterConstants.PAUSE_SIMULATION,
        MessageType.RESUME_SIMULATION.value: RouterConstants.RESUME_SIMULATION,
        MessageType.RESET_SIMULATION.value: RouterConstants.RESET_SIMULATION,
        MessageType.GET_STATUS.value: RouterConstants.GET_STATUS,
        MessageType.UPDATE_CONFIG.value: RouterConstants.SET_CONFIG,
        MessageType.STATUS_RESPONSE.value: RouterConstants.STATUS_UPDATE,
        MessageType.SIMULATION_STARTED.value: RouterConstants.SIMULATION_STARTED,
        MessageType.SIMULATION_STOPPED.value: RouterConstants.SIMULATION_STOPPED,
        MessageType.SIMULATION_PAUSED.value: RouterConstants.SIMULATION_PAUSED,
        MessageType.SIMULATION_RESUMED.value: RouterConstants.SIMULATION_RESUMED,
        MessageType.SIMULATION_RESET.value: RouterConstants.SIMULATION_RESET,
        MessageType.STATUS_UPDATE.value: RouterConstants.STATUS_UPDATE,
        MessageType.GAME_STATE_UPDATE.value: RouterConstants.GAME_STATE_UPDATE,
        MessageType.DECISION_CYCLE_COMPLETE.value: RouterConstants.PERFORMANCE_UPDATE,
        MessageType.GAME_OVER.value: RouterConstants.STATUS_UPDATE,
        MessageType.ERROR_OCCURRED.value: RouterConstants.ERROR,
        MessageType.CLIENT_CONNECTED.value: RouterConstants.STATUS,
        MessageType.CLIENT_DISCONNECTED.value: RouterConstants.STATUS,
        MessageType.CONFIG_UPDATED.value: RouterConstants.CONFIG_UPDATE,
    }

    def __init__(
        self,
        router_address: str = "tcp://localhost:5556",
        client_type: str = "HydraClient",
        heartbeat_interval: float = 5.0,
        client_id: Optional[str] = None,
    ):
        """
        Initialize MQ client.

        Args:
            router_address: Address of the AI Hydra router
            client_type: Type of client (HydraClient, HydraServer, etc.)
            heartbeat_interval: Interval between heartbeat messages in seconds
            client_id: Optional custom client ID, auto-generated if None
        """
        self.router_address = router_address
        self.client_type = client_type
        self.heartbeat_interval = heartbeat_interval

        # Generate unique client ID
        if client_id:
            self.client_id = client_id
        else:
            random_suffix = random.randint(1000, 9999)
            self.client_id = f"{client_type}-{random_suffix}"

        # ZeroMQ setup
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, self.client_id.encode())

        # Connection state
        self.is_connected = False
        self.stop_event = asyncio.Event()
        self.heartbeat_task = None

        # Logging
        self.logger = logging.getLogger(f"MQClient.{self.client_id}")

    def _convert_to_router_format(self, message: ZMQMessage) -> Dict[str, Any]:
        """
        Convert ZMQMessage to RouterConstants format for router communication.

        Args:
            message: ZMQMessage to convert

        Returns:
            Dict containing RouterConstants format message

        Raises:
            ValueError: If message type is not supported
        """
        try:
            # Map message_type to elem
            elem = self._map_message_type_to_elem(message.message_type.value)

            router_message = {
                RouterConstants.SENDER: self.client_type,
                RouterConstants.ELEM: elem,
                RouterConstants.DATA: message.data or {},
                RouterConstants.CLIENT_ID: self.client_id,
                RouterConstants.TIMESTAMP: message.timestamp,
                RouterConstants.REQUEST_ID: message.request_id,
            }

            # Validate the converted message
            is_valid, error = self._validate_router_message(router_message)
            if not is_valid:
                raise ValueError(f"Converted message validation failed: {error}")

            return router_message

        except Exception as e:
            self.logger.error(f"Message conversion failed: {e}")
            raise

    def _convert_from_router_format(self, router_message: Dict[str, Any]) -> ZMQMessage:
        """
        Convert RouterConstants format to ZMQMessage for internal use.

        Args:
            router_message: RouterConstants format message

        Returns:
            ZMQMessage object

        Raises:
            ValueError: If router message format is invalid
        """
        try:
            # Validate router message format
            is_valid, error = self._validate_router_message(router_message)
            if not is_valid:
                raise ValueError(f"Invalid router message format: {error}")

            # Map elem back to message_type
            elem = router_message.get(RouterConstants.ELEM)
            message_type = self._map_elem_to_message_type(elem)

            return ZMQMessage(
                message_type=message_type,
                timestamp=router_message.get(RouterConstants.TIMESTAMP, time.time()),
                client_id=router_message.get(RouterConstants.CLIENT_ID),
                request_id=router_message.get(RouterConstants.REQUEST_ID),
                data=router_message.get(RouterConstants.DATA, {}),
            )

        except Exception as e:
            self.logger.error(f"Router message conversion failed: {e}")
            raise

    def _map_message_type_to_elem(self, message_type: str) -> str:
        """
        Map ZMQMessage type to RouterConstants element.

        Args:
            message_type: MessageType enum value

        Returns:
            RouterConstants element string

        Raises:
            ValueError: If message type is not supported
        """
        if message_type not in self.MESSAGE_TYPE_MAPPING:
            raise ValueError(f"Unsupported message type: {message_type}")

        return self.MESSAGE_TYPE_MAPPING[message_type]

    def _map_elem_to_message_type(self, elem: str) -> MessageType:
        """
        Map RouterConstants element to MessageType enum.

        Args:
            elem: RouterConstants element string

        Returns:
            MessageType enum value

        Raises:
            ValueError: If elem is not supported
        """
        # Create reverse mapping
        reverse_mapping = {v: k for k, v in self.MESSAGE_TYPE_MAPPING.items()}

        if elem not in reverse_mapping:
            raise ValueError(f"Unsupported router element: {elem}")

        return MessageType(reverse_mapping[elem])

    def _validate_router_message(
        self, message: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate RouterConstants format compliance.

        Args:
            message: Message to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = [
            RouterConstants.SENDER,
            RouterConstants.ELEM,
            RouterConstants.DATA,
            RouterConstants.CLIENT_ID,
            RouterConstants.TIMESTAMP,
        ]

        for field in required_fields:
            if field not in message:
                return False, f"Missing required field: {field}"

        # Validate field types
        if not isinstance(message[RouterConstants.SENDER], str):
            return False, f"Field {RouterConstants.SENDER} must be string"

        if not isinstance(message[RouterConstants.ELEM], str):
            return False, f"Field {RouterConstants.ELEM} must be string"

        if not isinstance(message[RouterConstants.DATA], dict):
            return False, f"Field {RouterConstants.DATA} must be dict"

        if not isinstance(message[RouterConstants.CLIENT_ID], str):
            return False, f"Field {RouterConstants.CLIENT_ID} must be string"

        if not isinstance(message[RouterConstants.TIMESTAMP], (int, float)):
            return False, f"Field {RouterConstants.TIMESTAMP} must be number"

        return True, None

    async def connect(self) -> bool:
        """
        Connect to the router and start heartbeat.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.socket.connect(self.router_address)
            self.is_connected = True

            # Start heartbeat task
            self.heartbeat_task = asyncio.create_task(self._send_heartbeat())

            self.logger.info(f"Connected to router at {self.router_address}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to router: {e}")
            self.is_connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from router and cleanup resources."""
        self.logger.info("Disconnecting from router...")

        # Stop heartbeat
        self.stop_event.set()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close socket
        if self.socket:
            try:
                self.socket.disconnect(self.router_address)
                self.socket.close(linger=0)
            except Exception as e:
                self.logger.warning(f"Error closing socket: {e}")

        # Terminate context
        if self.context:
            self.context.term()

        self.is_connected = False
        self.logger.info("Disconnected from router")

    async def send_message(self, message: ZMQMessage) -> None:
        """
        Send a message through the router.

        Args:
            message: ZMQMessage to send
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to router")

        try:
            # Convert ZMQMessage to RouterConstants format
            router_message = self._convert_to_router_format(message)

            await self.socket.send_json(router_message)
            self.logger.debug(f"Sent message: {router_message[RouterConstants.ELEM]}")

        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            raise

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """
        Receive a message from the router.

        Returns:
            Dict containing the message, or None if no message available
        """
        if not self.is_connected:
            return None

        try:
            # Non-blocking receive
            if await self.socket.poll(timeout=0):
                message_dict = await self.socket.recv_json()
                self.logger.debug(
                    f"Received message: {message_dict.get('message_type', 'unknown')}"
                )
                return message_dict
            return None

        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            return None

    async def receive_message_blocking(
        self, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Receive a message from the router (blocking).

        Args:
            timeout: Timeout in seconds, None for no timeout

        Returns:
            Dict containing the message, or None if timeout
        """
        if not self.is_connected:
            return None

        try:
            timeout_ms = int(timeout * 1000) if timeout else -1
            if await self.socket.poll(timeout=timeout_ms):
                message_dict = await self.socket.recv_json()
                self.logger.debug(
                    f"Received message: {message_dict.get('message_type', 'unknown')}"
                )
                return message_dict
            return None

        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            return None

    async def send_command(
        self,
        message_type: MessageType,
        data: Dict[str, Any] = None,
        timeout: float = 10.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a command and wait for response.

        Args:
            message_type: Type of message to send
            data: Message data
            timeout: Response timeout in seconds

        Returns:
            Response message dict, or None if timeout/error
        """
        if data is None:
            data = {}

        # Create message
        request_id = str(uuid.uuid4())
        message = ZMQMessage.create_command(
            message_type=message_type,
            client_id=self.client_id,
            request_id=request_id,
            data=data,
        )

        try:
            # Send message
            await self.send_message(message)

            # Wait for response
            response = await self.receive_message_blocking(timeout=timeout)

            if response and response.get("request_id") == request_id:
                return response
            else:
                self.logger.warning(f"No response received for {message_type.value}")
                return None

        except Exception as e:
            self.logger.error(f"Command failed: {e}")
            return None

    async def _send_heartbeat(self) -> None:
        """Send periodic heartbeat messages to router."""
        while not self.stop_event.is_set():
            try:
                heartbeat_msg = {
                    RouterConstants.SENDER: self.client_type,
                    RouterConstants.ELEM: RouterConstants.HEARTBEAT,
                    RouterConstants.DATA: {},
                    RouterConstants.CLIENT_ID: self.client_id,
                    RouterConstants.TIMESTAMP: time.time(),
                    RouterConstants.REQUEST_ID: str(uuid.uuid4()),
                }

                await self.socket.send_json(heartbeat_msg)
                self.logger.debug(f"Sent heartbeat from {self.client_id}")

            except Exception as e:
                self.logger.error(f"Heartbeat failed: {e}")

            try:
                await asyncio.wait_for(
                    self.stop_event.wait(), timeout=self.heartbeat_interval
                )
                break  # Stop event was set
            except asyncio.TimeoutError:
                continue  # Continue heartbeat loop

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.is_connected:
            # Run disconnect in event loop if available
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.disconnect())
                else:
                    loop.run_until_complete(self.disconnect())
            except RuntimeError:
                # No event loop, create one
                asyncio.run(self.disconnect())
