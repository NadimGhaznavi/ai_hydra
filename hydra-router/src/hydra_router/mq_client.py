"""
Hydra Router MQ Client

Generic ZeroMQ client for connecting to the Hydra Router.
Provides message format conversion and connection management.
"""

import asyncio
import logging
import random
import time
import uuid
from typing import Optional, Dict, Any, Tuple

import zmq
import zmq.asyncio

from .zmq_protocol import ZMQMessage, MessageType
from .router_constants import RouterConstants
from .exceptions import ConnectionError, MessageFormatError


class MQClient:
    """Generic ZeroMQ client for Hydra Router communication."""

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
        MessageType.GAME_OVER.value: RouterConstants.GAME_STATE_UPDATE,  # Map GAME_OVER to GAME_STATE_UPDATE
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
            router_address: Address of the Hydra Router
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
            MessageFormatError: If message type is not supported or validation fails
            TypeError: If message has incorrect type
        """
        try:
            # Validate input message type
            if not isinstance(message, ZMQMessage):
                raise TypeError(f"Expected ZMQMessage, got {type(message).__name__}")

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
                raise MessageFormatError(
                    f"Converted message validation failed: {error}",
                    source_format="ZMQMessage",
                    target_format="RouterConstants",
                    conversion_stage="validation",
                )

            self.logger.debug(
                f"Successfully converted {message.message_type.value} to RouterConstants format"
            )
            return router_message

        except MessageFormatError:
            # Re-raise MessageFormatError as-is
            raise
        except Exception as e:
            # Handle unexpected errors gracefully
            self.logger.error(
                f"Unexpected error during message conversion: {e}", exc_info=True
            )
            raise MessageFormatError(
                f"Message conversion failed due to unexpected error: {e}",
                source_format="ZMQMessage",
                target_format="RouterConstants",
                conversion_stage="conversion",
            )

    def _convert_from_router_format(self, router_message: Dict[str, Any]) -> ZMQMessage:
        """
        Convert RouterConstants format to ZMQMessage for internal use.

        Args:
            router_message: RouterConstants format message

        Returns:
            ZMQMessage object

        Raises:
            MessageFormatError: If router message format is invalid
            TypeError: If router_message has incorrect type
        """
        try:
            # Validate input type
            if not isinstance(router_message, dict):
                raise TypeError(f"Expected dict, got {type(router_message).__name__}")

            # Validate router message format
            is_valid, error = self._validate_router_message(router_message)
            if not is_valid:
                raise MessageFormatError(
                    f"Invalid router message format: {error}",
                    source_format="RouterConstants",
                    target_format="ZMQMessage",
                    conversion_stage="validation",
                )

            # Map elem back to message_type
            elem = router_message.get(RouterConstants.ELEM)
            message_type = self._map_elem_to_message_type(elem)

            zmq_message = ZMQMessage(
                message_type=message_type,
                timestamp=router_message.get(RouterConstants.TIMESTAMP, time.time()),
                client_id=router_message.get(RouterConstants.CLIENT_ID),
                request_id=router_message.get(RouterConstants.REQUEST_ID),
                data=router_message.get(RouterConstants.DATA, {}),
            )

            self.logger.debug(
                f"Successfully converted RouterConstants {elem} to ZMQMessage format"
            )
            return zmq_message

        except MessageFormatError:
            # Re-raise MessageFormatError as-is
            raise
        except Exception as e:
            # Handle unexpected errors gracefully
            self.logger.error(
                f"Unexpected error during router message conversion: {e}", exc_info=True
            )
            raise MessageFormatError(
                f"Router message conversion failed due to unexpected error: {e}",
                source_format="RouterConstants",
                target_format="ZMQMessage",
                conversion_stage="conversion",
            )

    def _map_message_type_to_elem(self, message_type: str) -> str:
        """
        Map ZMQMessage type to RouterConstants element.

        Args:
            message_type: MessageType enum value

        Returns:
            RouterConstants element string

        Raises:
            MessageFormatError: If message type is not supported
        """
        if not isinstance(message_type, str):
            raise MessageFormatError(
                f"Message type must be string, got {type(message_type).__name__}",
                conversion_stage="type_mapping",
            )

        if message_type not in self.MESSAGE_TYPE_MAPPING:
            supported_types = list(self.MESSAGE_TYPE_MAPPING.keys())
            raise MessageFormatError(
                f"Unsupported message type: '{message_type}'. "
                f"Supported types: {supported_types}",
                conversion_stage="type_mapping",
            )

        return self.MESSAGE_TYPE_MAPPING[message_type]

    def _map_elem_to_message_type(self, elem: str) -> MessageType:
        """
        Map RouterConstants element to MessageType enum.

        Args:
            elem: RouterConstants element string

        Returns:
            MessageType enum value

        Raises:
            MessageFormatError: If elem is not supported
        """
        if not isinstance(elem, str):
            raise MessageFormatError(
                f"RouterConstants elem must be string, got {type(elem).__name__}",
                conversion_stage="type_mapping",
            )

        # Create reverse mapping
        reverse_mapping = {v: k for k, v in self.MESSAGE_TYPE_MAPPING.items()}

        if elem not in reverse_mapping:
            supported_elems = list(reverse_mapping.keys())
            raise MessageFormatError(
                f"Unsupported router element: '{elem}'. "
                f"Supported elements: {supported_elems}",
                conversion_stage="type_mapping",
            )

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
        if not isinstance(message, dict):
            return False, f"Message must be dict, got {type(message).__name__}"

        required_fields = [
            RouterConstants.SENDER,
            RouterConstants.ELEM,
            RouterConstants.DATA,
            RouterConstants.CLIENT_ID,
            RouterConstants.TIMESTAMP,
        ]

        # Check for missing required fields
        missing_fields = []
        for field in required_fields:
            if field not in message:
                missing_fields.append(field)

        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"

        # Validate field types with detailed error messages
        type_validations = [
            (RouterConstants.SENDER, str, "sender must be string"),
            (RouterConstants.ELEM, str, "elem must be string"),
            (RouterConstants.DATA, dict, "data must be dict"),
            (RouterConstants.CLIENT_ID, str, "client_id must be string"),
            (RouterConstants.TIMESTAMP, (int, float), "timestamp must be number"),
        ]

        for field, expected_type, error_msg in type_validations:
            if not isinstance(message[field], expected_type):
                actual_type = type(message[field]).__name__
                return False, f"{error_msg}, got {actual_type}"

        # Validate field values
        if not message[RouterConstants.SENDER]:
            return False, "sender field cannot be empty"

        if not message[RouterConstants.ELEM]:
            return False, "elem field cannot be empty"

        if not message[RouterConstants.CLIENT_ID]:
            return False, "client_id field cannot be empty"

        # Validate timestamp is reasonable (not negative, reasonable range)
        timestamp = message[RouterConstants.TIMESTAMP]
        if timestamp < 0:
            return False, "timestamp cannot be negative"
        # Allow reasonable timestamp range (1970 to 2100)
        if timestamp > 4102444800:  # Year 2100
            return False, "timestamp is unreasonably far in the future"

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

        Raises:
            ConnectionError: If not connected to router
            MessageFormatError: If message format conversion fails
            TypeError: If message has incorrect type
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to router")

        if not isinstance(message, ZMQMessage):
            raise TypeError(f"Expected ZMQMessage, got {type(message).__name__}")

        try:
            # Convert ZMQMessage to RouterConstants format
            router_message = self._convert_to_router_format(message)

            await self.socket.send_json(router_message)
            self.logger.debug(f"Sent message: {router_message[RouterConstants.ELEM]}")

        except MessageFormatError as e:
            # Log conversion errors with context
            self.logger.error(
                f"Failed to send message due to format conversion error: {e}. "
                f"Message type: {message.message_type.value}, Client: {self.client_id}"
            )
            raise
        except TypeError as e:
            # Log type errors with context
            self.logger.error(
                f"Failed to send message due to type error: {e}. "
                f"Client: {self.client_id}"
            )
            raise
        except Exception as e:
            # Handle unexpected errors gracefully
            self.logger.error(
                f"Unexpected error sending message: {e}. "
                f"Message type: {message.message_type.value}, Client: {self.client_id}",
                exc_info=True,
            )
            raise ConnectionError(
                f"Failed to send message due to unexpected error: {e}"
            )

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

                # Validate received message format
                if isinstance(message_dict, dict):
                    # Check if it's a RouterConstants format message
                    if RouterConstants.ELEM in message_dict:
                        try:
                            # Validate the router message format
                            is_valid, error = self._validate_router_message(
                                message_dict
                            )
                            if not is_valid:
                                self.logger.warning(
                                    f"Received invalid RouterConstants message: {error}. "
                                    f"Message: {message_dict}"
                                )
                                return None

                            self.logger.debug(
                                f"Received RouterConstants message: {message_dict.get(RouterConstants.ELEM, 'unknown')}"
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Error validating received RouterConstants message: {e}. "
                                f"Message: {message_dict}"
                            )
                            return None
                    else:
                        # Legacy format or other message type
                        self.logger.debug(
                            f"Received message: {message_dict.get('message_type', 'unknown')}"
                        )
                else:
                    self.logger.warning(
                        f"Received non-dict message: {type(message_dict).__name__}"
                    )
                    return None

                return message_dict
            return None

        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}", exc_info=True)
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

                # Validate received message format
                if isinstance(message_dict, dict):
                    # Check if it's a RouterConstants format message
                    if RouterConstants.ELEM in message_dict:
                        try:
                            # Validate the router message format
                            is_valid, error = self._validate_router_message(
                                message_dict
                            )
                            if not is_valid:
                                self.logger.warning(
                                    f"Received invalid RouterConstants message: {error}. "
                                    f"Message: {message_dict}"
                                )
                                return None

                            self.logger.debug(
                                f"Received RouterConstants message: {message_dict.get(RouterConstants.ELEM, 'unknown')}"
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Error validating received RouterConstants message: {e}. "
                                f"Message: {message_dict}"
                            )
                            return None
                    else:
                        # Legacy format or other message type
                        self.logger.debug(
                            f"Received message: {message_dict.get('message_type', 'unknown')}"
                        )
                else:
                    self.logger.warning(
                        f"Received non-dict message: {type(message_dict).__name__}"
                    )
                    return None

                return message_dict
            return None

        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}", exc_info=True)
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

        Raises:
            ConnectionError: If not connected to router
            MessageFormatError: If message format is invalid
            TypeError: If parameters have incorrect types
        """
        if data is None:
            data = {}

        if not isinstance(message_type, MessageType):
            raise TypeError(f"Expected MessageType, got {type(message_type).__name__}")

        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for data, got {type(data).__name__}")

        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"Timeout must be positive number, got {timeout}")

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
                self.logger.debug(f"Received response for {message_type.value}")
                return response
            else:
                self.logger.warning(
                    f"No matching response received for {message_type.value} "
                    f"(request_id: {request_id}) within {timeout}s timeout"
                )
                return None

        except (MessageFormatError, TypeError) as e:
            # Re-raise validation errors
            self.logger.error(f"Command failed due to validation error: {e}")
            raise
        except ConnectionError as e:
            # Re-raise connection errors
            self.logger.error(f"Command failed due to connection error: {e}")
            raise
        except Exception as e:
            # Handle unexpected errors gracefully
            self.logger.error(
                f"Unexpected error in send_command: {e}. "
                f"Message type: {message_type.value}, Client: {self.client_id}",
                exc_info=True,
            )
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

                # Validate heartbeat message before sending
                is_valid, error = self._validate_router_message(heartbeat_msg)
                if not is_valid:
                    self.logger.error(f"Invalid heartbeat message: {error}")
                    # Try to continue with next heartbeat
                    continue

                await self.socket.send_json(heartbeat_msg)
                self.logger.debug(f"Sent heartbeat from {self.client_id}")

            except Exception as e:
                self.logger.error(
                    f"Heartbeat failed for client {self.client_id}: {e}", exc_info=True
                )
                # Continue trying to send heartbeats even if one fails

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
