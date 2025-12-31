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
from typing import Optional, Dict, Any

import zmq
import zmq.asyncio

from ai_hydra.zmq_protocol import ZMQMessage, MessageType


class MQClient:
    """Generic ZeroMQ client for AI Hydra router communication."""
    
    def __init__(
        self, 
        router_address: str = "tcp://localhost:5556",
        client_type: str = "HydraClient",
        heartbeat_interval: float = 5.0,
        client_id: Optional[str] = None
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
            # Add client type to message data for routing
            message_dict = {
                "sender": self.client_type,
                "client_id": self.client_id,
                "message_type": message.message_type.value,
                "timestamp": message.timestamp,
                "request_id": message.request_id,
                "data": message.data
            }
            
            await self.socket.send_json(message_dict)
            self.logger.debug(f"Sent message: {message.message_type.value}")
            
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
                self.logger.debug(f"Received message: {message_dict.get('message_type', 'unknown')}")
                return message_dict
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            return None
    
    async def receive_message_blocking(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
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
                self.logger.debug(f"Received message: {message_dict.get('message_type', 'unknown')}")
                return message_dict
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            return None
    
    async def send_command(
        self, 
        message_type: MessageType, 
        data: Dict[str, Any] = None,
        timeout: float = 10.0
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
            data=data
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
                    "sender": self.client_type,
                    "client_id": self.client_id,
                    "message_type": "HEARTBEAT",
                    "timestamp": time.time(),
                    "request_id": str(uuid.uuid4()),
                    "data": {}
                }
                
                await self.socket.send_json(heartbeat_msg)
                self.logger.debug(f"Sent heartbeat from {self.client_id}")
                
            except Exception as e:
                self.logger.error(f"Heartbeat failed: {e}")
            
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(), 
                    timeout=self.heartbeat_interval
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