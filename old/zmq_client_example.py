"""
Example ZeroMQ client for the AI Hydra.

This module provides an example client that demonstrates how to interact
with the headless AI agent via ZeroMQ messages.
"""

import asyncio
import json
import uuid
import time
from typing import Optional

import zmq
import zmq.asyncio

from .zmq_protocol import ZMQMessage, MessageType


class ZMQClient:
    """
    Example ZeroMQ client for interacting with the AI Hydra.
    
    This client demonstrates the message protocol and can be used as a
    reference for building TUI clients or other monitoring tools.
    """
    
    def __init__(self, server_address: str = "tcp://localhost:5555"):
        """
        Initialize the ZeroMQ client.
        
        Args:
            server_address: Address of the ZeroMQ server
        """
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        
        # ZeroMQ context and socket
        self.context = zmq.asyncio.Context()
        self.socket = None
        
        # Client state
        self.is_connected = False
        self.request_counter = 0
    
    async def connect(self) -> bool:
        """
        Connect to the ZeroMQ server.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(self.server_address)
            self.is_connected = True
            print(f"Connected to server at {self.server_address}")
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the ZeroMQ server."""
        if self.socket:
            self.socket.close()
        self.context.term()
        self.is_connected = False
        print("Disconnected from server")
    
    async def send_message(self, message: ZMQMessage) -> Optional[ZMQMessage]:
        """
        Send a message to the server and wait for response.
        
        Args:
            message: Message to send
            
        Returns:
            ZMQMessage: Response from server, or None if error
        """
        if not self.is_connected:
            print("Not connected to server")
            return None
        
        try:
            # Send message
            await self.socket.send_string(message.to_json())
            
            # Wait for response with timeout
            if await self.socket.poll(timeout=10000):  # 10 second timeout
                response_data = await self.socket.recv_string()
                return ZMQMessage.from_json(response_data)
            else:
                print("Timeout waiting for server response")
                return None
                
        except Exception as e:
            print(f"Error sending message: {e}")
            return None
    
    def _create_request_id(self) -> str:
        """Create a unique request ID."""
        self.request_counter += 1
        return f"{self.client_id}_{self.request_counter}"
    
    async def start_simulation(self, config: dict) -> Optional[ZMQMessage]:
        """
        Start a simulation with the given configuration.
        
        Args:
            config: Simulation configuration dictionary
            
        Returns:
            ZMQMessage: Response from server
        """
        message = ZMQMessage.create_command(
            MessageType.START_SIMULATION,
            self.client_id,
            self._create_request_id(),
            {"config": config}
        )
        
        response = await self.send_message(message)
        if response:
            print(f"Start simulation response: {response.message_type.value}")
            if response.data:
                print(f"Response data: {json.dumps(response.data, indent=2)}")
        
        return response
    
    async def stop_simulation(self) -> Optional[ZMQMessage]:
        """Stop the current simulation."""
        message = ZMQMessage.create_command(
            MessageType.STOP_SIMULATION,
            self.client_id,
            self._create_request_id()
        )
        
        response = await self.send_message(message)
        if response:
            print(f"Stop simulation response: {response.message_type.value}")
        
        return response
    
    async def get_status(self) -> Optional[ZMQMessage]:
        """Get current server and simulation status."""
        message = ZMQMessage.create_command(
            MessageType.GET_STATUS,
            self.client_id,
            self._create_request_id()
        )
        
        response = await self.send_message(message)
        if response:
            print(f"Status response: {response.message_type.value}")
            if response.data:
                # Pretty print status data
                status = response.data
                print(f"Server ID: {status.get('server_id', 'Unknown')}")
                print(f"Simulation State: {status.get('simulation_state', 'Unknown')}")
                print(f"Uptime: {status.get('uptime_seconds', 0):.1f} seconds")
                print(f"Connected Clients: {status.get('connected_clients', 0)}")
                print(f"Messages Received: {status.get('messages_received', 0)}")
                print(f"Messages Sent: {status.get('messages_sent', 0)}")
                
                # Game state info
                game_state = status.get('game_state')
                if game_state:
                    print(f"Game Score: {game_state.get('score', 0)}")
                    print(f"Game Moves: {game_state.get('moves_count', 0)}")
                    print(f"Snake Length: {len(game_state.get('snake_body', []))}")
                    print(f"Game Over: {game_state.get('is_game_over', False)}")
                
                # Performance metrics
                performance = status.get('performance', {})
                if performance:
                    print(f"Decisions/sec: {performance.get('decisions_per_second', 0):.2f}")
                    print(f"Memory Usage: {performance.get('memory_usage_mb', 0):.1f} MB")
                    print(f"CPU Usage: {performance.get('cpu_usage_percent', 0):.1f}%")
        
        return response
    
    async def pause_simulation(self) -> Optional[ZMQMessage]:
        """Pause the current simulation."""
        message = ZMQMessage.create_command(
            MessageType.PAUSE_SIMULATION,
            self.client_id,
            self._create_request_id()
        )
        
        response = await self.send_message(message)
        if response:
            print(f"Pause simulation response: {response.message_type.value}")
        
        return response
    
    async def resume_simulation(self) -> Optional[ZMQMessage]:
        """Resume the paused simulation."""
        message = ZMQMessage.create_command(
            MessageType.RESUME_SIMULATION,
            self.client_id,
            self._create_request_id()
        )
        
        response = await self.send_message(message)
        if response:
            print(f"Resume simulation response: {response.message_type.value}")
        
        return response
    
    async def reset_simulation(self) -> Optional[ZMQMessage]:
        """Reset the simulation."""
        message = ZMQMessage.create_command(
            MessageType.RESET_SIMULATION,
            self.client_id,
            self._create_request_id()
        )
        
        response = await self.send_message(message)
        if response:
            print(f"Reset simulation response: {response.message_type.value}")
        
        return response


async def demo_client():
    """Demonstrate the ZeroMQ client functionality."""
    client = ZMQClient()
    
    # Connect to server
    if not await client.connect():
        return
    
    try:
        print("\n=== ZeroMQ Client Demo ===")
        
        # Get initial status
        print("\n1. Getting initial server status...")
        await client.get_status()
        
        # Start simulation
        print("\n2. Starting simulation...")
        config = {
            "grid_size": [10, 10],
            "initial_snake_length": 3,
            "move_budget": 50,
            "random_seed": 42,
            "nn_enabled": True
        }
        await client.start_simulation(config)
        
        # Wait a bit and check status
        print("\n3. Waiting 3 seconds...")
        await asyncio.sleep(3)
        
        print("\n4. Getting simulation status...")
        await client.get_status()
        
        # Pause simulation
        print("\n5. Pausing simulation...")
        await client.pause_simulation()
        
        # Wait and resume
        print("\n6. Waiting 2 seconds...")
        await asyncio.sleep(2)
        
        print("\n7. Resuming simulation...")
        await client.resume_simulation()
        
        # Wait a bit more
        print("\n8. Waiting 5 seconds...")
        await asyncio.sleep(5)
        
        # Final status check
        print("\n9. Final status check...")
        await client.get_status()
        
        # Stop simulation
        print("\n10. Stopping simulation...")
        await client.stop_simulation()
        
        print("\n=== Demo Complete ===")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        await client.disconnect()


async def interactive_client():
    """Interactive client for manual testing."""
    client = ZMQClient()
    
    if not await client.connect():
        return
    
    print("\n=== Interactive ZeroMQ Client ===")
    print("Commands:")
    print("  status - Get server status")
    print("  start - Start simulation")
    print("  stop - Stop simulation")
    print("  pause - Pause simulation")
    print("  resume - Resume simulation")
    print("  reset - Reset simulation")
    print("  quit - Exit client")
    print()
    
    try:
        while True:
            command = input("Enter command: ").strip().lower()
            
            if command == "quit":
                break
            elif command == "status":
                await client.get_status()
            elif command == "start":
                config = {
                    "grid_size": [8, 8],
                    "move_budget": 30,
                    "random_seed": int(time.time()) % 1000
                }
                await client.start_simulation(config)
            elif command == "stop":
                await client.stop_simulation()
            elif command == "pause":
                await client.pause_simulation()
            elif command == "resume":
                await client.resume_simulation()
            elif command == "reset":
                await client.reset_simulation()
            else:
                print(f"Unknown command: {command}")
            
            print()  # Add blank line for readability
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ZeroMQ Client for AI Hydra")
    parser.add_argument("--server", default="tcp://localhost:5555", help="Server address")
    parser.add_argument("--mode", choices=["demo", "interactive"], default="demo", 
                       help="Client mode")
    
    args = parser.parse_args()
    
    # Update client with server address
    if args.mode == "demo":
        asyncio.run(demo_client())
    else:
        asyncio.run(interactive_client())