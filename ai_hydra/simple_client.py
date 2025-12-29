#!/usr/bin/env python3
"""
Simple AI Hydra MQ Client

A basic command-line client for testing ZeroMQ communication with the AI Hydra server.
No Textual dependencies - just pure ZMQ communication for debugging.
"""

import asyncio
import argparse
import json
import logging
import sys
import time
import uuid
from typing import Dict, Any, Optional

import zmq
import zmq.asyncio

from ai_hydra.zmq_protocol import ZMQMessage, MessageType


class SimpleHydraClient:
    """Simple command-line client for AI Hydra server communication testing."""
    
    def __init__(self, server_address: str = "tcp://localhost:5555"):
        self.server_address = server_address
        self.client_id = f"simple-client-{uuid.uuid4().hex[:8]}"
        
        # ZMQ setup
        self.context = None
        self.socket = None
        self.is_connected = False
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    async def connect(self) -> bool:
        """Connect to the AI Hydra server."""
        try:
            print(f"Connecting to AI Hydra server at {self.server_address}...")
            
            # Create async ZMQ context and socket
            self.context = zmq.asyncio.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(self.server_address)
            
            # Test connection with get_status
            print("Testing connection with GET_STATUS...")
            response = await self.send_command(MessageType.GET_STATUS, {})
            
            if response:
                self.is_connected = True
                print(f"✓ Connected successfully!")
                print(f"✓ Client ID: {self.client_id}")
                print(f"✓ Server response: {response.message_type.value}")
                return True
            else:
                print("✗ Failed to get response from server")
                return False
                
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            self.logger.error(f"Connection error: {e}")
            return False
    
    async def send_command(self, message_type: MessageType, data: Dict[str, Any]) -> Optional[ZMQMessage]:
        """Send a command to the server and wait for response."""
        if not self.socket:
            print("✗ No socket available")
            return None
            
        try:
            # Create AI Hydra protocol message
            request_id = str(uuid.uuid4())
            message = ZMQMessage.create_command(
                message_type=message_type,
                client_id=self.client_id,
                request_id=request_id,
                data=data
            )
            
            print(f"→ Sending: {message_type.value}")
            self.logger.debug(f"Message: {message.to_json()}")
            
            # Send message
            await self.socket.send_string(message.to_json())
            
            # Wait for response with timeout
            print("  Waiting for response...")
            timeout_ms = 10000 if message_type == MessageType.STOP_SIMULATION else 5000
            if await self.socket.poll(timeout=timeout_ms):
                response_data = await self.socket.recv_string()
                response = ZMQMessage.from_json(response_data)
                
                print(f"← Received: {response.message_type.value}")
                self.logger.debug(f"Response: {response_data}")
                
                return response
            else:
                print(f"✗ Timeout waiting for response to {message_type.value}")
                # Reset socket after timeout to avoid state issues
                await self._reset_socket()
                return None
                
        except Exception as e:
            print(f"✗ Error sending command {message_type.value}: {e}")
            self.logger.error(f"Send error: {e}")
            # Reset socket after error to avoid state issues
            await self._reset_socket()
            return None
    
    async def _reset_socket(self) -> None:
        """Reset the socket after timeout or error."""
        try:
            if self.socket:
                self.socket.close()
            
            # Create new socket
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(self.server_address)
            print("  Socket reset due to timeout/error")
            
        except Exception as e:
            print(f"✗ Error resetting socket: {e}")
            self.socket = None
    
    async def get_status(self) -> None:
        """Get server status."""
        print("\n=== GET STATUS ===")
        response = await self.send_command(MessageType.GET_STATUS, {})
        
        if response and response.data:
            print("Status received:")
            if "simulation_status" in response.data:
                print(f"  Simulation: {response.data['simulation_status']}")
            if "game_state" in response.data:
                game_state = response.data["game_state"]
                print(f"  Score: {game_state.get('score', 'N/A')}")
                print(f"  Moves: {game_state.get('moves_count', 'N/A')}")
            if "performance" in response.data:
                perf = response.data["performance"]
                print(f"  Decisions/sec: {perf.get('decisions_per_second', 'N/A')}")
        else:
            print("No status data received")
    
    async def start_simulation(self) -> None:
        """Start a simulation."""
        print("\n=== START SIMULATION ===")
        config = {
            "grid_size": [20, 20],
            "move_budget": 100,
            "initial_snake_length": 3,
            "random_seed": 42
        }
        
        response = await self.send_command(MessageType.START_SIMULATION, {"config": config})
        
        if response:
            if response.message_type == MessageType.SIMULATION_STARTED:
                print("✓ Simulation started successfully")
            elif response.message_type == MessageType.ERROR_OCCURRED:
                error_msg = response.data.get("error_message", "Unknown error")
                print(f"✗ Failed to start simulation: {error_msg}")
            else:
                print(f"? Unexpected response: {response.message_type.value}")
        else:
            print("✗ No response received")
    
    async def stop_simulation(self) -> None:
        """Stop the simulation."""
        print("\n=== STOP SIMULATION ===")
        response = await self.send_command(MessageType.STOP_SIMULATION, {})
        
        if response:
            if response.message_type == MessageType.SIMULATION_STOPPED:
                print("✓ Simulation stopped successfully")
            elif response.message_type == MessageType.ERROR_OCCURRED:
                error_msg = response.data.get("error_message", "Unknown error")
                print(f"✗ Failed to stop simulation: {error_msg}")
            else:
                print(f"? Unexpected response: {response.message_type.value}")
        else:
            print("✗ No response received")
    
    async def reset_simulation(self) -> None:
        """Reset the simulation."""
        print("\n=== RESET SIMULATION ===")
        response = await self.send_command(MessageType.RESET_SIMULATION, {})
        
        if response:
            if response.message_type == MessageType.SIMULATION_RESET:
                print("✓ Simulation reset successfully")
            elif response.message_type == MessageType.ERROR_OCCURRED:
                error_msg = response.data.get("error_message", "Unknown error")
                print(f"✗ Failed to reset simulation: {error_msg}")
            else:
                print(f"? Unexpected response: {response.message_type.value}")
        else:
            print("✗ No response received")
    
    async def interactive_mode(self) -> None:
        """Run interactive command mode."""
        print("\n=== INTERACTIVE MODE ===")
        print("Commands: status, start, stop, reset, quit")
        
        while True:
            try:
                command = input("\nhydra> ").strip().lower()
                
                if command == "quit" or command == "q":
                    break
                elif command == "status" or command == "s":
                    await self.get_status()
                elif command == "start":
                    await self.start_simulation()
                elif command == "stop":
                    await self.stop_simulation()
                elif command == "reset":
                    await self.reset_simulation()
                elif command == "help" or command == "h":
                    print("Available commands:")
                    print("  status (s) - Get server status")
                    print("  start      - Start simulation")
                    print("  stop       - Stop simulation") 
                    print("  reset      - Reset simulation")
                    print("  quit (q)   - Exit")
                elif command == "":
                    continue
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except EOFError:
                print("\nExiting...")
                break
    
    async def run_tests(self) -> None:
        """Run a series of communication tests."""
        print("\n=== RUNNING COMMUNICATION TESTS ===")
        
        tests = [
            ("Status Check", self.get_status),
            ("Start Simulation", self.start_simulation),
            ("Status Check (after start)", self.get_status),
            ("Stop Simulation", self.stop_simulation),
            ("Reset Simulation", self.reset_simulation),
        ]
        
        for test_name, test_func in tests:
            print(f"\n--- {test_name} ---")
            try:
                await test_func()
                await asyncio.sleep(0.5)  # Brief pause between tests
            except Exception as e:
                print(f"✗ Test failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from server."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        print("Disconnected from server")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple AI Hydra MQ Client")
    parser.add_argument(
        "--server", 
        default="tcp://localhost:5555",
        help="ZeroMQ server address (default: tcp://localhost:5555)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run automated tests instead of interactive mode"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("AI Hydra Simple MQ Client")
    print("=" * 40)
    
    client = SimpleHydraClient(server_address=args.server)
    
    try:
        # Connect to server
        if await client.connect():
            if args.test:
                await client.run_tests()
            else:
                await client.interactive_mode()
        else:
            print("Failed to connect to server. Exiting.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        logging.exception("Unexpected error")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())