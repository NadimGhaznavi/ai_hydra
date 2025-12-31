#!/usr/bin/env python3
"""
AI Hydra TUI Client Demo

This script demonstrates how to use the AI Hydra TUI client.
It includes a mock server for testing without the full AI Hydra system.
"""

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import zmq
import zmq.asyncio

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class MockHydraServer:
    """Mock AI Hydra server for TUI testing."""
    
    def __init__(self, port=5555):
        self.port = port
        self.context = zmq.asyncio.Context()
        self.socket = None
        self.running = False
        
        # Mock game state
        self.game_state = {
            "snake_head": [10, 10],
            "snake_body": [[9, 10], [8, 10], [7, 10]],
            "food_position": [15, 12],
            "score": 0,
            "moves_count": 0,
            "grid_size": [20, 20],
            "is_game_over": False,
            "epoch": 1
        }
        
        self.simulation_state = "idle"
        self.start_time = time.time()
        
    async def start(self):
        """Start the mock server."""
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        self.running = True
        
        print(f"🚀 Mock AI Hydra server started on port {self.port}")
        print("📡 Waiting for TUI client connections...")
        
        while self.running:
            try:
                # Wait for request
                message_data = await self.socket.recv_string()
                message = json.loads(message_data)
                
                # Process command
                response = await self.handle_command(message)
                
                # Send response
                await self.socket.send_string(json.dumps(response))
                
            except Exception as e:
                print(f"❌ Server error: {e}")
                break
    
    async def handle_command(self, message):
        """Handle incoming commands from TUI client."""
        command = message.get("command", "")
        client_id = message.get("client_id", "unknown")
        
        print(f"📨 Received command: {command} from {client_id}")
        
        if command == "ping":
            return {"success": True, "message": "pong"}
            
        elif command == "start_simulation":
            self.simulation_state = "running"
            config = message.get("data", {})
            self.game_state["grid_size"] = config.get("grid_size", [20, 20])
            self.start_time = time.time()
            
            # Start mock game updates
            asyncio.create_task(self.simulate_game())
            
            return {"success": True, "message": "Simulation started"}
            
        elif command == "stop_simulation":
            self.simulation_state = "stopped"
            return {"success": True, "message": "Simulation stopped"}
            
        elif command == "pause_simulation":
            self.simulation_state = "paused"
            return {"success": True, "message": "Simulation paused"}
            
        elif command == "resume_simulation":
            self.simulation_state = "running"
            return {"success": True, "message": "Simulation resumed"}
            
        elif command == "reset_simulation":
            self.simulation_state = "idle"
            self.game_state = {
                "snake_head": [10, 10],
                "snake_body": [[9, 10], [8, 10], [7, 10]],
                "food_position": [15, 12],
                "score": 0,
                "moves_count": 0,
                "grid_size": [20, 20],
                "is_game_over": False,
                "epoch": 1
            }
            return {"success": True, "message": "Simulation reset"}
            
        elif command == "get_status":
            runtime_seconds = int(time.time() - self.start_time)
            
            return {
                "success": True,
                "data": {
                    "simulation_state": self.simulation_state,
                    "game_state": self.game_state,
                    "runtime_seconds": runtime_seconds
                }
            }
            
        elif command == "heartbeat":
            return {"success": True, "message": "heartbeat acknowledged"}
            
        else:
            return {"success": False, "error": f"Unknown command: {command}"}
    
    async def simulate_game(self):
        """Simulate game updates while running."""
        move_count = 0
        
        while self.simulation_state == "running":
            # Simulate snake movement
            head_x, head_y = self.game_state["snake_head"]
            
            # Move right and wrap around
            head_x = (head_x + 1) % self.game_state["grid_size"][0]
            
            # Update snake position
            old_head = self.game_state["snake_head"]
            self.game_state["snake_head"] = [head_x, head_y]
            
            # Move body
            self.game_state["snake_body"].insert(0, old_head)
            
            # Check if food eaten
            if [head_x, head_y] == self.game_state["food_position"]:
                self.game_state["score"] += 10
                # Move food to random position
                import random
                grid_w, grid_h = self.game_state["grid_size"]
                self.game_state["food_position"] = [
                    random.randint(0, grid_w - 1),
                    random.randint(0, grid_h - 1)
                ]
            else:
                # Remove tail if no food eaten
                self.game_state["snake_body"].pop()
            
            move_count += 1
            self.game_state["moves_count"] = move_count
            
            # Simulate game over and epoch increment every 20 moves
            if move_count % 20 == 0:
                self.game_state["is_game_over"] = True
                self.game_state["epoch"] += 1
                # Reset for next epoch
                self.game_state["snake_head"] = [10, 10]
                self.game_state["snake_body"] = [[9, 10], [8, 10], [7, 10]]
                self.game_state["score"] = 0
                self.game_state["moves_count"] = 0
                self.game_state["is_game_over"] = False
                move_count = 0
            
            # Wait before next move
            await asyncio.sleep(0.5)  # Move every 500ms
    
    def stop(self):
        """Stop the mock server."""
        self.running = False
        if self.socket:
            self.socket.close()
        self.context.term()


def run_mock_server():
    """Run the mock server in a separate thread."""
    async def server_main():
        server = MockHydraServer()
        try:
            await server.start()
        except KeyboardInterrupt:
            print("\n🛑 Mock server shutting down...")
            server.stop()
    
    asyncio.run(server_main())


def main():
    """Main demo function."""
    print("🎮 AI Hydra TUI Client Demo")
    print("=" * 50)
    print()
    print("This demo will:")
    print("1. Start a mock AI Hydra server")
    print("2. Launch the TUI client")
    print("3. You can test the interface with fake data")
    print()
    
    choice = input("Start demo? (y/n): ").lower().strip()
    if choice != 'y':
        print("Demo cancelled.")
        return
    
    print("\n🚀 Starting mock server...")
    
    # Start mock server in background thread
    server_thread = threading.Thread(target=run_mock_server, daemon=True)
    server_thread.start()
    
    # Give server time to start
    time.sleep(2)
    
    print("🖥️  Starting TUI client...")
    print("💡 Try the following in the TUI:")
    print("   - Click 'Start' to begin simulation")
    print("   - Watch the snake move automatically")
    print("   - Try 'Pause', 'Resume', 'Stop', 'Reset'")
    print("   - Press Ctrl+C to quit")
    print()
    
    # Import and run TUI client
    try:
        from ai_hydra.tui.client import HydraClient
        
        app = HydraClient(server_address="tcp://localhost:5555")
        app.run()
        
    except ImportError as e:
        print(f"❌ Could not import TUI client: {e}")
        print("💡 Make sure to install TUI dependencies: pip install -e .[tui]")
    except Exception as e:
        print(f"❌ TUI client error: {e}")
    
    print("\n👋 Demo finished!")


if __name__ == "__main__":
    main()