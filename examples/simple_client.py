#!/usr/bin/env python3
"""
Simple AI Hydra Client Example

This script shows how to connect to an AI Hydra server and control it.
Make sure you have a server running first with: ai-hydra-server
"""

import zmq
import json
import time
import sys


def create_client():
    """Create a ZeroMQ client connection."""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://localhost:5555')
    return socket


def send_message(socket, message_type, data=None):
    """Send a message to the server and get response."""
    message = {
        'type': message_type,
        'data': data or {}
    }
    
    try:
        socket.send_string(json.dumps(message))
        response = socket.recv_string(zmq.NOBLOCK)
        return json.loads(response)
    except zmq.Again:
        print("⚠️  Server didn't respond (is it running?)")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Main client demonstration."""
    print("🐍 AI Hydra Simple Client")
    print("=" * 30)
    
    # Connect to server
    print("📡 Connecting to server...")
    socket = create_client()
    
    # Test connection
    response = send_message(socket, 'PING')
    if not response:
        print("❌ Cannot connect to server!")
        print("💡 Make sure server is running: ai-hydra-server")
        sys.exit(1)
    
    print("✅ Connected to server!")
    
    # Start a simulation
    print("\n🎮 Starting new game...")
    game_config = {
        'grid_size': [8, 8],
        'move_budget': 30,
        'random_seed': 42
    }
    
    response = send_message(socket, 'START_SIMULATION', game_config)
    if response and response.get('success'):
        print("✅ Game started!")
        simulation_id = response.get('simulation_id')
        print(f"🆔 Simulation ID: {simulation_id}")
    else:
        print("❌ Failed to start game")
        sys.exit(1)
    
    # Monitor the game
    print("\n📊 Monitoring game progress...")
    print("(Press Ctrl+C to stop monitoring)")
    
    try:
        last_score = 0
        while True:
            # Get current status
            status = send_message(socket, 'GET_STATUS')
            
            if status:
                current_score = status.get('current_score', 0)
                is_running = status.get('is_running', False)
                moves_made = status.get('moves_made', 0)
                
                # Show progress if score changed
                if current_score != last_score:
                    print(f"🎯 Score: {current_score} | Moves: {moves_made}")
                    last_score = current_score
                
                # Check if game ended
                if not is_running:
                    final_score = status.get('final_score', current_score)
                    print(f"\n🏁 Game Over!")
                    print(f"📊 Final Score: {final_score}")
                    
                    if final_score > 10:
                        print("🎉 Great job! Score > 10 means good collision avoidance!")
                    else:
                        print("💪 Try again! Scores > 10 indicate successful AI learning.")
                    
                    break
            
            time.sleep(1)  # Check every second
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
        
        # Ask if user wants to stop the game
        try:
            stop = input("Stop the current game? (y/n): ").lower().strip()
            if stop == 'y':
                response = send_message(socket, 'STOP_SIMULATION')
                if response and response.get('success'):
                    print("✅ Game stopped")
                else:
                    print("⚠️  Could not stop game")
        except KeyboardInterrupt:
            pass
    
    print("\n👋 Client disconnecting...")
    socket.close()


if __name__ == "__main__":
    main()