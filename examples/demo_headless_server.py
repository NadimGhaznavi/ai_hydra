#!/usr/bin/env python3
"""
Demonstration of the Headless ZeroMQ Server.

This script shows how to run AI Hydra as a completely
headless AI agent that communicates only via ZeroMQ messages.
"""

import asyncio
import time
import logging
from ai_hydra.headless_server import HeadlessServer
from ai_hydra.zmq_client_example import ZMQClient


async def demo_headless_system():
    """Demonstrate the complete headless system."""
    print("=== AI Hydra Headless Demo ===\n")
    
    # Configure logging to be less verbose for demo
    logging.getLogger().setLevel(logging.WARNING)
    
    # Start server in background
    print("1. Starting headless server...")
    server = HeadlessServer(
        bind_address="tcp://*:5557",  # Use different port for demo
        heartbeat_interval=2.0,
        log_level="WARNING"
    )
    
    # Start server task
    server_task = asyncio.create_task(server.start())
    
    # Give server time to start
    await asyncio.sleep(2)
    print("   ✓ Server started on tcp://*:5557")
    
    try:
        # Create client and connect
        print("\n2. Connecting client to server...")
        client = ZMQClient("tcp://localhost:5557")
        
        if not await client.connect():
            print("   ✗ Failed to connect to server")
            return
        
        print("   ✓ Client connected successfully")
        
        # Get initial status
        print("\n3. Getting server status...")
        await client.get_status()
        
        # Start a simulation
        print("\n4. Starting Snake AI simulation...")
        config = {
            "grid_size": [8, 8],
            "initial_snake_length": 3,
            "move_budget": 20,  # Small budget for quick demo
            "random_seed": 42,
            "nn_enabled": True,
            "food_reward": 10,
            "collision_penalty": -10
        }
        
        response = await client.start_simulation(config)
        if response and response.message_type.value == "simulation_started":
            print("   ✓ Simulation started successfully")
        else:
            print("   ✗ Failed to start simulation")
            return
        
        # Monitor simulation for a few seconds
        print("\n5. Monitoring simulation progress...")
        for i in range(5):
            await asyncio.sleep(1)
            print(f"   Checking status ({i+1}/5)...")
            await client.get_status()
        
        # Pause simulation
        print("\n6. Pausing simulation...")
        await client.pause_simulation()
        print("   ✓ Simulation paused")
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Resume simulation
        print("\n7. Resuming simulation...")
        await client.resume_simulation()
        print("   ✓ Simulation resumed")
        
        # Let it run a bit more
        await asyncio.sleep(3)
        
        # Stop simulation
        print("\n8. Stopping simulation...")
        await client.stop_simulation()
        print("   ✓ Simulation stopped")
        
        # Final status check
        print("\n9. Final status check...")
        await client.get_status()
        
        # Disconnect client
        print("\n10. Disconnecting client...")
        await client.disconnect()
        print("    ✓ Client disconnected")
        
        print("\n=== Demo Complete ===")
        print("\nKey Features Demonstrated:")
        print("• Headless AI agent (no GUI, no direct user interaction)")
        print("• ZeroMQ message-based communication")
        print("• Real-time status monitoring")
        print("• Simulation control (start/stop/pause/resume)")
        print("• Performance metrics tracking")
        print("• Complete separation of AI logic and presentation")
        
    except Exception as e:
        print(f"\nDemo error: {e}")
    finally:
        # Stop server
        print("\nStopping server...")
        await server.stop()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        print("✓ Server stopped")


async def quick_test():
    """Quick test to verify components work."""
    print("=== Quick Component Test ===\n")
    
    # Test message protocol
    from ai_hydra.zmq_protocol import ZMQMessage, MessageType, MessageBuilder
    
    print("1. Testing message protocol...")
    message = ZMQMessage.create_command(
        MessageType.START_SIMULATION,
        "test_client",
        "req_123",
        {"config": {"grid_size": [10, 10]}}
    )
    
    json_str = message.to_json()
    deserialized = ZMQMessage.from_json(json_str)
    
    assert deserialized.message_type == MessageType.START_SIMULATION
    assert deserialized.client_id == "test_client"
    print("   ✓ Message serialization/deserialization works")
    
    # Test server initialization
    print("\n2. Testing server initialization...")
    server = HeadlessServer(bind_address="tcp://*:5558")
    assert server.bind_address == "tcp://*:5558"
    print("   ✓ Server initializes correctly")
    
    # Test client initialization
    print("\n3. Testing client initialization...")
    client = ZMQClient("tcp://localhost:5558")
    assert client.server_address == "tcp://localhost:5558"
    assert not client.is_connected
    print("   ✓ Client initializes correctly")
    
    print("\n=== All Components Working ===")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Headless Server Demo")
    parser.add_argument("--mode", choices=["demo", "test"], default="demo",
                       help="Run full demo or quick test")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        print("Starting full headless system demo...")
        print("This will take about 30 seconds to complete.\n")
        asyncio.run(demo_headless_system())
    else:
        asyncio.run(quick_test())