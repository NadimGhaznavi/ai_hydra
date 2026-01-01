# AI Hydra Examples

This directory contains simple examples to help you get started with AI Hydra.

## Quick Start

1. **Install AI Hydra**:
   ```bash
   pip install ai-hydra
   ```

2. **Start the server** (in one terminal):
   ```bash
   ai-hydra-server
   ```

3. **Run the client** (in another terminal):
   ```bash
   python examples/simple_client.py
   ```

## Examples

### `simple_client.py`
A basic client that:
- Connects to an AI Hydra server
- Starts a new Snake game
- Monitors the game progress
- Shows the final score

Perfect for understanding how to communicate with AI Hydra servers.

### `test_router_demo.py`
A comprehensive router demonstration that shows:
- Starting the Hydra Router system
- Connecting multiple clients and a server
- Message routing between clients and server
- Heartbeat monitoring and client lifecycle
- Complete workflow from startup to shutdown

This is the most complete example showing how all router components work together. Run it with:

```bash
python examples/test_router_demo.py
```

### `simple_router_test.py`
A basic router functionality test that:
- Creates and starts a router instance
- Tests basic router operations
- Demonstrates proper startup and shutdown

Good for understanding router basics and testing router functionality.

### `debug_test.py`
A minimal router debugging script that:
- Tests router creation and method availability
- Validates router startup process
- Useful for troubleshooting router issues

## What You'll See

### Simple Client Output

When you run the simple client, you'll see output like:

```
🐍 AI Hydra Simple Client
==============================
📡 Connecting to server...
✅ Connected to server!

🎮 Starting new game...
✅ Game started!
🆔 Simulation ID: abc123

📊 Monitoring game progress...
🎯 Score: 1 | Moves: 15
🎯 Score: 2 | Moves: 28
🎯 Score: 3 | Moves: 41
...
🏁 Game Over!
📊 Final Score: 12
🎉 Great job! Score > 10 means good collision avoidance!
```

### Router Demo Output

When you run the router demo, you'll see comprehensive output showing the complete router workflow:

```
🎯 Hydra Router Demo
This demo shows the router working with multiple clients and a server
Press Ctrl+C to stop

🚀 Starting Hydra Router Demo
==================================================
📡 Starting router on localhost:5556...
🖥️  Starting demo server...
✅ Demo server connected and ready
👤 Starting demo client demo-client-001...
✅ Demo client demo-client-001 connected
👤 Starting demo client demo-client-002...
✅ Demo client demo-client-002 connected

🎬 Starting demo scenario...
------------------------------

📋 Step 1: Client 1 requests server status
👤 Client demo-client-001 sent: GET_STATUS
🖥️  Server received: GET_STATUS with data: {}
🖥️  Server sent: STATUS_RESPONSE
👤 Client demo-client-001 received: STATUS_RESPONSE with data: {'status': 'running', 'uptime': 1704067200.0, 'active_simulations': 1}

📋 Step 2: Client 2 starts simulation
👤 Client demo-client-002 sent: START_SIMULATION
🖥️  Server received: START_SIMULATION with data: {'config': 'demo_config', 'seed': 42}
🖥️  Server sent: SIMULATION_STARTED
👤 Client demo-client-002 received: SIMULATION_STARTED with data: {'status': 'Simulation started successfully', 'sim_id': 'demo-001'}

📋 Step 3: Both clients listen for server responses
👤 Client demo-client-001 listening for responses...
👤 Client demo-client-002 listening for responses...

📋 Step 4: Demonstrate heartbeat monitoring
💓 Clients are sending heartbeats every 5 seconds...

🎉 Demo completed successfully!

🧹 Cleaning up...
👤 Stopping demo client demo-client-001...
👤 Stopping demo client demo-client-002...
🖥️  Stopping demo server...
✅ Cleanup complete
```

## Customizing Games

You can modify the game settings in `simple_client.py`:

```python
game_config = {
    'grid_size': [10, 10],    # Larger grid
    'move_budget': 50,        # More thinking time
    'random_seed': 123        # Different random seed
}
```

## Router Demo Features

The `test_router_demo.py` example demonstrates advanced router features:

- **Multiple Client Support**: Shows how multiple clients can connect simultaneously
- **Server Integration**: Demonstrates server-client communication through the router
- **Message Routing**: Shows how messages are routed between clients and server
- **Heartbeat Monitoring**: Demonstrates automatic client lifecycle management
- **Error Handling**: Shows proper cleanup and error recovery
- **Async Operations**: Uses modern async/await patterns for concurrent operations

To customize the router demo:

```python
# Modify router configuration
router = HydraRouter(
    router_address="127.0.0.1", 
    router_port=5556,           # Change port if needed
    log_level="DEBUG"           # More verbose logging
)

# Modify client configuration
client = MQClient(
    router_address="tcp://localhost:5556",
    client_type="HydraClient",
    client_id="custom-client-001"  # Custom client ID
)
```

## Need Help?

- Make sure the server is running before starting the client
- Check that port 5555 is not blocked by firewall
- See the main documentation for advanced usage