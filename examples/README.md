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

## What You'll See

When you run the client, you'll see output like:

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

## Customizing Games

You can modify the game settings in `simple_client.py`:

```python
game_config = {
    'grid_size': [10, 10],    # Larger grid
    'move_budget': 50,        # More thinking time
    'random_seed': 123        # Different random seed
}
```

## Need Help?

- Make sure the server is running before starting the client
- Check that port 5555 is not blocked by firewall
- See the main documentation for advanced usage