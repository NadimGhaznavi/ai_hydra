Getting Started with AI Hydra
=============================

Welcome to AI Hydra! This guide will get you up and running in just a few minutes.

What is AI Hydra?
------------------

AI Hydra is an intelligent Snake game AI that thinks before it moves. Unlike simple AIs that make quick decisions, AI Hydra spawns multiple "thinking processes" to explore different options before choosing the best move. This makes it much better at avoiding collisions and achieving high scores.

Quick Installation
------------------

**Step 1: Install AI Hydra**

.. code-block:: bash

   pip install ai-hydra

That's it! AI Hydra and all its dependencies are now installed.

**Step 2: Verify Installation**

.. code-block:: bash

   ai-hydra --help

You should see a help message with available commands.

Running Your First Simulation
------------------------------

**Option 1: Router Demo (Comprehensive Example)**

The most complete way to see AI Hydra in action is with the router demonstration:

.. code-block:: bash

   # Run the comprehensive router demo
   python examples/test_router_demo.py

This demo shows:
- Complete router system startup and configuration
- Multiple clients connecting to a server through the router
- Real-time message routing and communication
- Heartbeat monitoring and client lifecycle management
- Proper cleanup and shutdown procedures

The demo provides detailed output showing each step of the process, making it perfect for understanding how all components work together.

**Option 2: Terminal User Interface (Interactive)**

The easiest way to interact with AI Hydra is with the TUI:

.. code-block:: bash

   # Install TUI dependencies
   pip install ai-hydra[tui]
   
   # Start the server (in one terminal)
   ai-hydra-server
   
   # Start the TUI client (in another terminal)
   ai-hydra-tui

This opens a beautiful terminal interface where you can:
- Watch the Snake game play in real-time with colors
- Control simulations with buttons (Start, Stop, Pause, Resume)
- Monitor live statistics (score, moves, runtime)
- Adjust settings (grid size, move budget)
- View real-time messages and logs

**Option 3: Simple Command Line**

Run a basic simulation with default settings:

.. code-block:: bash

   ai-hydra

This will:
- Start a Snake game on a 10x10 grid
- Let the AI play for up to 1000 moves
- Show you the final score
- Save results to a file

**Option 4: Custom Settings**

Run with your own settings:

.. code-block:: bash

   ai-hydra simulation.grid_size=[8,8] simulation.move_budget=50

This creates a smaller 8x8 game with a budget of 50 "thinking moves" per decision.

Understanding the Output
------------------------

When AI Hydra runs, you'll see output like this:

.. code-block:: text

   🐍 AI Hydra Starting...
   📊 Grid: 10x10, Budget: 100 moves per decision
   🧠 Neural Network: Enabled
   
   🎯 Decision 1: Exploring 3 options... → STRAIGHT (Score: 1)
   🎯 Decision 2: Exploring 3 options... → LEFT (Score: 2)
   🎯 Decision 3: Exploring 3 options... → STRAIGHT (Score: 3)
   ...
   
   🏁 Game Over! Final Score: 15
   ✅ Success! Score > 10 (collision avoidance achieved)

**What this means:**
- Each decision explores multiple possibilities
- The AI learns and improves as it plays
- Scores above 10 indicate good collision avoidance

Setting Up a Server and Client
-------------------------------

Want to run AI Hydra as a service? Here's how to set up a server that other programs can talk to.

**Step 1: Start the Server**

Open a terminal and run:

.. code-block:: bash

   ai-hydra-server

You'll see:

.. code-block:: text

   🚀 AI Hydra Server Starting...
   📡 Listening on: tcp://*:5555
   💓 Heartbeat: Every 5 seconds
   ✅ Server ready for connections

Keep this terminal open - your server is now running!

**Step 2: Connect with TUI Client (Recommended)**

Open a **new terminal** and start the interactive TUI:

.. code-block:: bash

   ai-hydra-tui

This opens a beautiful terminal interface where you can:
- Click "Start" to begin a simulation
- Watch the Snake game play in real-time
- Monitor live statistics and messages
- Control the simulation with buttons

**Alternative: Connect with Python Client**

If you prefer programmatic control, create a simple client:

.. code-block:: bash

   python -c "
   import zmq
   import json
   import time
   
   # Connect to server
   context = zmq.Context()
   socket = context.socket(zmq.REQ)
   socket.connect('tcp://localhost:5555')
   
   # Start a game
   message = {
       'type': 'START_SIMULATION',
       'data': {
           'grid_size': [8, 8],
           'move_budget': 30
       }
   }
   
   socket.send_string(json.dumps(message))
   response = socket.recv_string()
   print('Server response:', response)
   
   # Check status
   time.sleep(2)
   status_msg = {'type': 'GET_STATUS', 'data': {}}
   socket.send_string(json.dumps(status_msg))
   status = socket.recv_string()
   print('Game status:', status)
   "

This will start a game on your server and check its status.

**Step 3: Monitor Your Game**

You can send different commands to control the game:

.. code-block:: python

   # Get current game state
   {'type': 'GET_STATUS', 'data': {}}
   
   # Pause the game
   {'type': 'PAUSE_SIMULATION', 'data': {}}
   
   # Resume the game
   {'type': 'RESUME_SIMULATION', 'data': {}}
   
   # Stop the game
   {'type': 'STOP_SIMULATION', 'data': {}}

Common Use Cases
----------------

**1. Testing AI Performance**

.. code-block:: bash

   # Run 10 games and see average performance
   ai-hydra experiment.num_simulations=10

**2. Quick Games for Development**

.. code-block:: bash

   # Smaller, faster games for testing
   ai-hydra simulation.grid_size=[6,6] simulation.move_budget=20

**3. High-Performance Games**

.. code-block:: bash

   # Larger grid with more thinking time
   ai-hydra simulation.grid_size=[15,15] simulation.move_budget=200

**4. Headless Server for Integration**

.. code-block:: bash

   # Run server in background
   ai-hydra-server --bind "tcp://0.0.0.0:5555" --log-file /var/log/ai-hydra.log

Troubleshooting
---------------

**"Command not found" Error**

If `ai-hydra` command isn't found:

.. code-block:: bash

   # Try with python -m
   python -m ai_hydra.cli
   
   # Or check your PATH
   pip show ai-hydra

**Server Won't Start**

If the server fails to start:

.. code-block:: bash

   # Try a different port
   ai-hydra-server --bind "tcp://*:6666"
   
   # Check if port is in use
   netstat -an | grep 5555

**Games Run Too Slowly**

If simulations are too slow:

.. code-block:: bash

   # Use smaller settings
   ai-hydra simulation.grid_size=[6,6] simulation.move_budget=10

**Games End Too Quickly**

If the AI keeps crashing:

.. code-block:: bash

   # Give it more thinking time
   ai-hydra simulation.move_budget=100

Next Steps
----------

Now that you have AI Hydra running:

1. **Experiment with Settings**: Try different grid sizes and budgets
2. **Build a Client**: Create your own program to control AI Hydra
3. **Monitor Performance**: Watch how the AI learns and improves
4. **Integrate**: Use the server in your own applications

**Need Help?**

- Check the :doc:`troubleshooting` guide for common issues
- Read the :doc:`api_reference` for advanced usage
- Visit our GitHub repository for examples and support

**Want to Learn More?**

- :doc:`architecture` - How AI Hydra works under the hood
- :doc:`zmq_protocol` - Complete server communication guide
- :doc:`deployment` - Running AI Hydra in production