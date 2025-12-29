Terminal User Interface - Getting Started
=========================================

The AI Hydra Terminal User Interface (TUI) provides a rich, interactive way to visualize and control the Snake Game AI simulation directly from your terminal.

Installation
------------

Install the TUI dependencies:

.. code-block:: bash

   pip install -e .[tui]

This installs the required dependencies:

* **Textual**: Modern terminal UI framework
* **Rich**: Rich text and beautiful formatting

Quick Start
-----------

1. **Start the headless server** (in one terminal):

   .. code-block:: bash

      ai-hydra-server

2. **Start the TUI client** (in another terminal):

   .. code-block:: bash

      ai-hydra-tui --server tcp://localhost:5555

3. **Use the interface**:

   * Click **Start** to begin a simulation
   * Watch the Snake game play in real-time
   * Use **Pause**, **Resume**, **Stop**, **Reset** as needed
   * Adjust configuration parameters
   * Monitor live statistics

Command Line Options
--------------------

.. code-block:: bash

   ai-hydra-tui [OPTIONS]

Options:

* ``--server ADDRESS``: ZeroMQ server address (default: tcp://localhost:5555)
* ``--verbose``: Enable verbose logging
* ``--help``: Show help message

Examples:

.. code-block:: bash

   # Connect to local server
   ai-hydra-tui

   # Connect to remote server
   ai-hydra-tui --server tcp://192.168.1.100:5555

   # Enable verbose logging
   ai-hydra-tui --verbose

   # Run as Python module
   python -m ai_hydra.tui.client --server tcp://localhost:5555

Interface Overview
------------------

The TUI interface is organized into several panels:

**Game Board Panel**
   Real-time visualization of the Snake game with:
   
   * Colorful snake head (bright blue)
   * Snake body segments (green)
   * Food items (red)
   * Checkerboard background pattern

**Control Panel**
   Interactive controls including:
   
   * **Start**: Begin new simulation
   * **Stop**: Stop current simulation
   * **Pause**: Pause running simulation
   * **Resume**: Resume paused simulation
   * **Reset**: Reset and clear all data
   * **Configuration**: Grid size and move budget inputs

**Status Panel**
   Live monitoring display:
   
   * Current simulation state
   * Game score
   * Number of moves executed
   * Snake length
   * Runtime duration

**Message Log Panel**
   Real-time message display:
   
   * Connection status
   * Simulation events
   * Error messages (in red)
   * Warning messages (in yellow)
   * Info messages (in default color)

Navigation
----------

**Keyboard Controls:**

* ``Tab``: Navigate between UI elements
* ``Shift+Tab``: Navigate backwards
* ``Enter``: Activate buttons
* ``Escape``: Cancel current action
* ``Ctrl+C``: Quit application
* ``Arrow Keys``: Navigate within input fields

**Mouse Support:**

* Click buttons to activate
* Click input fields to edit
* Scroll in message log panel

Configuration
-------------

**Grid Size**
   Format: "width,height" (e.g., "20,20")
   
   * Minimum: 5x5
   * Maximum: 50x50
   * Default: 20x20

**Move Budget**
   Number of moves per decision cycle
   
   * Minimum: 1
   * Maximum: 1000
   * Default: 100

Troubleshooting
---------------

**Connection Issues**

If you can't connect to the server:

1. Verify the server is running: ``ai-hydra-server``
2. Check the server address and port
3. Ensure no firewall is blocking the connection
4. Try verbose mode: ``ai-hydra-tui --verbose``

**Display Issues**

If the interface looks wrong:

1. Ensure your terminal supports colors (most modern terminals do)
2. Try resizing the terminal window
3. Check that your terminal is at least 80x24 characters
4. Verify Textual compatibility with your terminal

**Performance Issues**

If the interface is slow:

1. Use a smaller grid size (e.g., 10x10 instead of 20x20)
2. Reduce the move budget
3. Check system resources (CPU, memory)
4. Close other resource-intensive applications

Next Steps
----------

* Learn about :doc:`tui_architecture` for technical details
* See :doc:`tui_controls` for complete control reference
* Check :doc:`troubleshooting` for common issues
* Review the :doc:`zmq_protocol` for server communication details