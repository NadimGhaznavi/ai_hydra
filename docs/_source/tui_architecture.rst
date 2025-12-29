TUI Architecture
================

The AI Hydra Terminal User Interface is built using the `Textual <https://textual.textualize.io/>`_ framework, providing a modern, reactive terminal interface for real-time visualization and control.

Architecture Overview
---------------------

The TUI follows a reactive architecture pattern where the UI responds to both user interactions and server-pushed updates:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                    TUI Client                           │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
   │  │   Textual   │  │    State    │  │    Data     │    │
   │  │  UI Layer   │◄─┤  Manager    │◄─┤  Manager    │    │
   │  └─────────────┘  └─────────────┘  └─────────────┘    │
   │         │                │                │            │
   │         └────────────────┼────────────────┘            │
   │                          │                             │
   │  ┌─────────────────────────────────────────────────┐   │
   │  │         Communication Manager                   │   │
   │  └─────────────────────────────────────────────────┘   │
   └─────────────────────┬───────────────────────────────────┘
                         │ ZeroMQ REQ/REP
   ┌─────────────────────▼───────────────────────────────────┐
   │                AI Hydra Server                          │
   └─────────────────────────────────────────────────────────┘

Core Components
---------------

HydraClient (Main Application)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main Textual application class that orchestrates all UI components:

.. code-block:: python

   class HydraClient(App):
       """Main TUI application for AI Hydra simulation system."""
       
       # Reactive variables for real-time updates
       simulation_state = var("idle")
       game_score = var(0)
       snake_length = var(3)
       moves_count = var(0)
       runtime_seconds = var(0)

**Key Features:**

* **Reactive Variables**: Automatic UI updates when values change
* **Async Communication**: Non-blocking ZeroMQ message handling
* **State Management**: Centralized application state tracking
* **Error Handling**: Graceful error recovery and user notification

HydraGameBoard (Game Visualization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Custom Textual widget for real-time Snake game visualization:

.. code-block:: python

   class HydraGameBoard(ScrollView):
       """Real-time game board visualization for AI Hydra."""
       
       # Reactive properties for game state
       snake_head = var(Offset(10, 10))
       snake_body = var([])
       food_position = var(Offset(5, 5))
       grid_size = var((20, 20))

**Rendering Features:**

* **Efficient Line Rendering**: Optimized for terminal performance
* **Color Coding**: Distinct colors for snake head, body, and food
* **Smooth Updates**: Reactive updates with minimal screen flicker
* **Scalable Grid**: Supports various grid sizes with automatic scaling

Communication Layer
~~~~~~~~~~~~~~~~~~~

Handles all ZeroMQ communication with the AI Hydra server:

**Message Flow:**

1. **Command Messages**: User actions → ZeroMQ REQ → Server
2. **Status Updates**: Server → ZeroMQ REP → UI Updates
3. **Heartbeat**: Periodic connection health checks
4. **Error Handling**: Automatic reconnection and error recovery

**Supported Commands:**

* ``ping``: Test server connectivity
* ``start_simulation``: Begin simulation with configuration
* ``stop_simulation``: Stop current simulation
* ``pause_simulation``: Pause running simulation
* ``resume_simulation``: Resume paused simulation
* ``reset_simulation``: Reset simulation state
* ``get_status``: Retrieve current status
* ``heartbeat``: Maintain connection

Reactive System
---------------

The TUI uses Textual's reactive system for efficient updates:

**Reactive Variables**

Variables that automatically trigger UI updates when changed:

.. code-block:: python

   # When simulation_state changes, UI automatically updates
   self.simulation_state = "running"  # Triggers watch_simulation_state()
   
   # When game_score changes, score display updates
   self.game_score = 42  # Triggers watch_game_score()

**Watcher Methods**

Methods that respond to reactive variable changes:

.. code-block:: python

   def watch_simulation_state(self, old_state: str, new_state: str) -> None:
       """React to simulation state changes."""
       # Update UI state classes for CSS styling
       if old_state:
           self.remove_class(old_state)
       self.add_class(new_state)
       
       # Update status display
       state_label = self.query_one("#sim_state", Label)
       state_label.update(new_state.title())

CSS Styling System
------------------

The TUI uses Textual's CSS system for theming and layout:

**Grid Layout**

.. code-block:: css

   Screen {
       layout: grid;
       grid-size: 3 4;
       grid-rows: 3 1fr 1fr 1fr;
       grid-columns: 1fr 2fr 1fr;
   }

**Component Styling**

.. code-block:: css

   #game_board {
       border: round #0c323e;
       align: center middle;
       background: black;
   }
   
   Button {
       align: center middle;
       border: round #0c323e;
       width: 10;
       background: black;
       margin: 0 1;
   }

**State-Based Visibility**

.. code-block:: css

   .running #btn_start {
       display: none;
   }
   
   .paused #btn_pause {
       display: none;
   }

Color Theme
-----------

The TUI uses a custom AI Hydra theme adapted from proven patterns:

.. code-block:: python

   HYDRA_THEME = Theme(
       name="hydra_dark",
       primary="#88C0D0",      # Light blue
       secondary="#1f6a83ff",  # Dark blue
       accent="#B48EAD",       # Purple
       foreground="#31b8e6",   # Cyan
       background="black",     # Black
       success="#A3BE8C",      # Green
       warning="#EBCB8B",      # Yellow
       error="#BF616A",        # Red
       surface="#111111",      # Dark gray
       panel="#000000",        # Black
       dark=True
   )

**Game Board Colors:**

* **Snake Head**: Bright blue (#88C0D0)
* **Snake Body**: Green (#A3BE8C)
* **Food**: Red (#BF616A)
* **Empty Squares**: Alternating dark gray (#111111) and black (#000000)

Performance Optimizations
-------------------------

**Efficient Rendering**

* **Line-by-Line**: Only renders visible lines
* **Dirty Region Tracking**: Updates only changed areas
* **Minimal Redraws**: Reactive system prevents unnecessary updates

**Memory Management**

* **Async Operations**: Non-blocking I/O operations
* **Resource Cleanup**: Proper cleanup on application exit
* **Connection Pooling**: Reuses ZeroMQ connections

**Update Throttling**

* **Status Polling**: Limited to 1Hz to prevent spam
* **Heartbeat**: 30-second intervals for connection health
* **UI Updates**: Batched updates for smooth performance

Error Handling Strategy
-----------------------

**Connection Errors**

* **Automatic Retry**: Exponential backoff for reconnection
* **User Notification**: Clear error messages in message log
* **Graceful Degradation**: UI remains functional during disconnection

**UI Errors**

* **Exception Handling**: Prevents crashes from UI errors
* **Error Logging**: Detailed logging for debugging
* **Recovery Actions**: Automatic recovery where possible

**Server Errors**

* **Timeout Handling**: 5-second timeout for server responses
* **Error Display**: Server error messages shown to user
* **Fallback Behavior**: Safe defaults when server unavailable

Extension Points
----------------

The TUI architecture supports easy extension:

**Adding New Widgets**

1. Create widget class inheriting from Textual base classes
2. Add to ``compose()`` method in ``HydraClient``
3. Update CSS styling in ``hydra_client.tcss``
4. Add reactive variables and watchers as needed

**Adding New Commands**

1. Add command handler in ``HydraClient``
2. Update ``send_command()`` method
3. Add server response handling
4. Update UI based on command results

**Adding New Status Fields**

1. Add reactive variable to ``HydraClient``
2. Add watcher method for UI updates
3. Update ``process_status_update()`` method
4. Add UI elements in ``compose()`` method

Testing Strategy
----------------

The TUI architecture supports comprehensive testing:

**Unit Tests**

* **Widget Testing**: Individual widget functionality
* **State Management**: Reactive variable behavior
* **Communication**: ZeroMQ message handling

**Integration Tests**

* **End-to-End**: Complete user workflows
* **Server Integration**: Real server communication
* **Error Scenarios**: Connection failures and recovery

**Mock Testing**

* **Mock Server**: Simulated server for isolated testing
* **Mock UI**: Headless testing of application logic
* **Mock Communication**: Simulated network conditions

This architecture provides a solid foundation for the TUI while maintaining flexibility for future enhancements and extensions.