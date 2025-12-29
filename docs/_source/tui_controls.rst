TUI Controls Reference
======================

Complete reference for all controls and interactions available in the AI Hydra Terminal User Interface.

Interface Layout
----------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                    AI Hydra - Snake Game AI Monitor         │
   ├─────────────────┬───────────────────────────┬───────────────┤
   │   Control       │                           │    Status     │
   │   Panel         │        Game Board         │    Panel      │
   │                 │                           │               │
   │ [Start] [Stop]  │    ████████████████       │ State: Running│
   │ [Pause][Resume] │    █▓▓▓▓▓▓▓▓▓▓▓▓▓█       │ Score: 42     │
   │ [Reset]         │    █▓░░░●░░░░░░░░▓█       │ Moves: 156    │
   │                 │    █▓░░░░░░░░░░░▓█       │ Length: 8     │
   │ Grid: 20,20     │    █▓░░░░░░░░░░░▓█       │ Runtime: 2:34 │
   │ Budget: 100     │    █▓░░░░░░░░░░░▓█       │               │
   │                 │    █▓▓▓▓▓▓▓▓▓▓▓▓▓█       │               │
   │                 │    ████████████████       │               │
   ├─────────────────┴───────────────────────────┴───────────────┤
   │                        Messages                             │
   │ [12:34:56] Connected to server at tcp://localhost:5555     │
   │ [12:34:57] Simulation started                              │
   │ [12:34:58] Score increased to 42                           │
   └─────────────────────────────────────────────────────────────┘

Control Panel
-------------

Simulation Control Buttons
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Start Button**
   * **Function**: Begin a new simulation with current configuration
   * **State**: Available when simulation is idle or stopped
   * **Color**: Green
   * **Keyboard**: Tab to button, Enter to activate

**Stop Button**
   * **Function**: Stop the current simulation immediately
   * **State**: Available when simulation is running or paused
   * **Color**: Red
   * **Effect**: Simulation stops, returns to idle state

**Pause Button**
   * **Function**: Pause the running simulation while maintaining state
   * **State**: Available only when simulation is running
   * **Color**: Yellow
   * **Effect**: Simulation pauses, can be resumed later

**Resume Button**
   * **Function**: Resume a paused simulation from where it left off
   * **State**: Available only when simulation is paused
   * **Color**: Blue
   * **Effect**: Simulation continues from paused state

**Reset Button**
   * **Function**: Reset simulation and clear all data
   * **State**: Available in any state
   * **Color**: Purple
   * **Effect**: Clears game board, resets scores, clears message log

Configuration Inputs
~~~~~~~~~~~~~~~~~~~~

**Grid Size Input**
   * **Format**: "width,height" (e.g., "20,20")
   * **Range**: 5x5 to 50x50
   * **Default**: "20,20"
   * **Validation**: Must be two integers separated by comma
   * **Effect**: Sets the game board dimensions for new simulations

**Move Budget Input**
   * **Format**: Integer number (e.g., "100")
   * **Range**: 1 to 1000
   * **Default**: "100"
   * **Validation**: Must be positive integer
   * **Effect**: Sets the computational budget per decision cycle

Game Board Panel
----------------

Visual Elements
~~~~~~~~~~~~~~~

**Snake Head**
   * **Color**: Bright blue (#88C0D0)
   * **Symbol**: Two-character wide block
   * **Movement**: Updates in real-time as snake moves

**Snake Body**
   * **Color**: Green (#A3BE8C)
   * **Symbol**: Two-character wide blocks
   * **Trail**: Shows the path the snake has taken

**Food**
   * **Color**: Red (#BF616A)
   * **Symbol**: Two-character wide block
   * **Behavior**: Appears randomly, disappears when eaten

**Empty Squares**
   * **Colors**: Alternating dark gray (#111111) and black (#000000)
   * **Pattern**: Checkerboard pattern for visual clarity

Board Interaction
~~~~~~~~~~~~~~~~~

**Scrolling**
   * **Mouse Wheel**: Scroll vertically through large game boards
   * **Arrow Keys**: Navigate when board is focused
   * **Auto-Center**: Board automatically centers on snake head

**Resizing**
   * **Terminal Resize**: Board adapts to terminal size changes
   * **Aspect Ratio**: Maintains proper proportions
   * **Minimum Size**: Ensures board remains visible

Status Panel
------------

Status Displays
~~~~~~~~~~~~~~~

**Simulation State**
   * **Values**: Idle, Running, Paused, Stopped
   * **Color**: Changes based on state
   * **Update**: Real-time updates from server

**Game Score**
   * **Format**: Integer (e.g., "42")
   * **Color**: Green when increasing
   * **Update**: Updates immediately when snake eats food

**Moves Count**
   * **Format**: Integer (e.g., "156")
   * **Function**: Total number of moves executed
   * **Update**: Increments with each move

**Snake Length**
   * **Format**: Integer (e.g., "8")
   * **Function**: Current number of snake segments
   * **Update**: Increases when food is eaten

**Runtime**
   * **Format**: HH:MM:SS (e.g., "02:34:15")
   * **Function**: Total simulation runtime
   * **Update**: Updates every second during simulation

Message Log Panel
-----------------

Message Types
~~~~~~~~~~~~~

**Info Messages**
   * **Color**: Default (cyan)
   * **Examples**: "Connected to server", "Simulation started"
   * **Format**: [HH:MM:SS] Message text

**Warning Messages**
   * **Color**: Yellow
   * **Examples**: "Connection timeout", "Invalid configuration"
   * **Format**: [HH:MM:SS] Warning text

**Error Messages**
   * **Color**: Red
   * **Examples**: "Connection failed", "Server error"
   * **Format**: [HH:MM:SS] Error text

Log Features
~~~~~~~~~~~~

**Auto-Scroll**
   * **Behavior**: Automatically scrolls to show newest messages
   * **Override**: Manual scrolling temporarily disables auto-scroll
   * **Resume**: Auto-scroll resumes when scrolled to bottom

**Message History**
   * **Capacity**: Stores last 1000 messages
   * **Persistence**: Messages cleared only on reset
   * **Search**: Use terminal search (Ctrl+F) to find messages

Keyboard Navigation
-------------------

Global Shortcuts
~~~~~~~~~~~~~~~~

* **Tab**: Move to next focusable element
* **Shift+Tab**: Move to previous focusable element
* **Enter**: Activate focused button or confirm input
* **Escape**: Cancel current action or clear focus
* **Ctrl+C**: Quit application (with confirmation)

Input Field Navigation
~~~~~~~~~~~~~~~~~~~~~~

* **Arrow Keys**: Move cursor within input field
* **Home**: Move to beginning of input
* **End**: Move to end of input
* **Backspace**: Delete character before cursor
* **Delete**: Delete character after cursor
* **Ctrl+A**: Select all text
* **Ctrl+X**: Cut selected text
* **Ctrl+C**: Copy selected text (when text is selected)
* **Ctrl+V**: Paste text

Button Activation
~~~~~~~~~~~~~~~~~

* **Space**: Activate focused button (alternative to Enter)
* **Arrow Keys**: Navigate between buttons in same row
* **Tab**: Move to next button or control

Mouse Support
-------------

Button Interaction
~~~~~~~~~~~~~~~~~~

* **Left Click**: Activate button
* **Hover**: Visual feedback (if terminal supports it)
* **Right Click**: No action (reserved for future use)

Input Field Interaction
~~~~~~~~~~~~~~~~~~~~~~~

* **Left Click**: Focus input field and position cursor
* **Double Click**: Select word
* **Triple Click**: Select all text
* **Drag**: Select text range

Scroll Areas
~~~~~~~~~~~~

* **Mouse Wheel**: Scroll vertically in game board and message log
* **Shift+Mouse Wheel**: Scroll horizontally (if applicable)
* **Click and Drag**: Scroll by dragging (if terminal supports it)

State-Based Behavior
--------------------

Button Visibility
~~~~~~~~~~~~~~~~~

The interface automatically shows/hides buttons based on simulation state:

**Idle State**
   * **Visible**: Start, Reset
   * **Hidden**: Stop, Pause, Resume

**Running State**
   * **Visible**: Stop, Pause, Reset
   * **Hidden**: Start, Resume

**Paused State**
   * **Visible**: Stop, Resume, Reset
   * **Hidden**: Start, Pause

**Stopped State**
   * **Visible**: Start, Reset
   * **Hidden**: Stop, Pause, Resume

Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~~

**Grid Size Validation**
   * **Valid**: "10,10", "20,15", "8,8"
   * **Invalid**: "10", "abc,def", "0,5", "100,100"
   * **Feedback**: Error message in log for invalid formats

**Move Budget Validation**
   * **Valid**: "50", "100", "500"
   * **Invalid**: "0", "-10", "abc", "1001"
   * **Feedback**: Error message in log for invalid values

Accessibility Features
----------------------

Visual Accessibility
~~~~~~~~~~~~~~~~~~~~

* **High Contrast**: Clear color distinctions for all elements
* **Color Coding**: Consistent color scheme throughout interface
* **Text Labels**: All interactive elements have clear labels
* **Visual Hierarchy**: Clear organization and grouping

Keyboard Accessibility
~~~~~~~~~~~~~~~~~~~~~~

* **Full Keyboard Navigation**: All functions accessible via keyboard
* **Focus Indicators**: Clear visual indication of focused elements
* **Logical Tab Order**: Intuitive navigation sequence
* **Keyboard Shortcuts**: Standard shortcuts where applicable

Screen Reader Support
~~~~~~~~~~~~~~~~~~~~~

* **Text Descriptions**: All visual elements have text equivalents
* **State Announcements**: Changes in simulation state are announced
* **Progress Updates**: Score and status changes are accessible
* **Error Messages**: All errors are provided as text

Customization Options
---------------------

Terminal Compatibility
~~~~~~~~~~~~~~~~~~~~~~

* **Color Support**: Automatically detects and adapts to terminal capabilities
* **Size Adaptation**: Works with various terminal sizes (minimum 80x24)
* **Font Support**: Compatible with monospace fonts
* **Unicode Support**: Uses standard ASCII characters for maximum compatibility

Performance Tuning
~~~~~~~~~~~~~~~~~~~

* **Update Frequency**: Status updates limited to 1Hz for performance
* **Rendering Optimization**: Only redraws changed areas
* **Memory Management**: Efficient memory usage for long-running sessions
* **Connection Tuning**: Configurable timeouts and retry intervals

This comprehensive control reference ensures users can effectively navigate and control all aspects of the AI Hydra TUI interface.