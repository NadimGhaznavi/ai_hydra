# AI Hydra TUI Client

A minimal terminal user interface for the AI Hydra simulation system built with [Textual](https://textual.textualize.io/).

## Features

- **Real-time Game Visualization**: Watch the Snake game play in a colorful terminal interface
- **Simulation Control**: Start, stop, pause, resume, and reset simulations
- **Live Status Monitoring**: See current score, moves, snake length, and runtime
- **Configuration**: Adjust grid size and move budget
- **Message Log**: View real-time messages and errors

## Installation

Install the TUI dependencies:

```bash
pip install -e .[tui]
```

## Usage

### Basic Usage

Start the TUI client (assumes server is running on localhost:5555):

```bash
ai-hydra-tui
```

### Custom Server Address

Connect to a different server:

```bash
ai-hydra-tui --server tcp://192.168.1.100:5555
```

### Verbose Logging

Enable detailed logging:

```bash
ai-hydra-tui --verbose
```

### Module Usage

You can also run it as a Python module:

```bash
python -m ai_hydra.tui.client --server tcp://localhost:5555
```

## Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Hydra - Snake Game AI Monitor         │
├─────────────────┬───────────────────────────┬───────────────┤
│   Control       │                           │    Status     │
│   Panel         │        Game Board         │    Panel      │
│                 │                           │               │
│ [Start] [Stop]  │    ████████████████       │ State: Running│
│ [Pause][Resume] │    █▓▓▓▓▓▓▓▓▓▓▓▓▓█       │ Score: 42     │
│ [Reset]         │    █▓░░░░░░░░░░░░▓█       │ Moves: 156    │
│                 │    █▓░░░●░░░░░░░▓█       │ Length: 8     │
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
```

## Controls

### Keyboard Shortcuts

- `Ctrl+C` or `q`: Quit the application
- `Tab`: Navigate between UI elements
- `Enter`: Activate buttons
- `Arrow Keys`: Navigate in input fields

### Button Controls

- **Start**: Begin a new simulation with current configuration
- **Stop**: Stop the current simulation
- **Pause**: Pause the running simulation
- **Resume**: Resume a paused simulation  
- **Reset**: Reset simulation and clear all data

### Configuration

- **Grid Size**: Set the game board dimensions (format: "width,height")
- **Move Budget**: Set the maximum number of moves per decision cycle

## Server Communication

The TUI client communicates with the AI Hydra server using ZeroMQ:

- **Protocol**: REQ/REP pattern for commands
- **Message Format**: JSON messages with client_id, command, and data
- **Heartbeat**: Periodic heartbeat messages to maintain connection
- **Auto-reconnect**: Automatic reconnection on connection loss (planned)

### Supported Commands

- `ping`: Test server connectivity
- `start_simulation`: Start simulation with configuration
- `stop_simulation`: Stop current simulation
- `pause_simulation`: Pause running simulation
- `resume_simulation`: Resume paused simulation
- `reset_simulation`: Reset simulation state
- `get_status`: Get current simulation status
- `heartbeat`: Maintain connection

## Development

### Architecture

The TUI client follows a reactive architecture:

- **HydraClient**: Main Textual app with reactive variables
- **HydraGameBoard**: Custom widget for game visualization
- **ZeroMQ Communication**: Async message handling
- **CSS Styling**: Textual CSS for theming

### Key Components

1. **client.py**: Main application class and entry point
2. **game_board.py**: Game board visualization widget
3. **hydra_client.tcss**: CSS styling for the interface

### Extending the TUI

To add new features:

1. Add reactive variables to `HydraClient` class
2. Create new widgets in the `compose()` method
3. Add message handlers in `process_status_update()`
4. Update CSS styling in `hydra_client.tcss`

## Troubleshooting

### Connection Issues

If you can't connect to the server:

1. Verify the server is running: `ai-hydra-server`
2. Check the server address and port
3. Ensure no firewall is blocking the connection
4. Try verbose mode: `ai-hydra-tui --verbose`

### Display Issues

If the interface looks wrong:

1. Ensure your terminal supports colors
2. Try resizing the terminal window
3. Check terminal compatibility with Textual

### Performance Issues

If the interface is slow:

1. Reduce the status update frequency
2. Use a smaller grid size
3. Check system resources

## Future Enhancements

Planned features for future versions:

- **Performance Metrics**: CPU, memory, and decision rate monitoring
- **History Tracking**: Game score history and high score leaderboard
- **Advanced Configuration**: Neural network settings and training parameters
- **Multi-client Support**: Multiple TUI clients connected to one server
- **Data Export**: Export simulation data and statistics
- **Themes**: Multiple color themes and customization options

## Contributing

To contribute to the TUI client:

1. Follow the existing code style and patterns
2. Add tests for new functionality
3. Update documentation
4. Test with different terminal environments

## License

GPL-3.0 - See the main project LICENSE file for details.