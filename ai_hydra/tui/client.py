"""
AI Hydra TUI Client

Terminal user interface for the AI Hydra simulation system.
Uses the correct AI Hydra ZeroMQ protocol for communication.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

import zmq
import zmq.asyncio
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import var
from textual.theme import Theme
from textual.widgets import Button, Input, Label, Log, Static

from ai_hydra.tui.game_board import HydraGameBoard
from ai_hydra.zmq_protocol import ZMQMessage, MessageType


# AI Hydra Theme (adapted from ai_snake_lab)
HYDRA_THEME = Theme(
    name="hydra_dark",
    primary="#88C0D0",
    secondary="#1f6a83ff", 
    accent="#B48EAD",
    foreground="#31b8e6",
    background="black",
    success="#A3BE8C",
    warning="#EBCB8B", 
    error="#BF616A",
    surface="#111111",
    panel="#000000",
    dark=True,
    variables={
        "block-cursor-text-style": "none",
        "footer-key-foreground": "#88C0D0",
        "input-selection-background": "#81a1c1 35%",
    },
)


class HydraClient(App):
    """TUI client for AI Hydra simulation system using correct ZeroMQ protocol."""
    
    TITLE = "AI Hydra - Snake Game AI Monitor"
    CSS_PATH = "hydra_client.tcss"  # Textual will look in the same directory as this file
    
    # Reactive variables for real-time updates
    simulation_state = var("idle")
    game_score = var(0)
    snake_length = var(3)
    moves_count = var(0)
    runtime_seconds = var(0)
    
    def __init__(self, server_address: str = "tcp://localhost:5555"):
        super().__init__()
        self.server_address = server_address
        
        # ZeroMQ setup - will be initialized in connect_to_server
        self.context = None
        self.socket = None
        self.client_id = f"hydra-client-{uuid.uuid4().hex[:8]}"
        self.is_connected = False
        
        # Background tasks
        self.status_poll_task = None
        
        # Game board
        self.game_board = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    async def on_mount(self) -> None:
        """Initialize application on startup."""
        # Register theme
        self.register_theme(HYDRA_THEME)
        self.theme = "hydra_dark"
        
        # Set initial state
        self.add_class("idle")
        
        # Connect to server and start background tasks
        await self.connect_to_server()
        
    def compose(self) -> ComposeResult:
        """Create the main UI layout."""
        # Title
        yield Label(self.TITLE, id="title")
        
        # Control panel
        yield Vertical(
            Horizontal(
                Button("Start", id="btn_start", variant="success", compact=True),
                Button("Stop", id="btn_stop", variant="error", compact=True), 
                Button("Pause", id="btn_pause", variant="warning", compact=True),
                Button("Resume", id="btn_resume", variant="primary", compact=True),
                Button("Reset", id="btn_reset", variant="default", compact=True),
                classes="button_row"
            ),
            Horizontal(
                Label("Grid Size:", classes="config_label"),
                Input(value="20,20", id="grid_size", classes="config_input", compact=True)
            ),
            Horizontal(
                Label("Move Budget:", classes="config_label"),
                Input(value="100", id="move_budget", classes="config_input", compact=True)
            ),
            id="control_panel"
        )
        
        # Game board
        self.game_board = HydraGameBoard(board_size=(20, 20), id="game_board")
        yield Vertical(
            self.game_board,
            id="game_box"
        )
        
        # Status display
        yield Vertical(
            Label("Status", classes="section_header"),
            Horizontal(
                Label("State:", classes="status_label"),
                Label("Idle", id="sim_state", classes="status_value")
            ),
            Horizontal(
                Label("Score:", classes="status_label"),
                Label("0", id="game_score", classes="status_value")
            ),
            Horizontal(
                Label("Moves:", classes="status_label"),
                Label("0", id="moves_count", classes="status_value")
            ),
            Horizontal(
                Label("Snake Length:", classes="status_label"),
                Label("3", id="snake_length", classes="status_value")
            ),
            Horizontal(
                Label("Runtime:", classes="status_label"),
                Label("00:00:00", id="runtime", classes="status_value")
            ),
            id="status_panel"
        )
        
        # Log output
        yield Vertical(
            Label("Messages", classes="section_header"),
            Log(highlight=False, auto_scroll=True, id="message_log"),
            id="log_panel"
        )
    
    async def connect_to_server(self) -> bool:
        """Establish connection to the AI Hydra server."""
        try:
            # Use async ZMQ context and socket
            self.context = zmq.asyncio.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(self.server_address)
            
            # Test connection with get_status
            response = await self.send_command(MessageType.GET_STATUS, {})
            
            if response:
                self.is_connected = True
                self.log_message(f"Connected to server at {self.server_address}")
                
                # Start background status polling
                self.status_poll_task = asyncio.create_task(self.poll_status())
                
                return True
            else:
                self.log_message("Failed to get response from server", level="error")
                return False
            
        except Exception as e:
            self.log_message(f"Connection failed: {e}", level="error")
            self.is_connected = False
            return False
    
    async def send_command(self, message_type: MessageType, data: Dict[str, Any]) -> Optional[ZMQMessage]:
        """Send a command to the server and wait for response."""
        if not self.socket:
            self.log_message("No socket available", level="error")
            return None
            
        try:
            # Create AI Hydra protocol message
            request_id = str(uuid.uuid4())
            message = ZMQMessage.create_command(
                message_type=message_type,
                client_id=self.client_id,
                request_id=request_id,
                data=data
            )
            
            # Send message
            await self.socket.send_string(message.to_json())
            
            # Wait for response with timeout
            if await self.socket.poll(timeout=5000):  # 5 second timeout
                response_data = await self.socket.recv_string()
                response = ZMQMessage.from_json(response_data)
                return response
            else:
                self.log_message(f"Timeout waiting for response to {message_type.value}", level="warning")
                return None
                
        except Exception as e:
            self.log_message(f"Error sending command {message_type.value}: {e}", level="error")
            return None
    
    async def poll_status(self) -> None:
        """Poll server status periodically."""
        while self.is_connected:
            try:
                response = await self.send_command(MessageType.GET_STATUS, {})
                if response and response.data:
                    await self.process_status_update(response.data)
                
                await asyncio.sleep(1.0)  # Poll every second
                    
            except Exception as e:
                self.log_message(f"Status polling error: {e}", level="error")
                await asyncio.sleep(1.0)
    
    async def process_status_update(self, status_data: Dict[str, Any]) -> None:
        """Process status update from server."""
        try:
            # Update reactive variables based on AI Hydra protocol
            if "simulation_status" in status_data:
                self.simulation_state = status_data["simulation_status"]
                
            if "game_state" in status_data:
                game_state = status_data["game_state"]
                self.game_score = game_state.get("score", 0)
                
                snake_body = game_state.get("snake_body", [])
                self.snake_length = len(snake_body)
                
                self.moves_count = game_state.get("moves_count", 0)
                
                # Update game board
                if self.game_board:
                    await self.game_board.update_game_state(game_state)
                    
            if "performance" in status_data:
                performance = status_data["performance"]
                # Could extract runtime from performance metrics if available
                
        except Exception as e:
            self.log_message(f"Error processing status update: {e}", level="error")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn_start":
            await self.start_simulation()
        elif button_id == "btn_stop":
            await self.stop_simulation()
        elif button_id == "btn_pause":
            await self.pause_simulation()
        elif button_id == "btn_resume":
            await self.resume_simulation()
        elif button_id == "btn_reset":
            await self.reset_simulation()
    
    async def start_simulation(self) -> None:
        """Start the simulation."""
        config = self.get_simulation_config()
        response = await self.send_command(MessageType.START_SIMULATION, {"config": config})
        
        if response and response.message_type == MessageType.SIMULATION_STARTED:
            self.simulation_state = "running"
            self.log_message("Simulation started")
        else:
            error_msg = "Unknown error"
            if response and response.data:
                error_msg = response.data.get("error_message", error_msg)
            self.log_message(f"Failed to start simulation: {error_msg}", level="error")
    
    async def stop_simulation(self) -> None:
        """Stop the simulation."""
        response = await self.send_command(MessageType.STOP_SIMULATION, {})
        
        if response and response.message_type == MessageType.SIMULATION_STOPPED:
            self.simulation_state = "stopped"
            self.log_message("Simulation stopped")
        else:
            error_msg = "Unknown error"
            if response and response.data:
                error_msg = response.data.get("error_message", error_msg)
            self.log_message(f"Failed to stop simulation: {error_msg}", level="error")
    
    async def pause_simulation(self) -> None:
        """Pause the simulation."""
        response = await self.send_command(MessageType.PAUSE_SIMULATION, {})
        
        if response and response.message_type == MessageType.SIMULATION_PAUSED:
            self.simulation_state = "paused"
            self.log_message("Simulation paused")
        else:
            error_msg = "Unknown error"
            if response and response.data:
                error_msg = response.data.get("error_message", error_msg)
            self.log_message(f"Failed to pause simulation: {error_msg}", level="error")
    
    async def resume_simulation(self) -> None:
        """Resume the simulation."""
        response = await self.send_command(MessageType.RESUME_SIMULATION, {})
        
        if response and response.message_type == MessageType.SIMULATION_RESUMED:
            self.simulation_state = "running"
            self.log_message("Simulation resumed")
        else:
            error_msg = "Unknown error"
            if response and response.data:
                error_msg = response.data.get("error_message", error_msg)
            self.log_message(f"Failed to resume simulation: {error_msg}", level="error")
    
    async def reset_simulation(self) -> None:
        """Reset the simulation."""
        response = await self.send_command(MessageType.RESET_SIMULATION, {})
        
        if response and response.message_type == MessageType.SIMULATION_RESET:
            self.simulation_state = "idle"
            self.game_score = 0
            self.snake_length = 3
            self.moves_count = 0
            self.runtime_seconds = 0
            
            # Clear game board
            if self.game_board:
                await self.game_board.reset()
                
            # Clear log
            log_widget = self.query_one("#message_log", Log)
            log_widget.clear()
            
            self.log_message("Simulation reset")
        else:
            error_msg = "Unknown error"
            if response and response.data:
                error_msg = response.data.get("error_message", error_msg)
            self.log_message(f"Failed to reset simulation: {error_msg}", level="error")
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """Get simulation configuration from UI inputs."""
        try:
            grid_size_str = self.query_one("#grid_size", Input).value
            grid_width, grid_height = map(int, grid_size_str.split(","))
            
            move_budget = int(self.query_one("#move_budget", Input).value)
            
            return {
                "grid_size": [grid_width, grid_height],
                "move_budget": move_budget,
                "initial_snake_length": 3,
                "random_seed": 42
            }
        except Exception as e:
            self.log_message(f"Error parsing configuration: {e}", level="error")
            return {
                "grid_size": [20, 20],
                "move_budget": 100,
                "initial_snake_length": 3,
                "random_seed": 42
            }
    
    def log_message(self, message: str, level: str = "info") -> None:
        """Log a message to both the UI and logger."""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Log to UI
        try:
            log_widget = self.query_one("#message_log", Log)
            if level == "error":
                log_widget.write_line(f"[red]{formatted_message}[/red]")
            elif level == "warning":
                log_widget.write_line(f"[yellow]{formatted_message}[/yellow]")
            else:
                log_widget.write_line(formatted_message)
        except Exception:
            pass  # UI might not be ready yet
        
        # Log to logger
        if level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    # Reactive variable watchers
    def watch_simulation_state(self, old_state: str, new_state: str) -> None:
        """React to simulation state changes."""
        # Update UI state classes
        if old_state:
            self.remove_class(old_state)
        self.add_class(new_state)
        
        # Update status display
        try:
            state_label = self.query_one("#sim_state", Label)
            state_label.update(new_state.title())
        except Exception:
            pass
    
    def watch_game_score(self, old_score: int, new_score: int) -> None:
        """React to score changes."""
        try:
            score_label = self.query_one("#game_score", Label)
            score_label.update(str(new_score))
        except Exception:
            pass
    
    def watch_snake_length(self, old_length: int, new_length: int) -> None:
        """React to snake length changes."""
        try:
            length_label = self.query_one("#snake_length", Label)
            length_label.update(str(new_length))
        except Exception:
            pass
    
    def watch_moves_count(self, old_count: int, new_count: int) -> None:
        """React to moves count changes."""
        try:
            moves_label = self.query_one("#moves_count", Label)
            moves_label.update(str(new_count))
        except Exception:
            pass
    
    def watch_runtime_seconds(self, old_time: int, new_time: int) -> None:
        """React to runtime changes."""
        try:
            runtime_label = self.query_one("#runtime", Label)
            runtime_label.update(self.format_runtime(new_time))
        except Exception:
            pass
    
    def format_runtime(self, seconds: int) -> str:
        """Format runtime seconds as HH:MM:SS."""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    async def action_quit(self) -> None:
        """Clean shutdown."""
        self.is_connected = False
        
        # Cancel background tasks
        if self.status_poll_task:
            self.status_poll_task.cancel()
        
        # Close socket
        if self.socket:
            self.socket.close()
        
        # Close context
        if self.context:
            self.context.term()
        
        await super().action_quit()


def main():
    """Main entry point for the TUI client."""
    parser = argparse.ArgumentParser(description="AI Hydra TUI Client")
    parser.add_argument(
        "--server", 
        default="tcp://localhost:5555",
        help="ZeroMQ server address (default: tcp://localhost:5555)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print(f"Connecting to AI Hydra server at {args.server}")
    if args.verbose:
        print("Verbose logging enabled")
    
    # Create and run the app
    app = HydraClient(server_address=args.server)
    app.run()


if __name__ == "__main__":
    main()