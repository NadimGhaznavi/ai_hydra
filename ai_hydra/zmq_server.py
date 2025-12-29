"""
ZeroMQ Server for AI Hydra.

This module provides a headless ZeroMQ server that wraps the HydraMgr and
exposes its functionality via message-based communication.
"""

import asyncio
import threading
import time
import uuid
import psutil
import logging
from typing import Dict, Set, Optional, Any
from dataclasses import asdict

import zmq
import zmq.asyncio

from .zmq_protocol import (
    ZMQMessage, MessageType, MessageBuilder, MessageValidator,
    GameStateData, PerformanceMetrics, SimulationConfig as ZMQSimulationConfig
)
from .hydra_mgr import HydraMgr
from .config import SimulationConfig, NetworkConfig, LoggingConfig
from .models import GameBoard


class SimulationState:
    """Tracks the current state of the simulation."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class ZMQServer:
    """
    Headless ZeroMQ server for the AI Hydra.
    
    This server provides a message-based interface to control and monitor
    the Snake Game AI simulation system. It runs completely headless and
    communicates only via ZeroMQ messages.
    """
    
    def __init__(self, bind_address: str = "tcp://*:5555",
                 heartbeat_interval: float = 5.0):
        """
        Initialize the ZeroMQ server.
        
        Args:
            bind_address: ZeroMQ bind address (e.g., "tcp://*:5555")
            heartbeat_interval: Seconds between heartbeat broadcasts
        """
        self.bind_address = bind_address
        self.heartbeat_interval = heartbeat_interval
        
        # ZeroMQ context and sockets
        self.context = zmq.asyncio.Context()
        self.socket = None
        
        # Server state
        self.server_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.is_running = False
        self.connected_clients: Set[str] = set()
        
        # Simulation components
        self.hydra_mgr: Optional[HydraMgr] = None
        self.simulation_state = SimulationState.IDLE
        self.simulation_thread: Optional[threading.Thread] = None
        self.simulation_config: Optional[SimulationConfig] = None
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics.create_empty()
        self.last_status_update = time.time()
        self.status_update_interval = 1.0  # seconds
        
        # Logging
        self.logger = logging.getLogger("zmq_server")
        self.logger.setLevel(logging.INFO)
        
        # Message statistics
        self.messages_received = 0
        self.messages_sent = 0
        
        self.logger.info(f"ZMQ Server initialized with ID: {self.server_id}")
    
    async def start(self) -> None:
        """Start the ZeroMQ server."""
        self.logger.info(f"Starting ZMQ server on {self.bind_address}")
        
        # Create and bind socket
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.bind_address)
        
        self.is_running = True
        
        # Start background tasks
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        status_task = asyncio.create_task(self._status_update_loop())
        message_task = asyncio.create_task(self._message_loop())
        
        self.logger.info("ZMQ server started successfully")
        
        try:
            # Wait for all tasks
            await asyncio.gather(heartbeat_task, status_task, message_task)
        except asyncio.CancelledError:
            self.logger.info("ZMQ server tasks cancelled")
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the ZeroMQ server."""
        self.logger.info("Stopping ZMQ server")
        
        self.is_running = False
        
        # Stop simulation if running
        if self.simulation_state == SimulationState.RUNNING:
            await self._stop_simulation()
        
        # Close socket and context
        if self.socket:
            self.socket.close()
        self.context.term()
        
        self.logger.info("ZMQ server stopped")
    
    async def _message_loop(self) -> None:
        """Main message processing loop."""
        while self.is_running:
            try:
                # Wait for message with timeout
                if await self.socket.poll(timeout=1000):  # 1 second timeout
                    message_data = await self.socket.recv_string()
                    self.messages_received += 1
                    
                    # Process message
                    response = await self._process_message(message_data)
                    
                    # Send response
                    await self.socket.send_string(response.to_json())
                    self.messages_sent += 1
                    
            except zmq.error.ContextTerminated:
                break
            except Exception as e:
                self.logger.error(f"Error in message loop: {e}")
                # Send error response if possible
                try:
                    error_response = MessageBuilder.error_occurred(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        component="zmq_server",
                        recoverable=True
                    )
                    await self.socket.send_string(error_response.to_json())
                    self.messages_sent += 1
                except:
                    pass
    
    async def _process_message(self, message_data: str) -> ZMQMessage:
        """
        Process an incoming message and return a response.
        
        Args:
            message_data: JSON string containing the message
            
        Returns:
            ZMQMessage: Response message
        """
        try:
            # Parse message
            message = ZMQMessage.from_json(message_data)
            
            # Validate message
            is_valid, error_msg = MessageValidator.validate_message(message)
            if not is_valid:
                return ZMQMessage.create_response(
                    MessageType.ERROR_OCCURRED,
                    message.request_id or "unknown",
                    {
                        "error_type": "ValidationError",
                        "error_message": error_msg,
                        "recoverable": False
                    }
                )
            
            # Track client
            if message.client_id:
                self.connected_clients.add(message.client_id)
            
            # Route message to appropriate handler
            return await self._route_message(message)
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return MessageBuilder.error_occurred(
                error_type=type(e).__name__,
                error_message=str(e),
                component="message_processor",
                recoverable=True
            )
    
    async def _route_message(self, message: ZMQMessage) -> ZMQMessage:
        """Route message to appropriate handler based on type."""
        handlers = {
            MessageType.START_SIMULATION: self._handle_start_simulation,
            MessageType.STOP_SIMULATION: self._handle_stop_simulation,
            MessageType.PAUSE_SIMULATION: self._handle_pause_simulation,
            MessageType.RESUME_SIMULATION: self._handle_resume_simulation,
            MessageType.GET_STATUS: self._handle_get_status,
            MessageType.UPDATE_CONFIG: self._handle_update_config,
            MessageType.RESET_SIMULATION: self._handle_reset_simulation,
        }
        
        handler = handlers.get(message.message_type)
        if handler:
            return await handler(message)
        else:
            return ZMQMessage.create_response(
                MessageType.ERROR_OCCURRED,
                message.request_id or "unknown",
                {
                    "error_type": "UnsupportedMessage",
                    "error_message": f"Unsupported message type: {message.message_type.value}",
                    "recoverable": False
                }
            )
    
    async def _handle_start_simulation(self, message: ZMQMessage) -> ZMQMessage:
        """Handle start simulation command."""
        if self.simulation_state == SimulationState.RUNNING:
            return ZMQMessage.create_response(
                MessageType.ERROR_OCCURRED,
                message.request_id,
                {
                    "error_type": "SimulationAlreadyRunning",
                    "error_message": "Simulation is already running",
                    "recoverable": False
                }
            )
        
        try:
            # Extract configuration
            config_data = message.data.get("config", {})
            
            # Create simulation configuration
            self.simulation_config = SimulationConfig(
                grid_size=tuple(config_data.get("grid_size", [20, 20])),
                initial_snake_length=config_data.get("initial_snake_length", 3),
                move_budget=config_data.get("move_budget", 100),
                random_seed=config_data.get("random_seed", 42),
                nn_enabled=config_data.get("nn_enabled", True),
                max_tree_depth=config_data.get("max_tree_depth"),
                food_reward=config_data.get("food_reward", 10),
                collision_penalty=config_data.get("collision_penalty", -10),
                empty_move_reward=config_data.get("empty_move_reward", 0)
            )
            
            # Start simulation in background thread
            self.simulation_state = SimulationState.RUNNING
            self.simulation_thread = threading.Thread(
                target=self._run_simulation_thread,
                daemon=True
            )
            self.simulation_thread.start()
            
            self.logger.info("Simulation started")
            
            return ZMQMessage.create_response(
                MessageType.SIMULATION_STARTED,
                message.request_id,
                {"config": asdict(ZMQSimulationConfig(**config_data))}
            )
            
        except Exception as e:
            self.simulation_state = SimulationState.ERROR
            self.logger.error(f"Error starting simulation: {e}")
            return ZMQMessage.create_response(
                MessageType.ERROR_OCCURRED,
                message.request_id,
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "recoverable": True
                }
            )
    
    async def _handle_stop_simulation(self, message: ZMQMessage) -> ZMQMessage:
        """Handle stop simulation command."""
        await self._stop_simulation()
        
        return ZMQMessage.create_response(
            MessageType.SIMULATION_STOPPED,
            message.request_id,
            {"message": "Simulation stopped successfully"}
        )
    
    async def _handle_pause_simulation(self, message: ZMQMessage) -> ZMQMessage:
        """Handle pause simulation command."""
        if self.simulation_state == SimulationState.RUNNING:
            self.simulation_state = SimulationState.PAUSED
            self.logger.info("Simulation paused")
            
            return ZMQMessage.create_response(
                MessageType.SIMULATION_PAUSED,
                message.request_id,
                {"message": "Simulation paused successfully"}
            )
        else:
            return ZMQMessage.create_response(
                MessageType.ERROR_OCCURRED,
                message.request_id,
                {
                    "error_type": "InvalidState",
                    "error_message": f"Cannot pause simulation in state: {self.simulation_state}",
                    "recoverable": False
                }
            )
    
    async def _handle_resume_simulation(self, message: ZMQMessage) -> ZMQMessage:
        """Handle resume simulation command."""
        if self.simulation_state == SimulationState.PAUSED:
            self.simulation_state = SimulationState.RUNNING
            self.logger.info("Simulation resumed")
            
            return ZMQMessage.create_response(
                MessageType.SIMULATION_RESUMED,
                message.request_id,
                {"message": "Simulation resumed successfully"}
            )
        else:
            return ZMQMessage.create_response(
                MessageType.ERROR_OCCURRED,
                message.request_id,
                {
                    "error_type": "InvalidState",
                    "error_message": f"Cannot resume simulation in state: {self.simulation_state}",
                    "recoverable": False
                }
            )
    
    async def _handle_get_status(self, message: ZMQMessage) -> ZMQMessage:
        """Handle get status command."""
        # Get current game state if simulation is running
        game_state = None
        if self.hydra_mgr and hasattr(self.hydra_mgr, 'master_game') and self.hydra_mgr.master_game:
            current_board = self.hydra_mgr.master_game.get_current_board()
            game_state = GameStateData.from_game_board(
                current_board, 
                self.hydra_mgr.total_moves
            )
            game_state.is_game_over = self.hydra_mgr.master_game.is_terminal()
        
        status_data = {
            "server_id": self.server_id,
            "simulation_state": self.simulation_state,
            "uptime_seconds": time.time() - self.start_time,
            "connected_clients": len(self.connected_clients),
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "game_state": asdict(game_state) if game_state else None,
            "performance": asdict(self.performance_metrics)
        }
        
        return ZMQMessage.create_response(
            MessageType.STATUS_RESPONSE,
            message.request_id,
            status_data
        )
    
    async def _handle_update_config(self, message: ZMQMessage) -> ZMQMessage:
        """Handle update config command."""
        # For now, config updates are only allowed when simulation is not running
        if self.simulation_state == SimulationState.RUNNING:
            return ZMQMessage.create_response(
                MessageType.ERROR_OCCURRED,
                message.request_id,
                {
                    "error_type": "InvalidState",
                    "error_message": "Cannot update config while simulation is running",
                    "recoverable": False
                }
            )
        
        # Update would be implemented here
        return ZMQMessage.create_response(
            MessageType.CONFIG_UPDATED,
            message.request_id,
            {"message": "Configuration updated successfully"}
        )
    
    async def _handle_reset_simulation(self, message: ZMQMessage) -> ZMQMessage:
        """Handle reset simulation command."""
        await self._stop_simulation()
        
        # Reset state
        self.hydra_mgr = None
        self.performance_metrics = PerformanceMetrics.create_empty()
        
        return ZMQMessage.create_response(
            MessageType.SIMULATION_RESET,
            message.request_id,
            {"message": "Simulation reset successfully"}
        )
    
    def _run_simulation_thread(self) -> None:
        """Run the simulation in a background thread."""
        try:
            # Create HydraMgr
            self.hydra_mgr = HydraMgr(
                simulation_config=self.simulation_config,
                network_config=NetworkConfig(),
                logging_config=LoggingConfig(level="INFO")
            )
            
            # Run simulation
            result = self.hydra_mgr.run_simulation()
            
            # Update final state
            self.simulation_state = SimulationState.STOPPED
            
            # Broadcast game over
            game_over_msg = MessageBuilder.game_over(
                final_score=result.final_score,
                total_moves=result.total_moves,
                duration_seconds=result.game_length_seconds,
                decision_cycles=self.hydra_mgr.decision_cycle_count
            )
            
            # Note: In a real implementation, we'd need a way to broadcast
            # this message to all connected clients
            self.logger.info(f"Simulation completed: Score={result.final_score}, Moves={result.total_moves}")
            
        except Exception as e:
            self.simulation_state = SimulationState.ERROR
            self.logger.error(f"Simulation thread error: {e}")
    
    async def _stop_simulation(self) -> None:
        """Stop the current simulation."""
        if self.simulation_state in [SimulationState.RUNNING, SimulationState.PAUSED]:
            self.simulation_state = SimulationState.STOPPED
            
            # Wait for simulation thread to finish
            if self.simulation_thread and self.simulation_thread.is_alive():
                self.simulation_thread.join(timeout=5.0)
            
            self.logger.info("Simulation stopped")
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat messages."""
        while self.is_running:
            try:
                uptime = time.time() - self.start_time
                heartbeat_msg = MessageBuilder.heartbeat(
                    server_status=self.simulation_state,
                    uptime_seconds=uptime,
                    active_clients=len(self.connected_clients)
                )
                
                # Note: In a real implementation with PUB/SUB pattern,
                # we'd broadcast this to all subscribers
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
    
    async def _status_update_loop(self) -> None:
        """Send periodic status updates during simulation."""
        while self.is_running:
            try:
                if self.simulation_state == SimulationState.RUNNING:
                    current_time = time.time()
                    time_since_last_update = current_time - self.last_status_update
                    
                    if time_since_last_update >= self.status_update_interval:
                        await self._send_status_update()
                        self.last_status_update = current_time
                        
                        # Sleep for the remaining time until next update
                        next_sleep = self.status_update_interval
                    else:
                        # Sleep for the remaining time until next update is due
                        next_sleep = self.status_update_interval - time_since_last_update
                    
                    # Use a smaller sleep interval for better timing precision
                    sleep_interval = min(next_sleep, 0.1)  # Max 100ms sleep
                    await asyncio.sleep(sleep_interval)
                else:
                    # When not running simulation, check less frequently
                    await asyncio.sleep(0.5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in status update loop: {e}")
    
    async def _send_status_update(self) -> None:
        """Send a status update broadcast."""
        if not self.hydra_mgr or not hasattr(self.hydra_mgr, 'master_game'):
            return
        
        try:
            # Get current game state
            current_board = self.hydra_mgr.master_game.get_current_board()
            game_state = GameStateData.from_game_board(
                current_board,
                self.hydra_mgr.total_moves
            )
            game_state.is_game_over = self.hydra_mgr.master_game.is_terminal()
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Create status update message
            status_msg = MessageBuilder.status_update(
                game_state=game_state,
                performance=self.performance_metrics,
                simulation_status=self.simulation_state
            )
            
            # Note: In a real implementation with PUB/SUB pattern,
            # we'd broadcast this to all subscribers
            
        except Exception as e:
            self.logger.error(f"Error sending status update: {e}")
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics based on current simulation state."""
        if not self.hydra_mgr:
            return
        
        try:
            # Calculate decisions per second
            uptime = time.time() - self.start_time
            if uptime > 0:
                self.performance_metrics.decisions_per_second = (
                    self.hydra_mgr.decision_cycle_count / uptime
                )
            
            # Update other metrics
            self.performance_metrics.total_decision_cycles = self.hydra_mgr.decision_cycle_count
            
            # Get system resource usage
            process = psutil.Process()
            self.performance_metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.performance_metrics.cpu_usage_percent = process.cpu_percent()
            
            # Neural network accuracy (if available)
            if hasattr(self.hydra_mgr, 'oracle_trainer') and self.hydra_mgr.oracle_trainer:
                self.performance_metrics.neural_network_accuracy = (
                    self.hydra_mgr.oracle_trainer.get_accuracy()
                )
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")


# Main entry point for running the server
async def main():
    """Main entry point for the ZeroMQ server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Hydra ZeroMQ Server")
    parser.add_argument("--bind", default="tcp://*:5555", help="ZeroMQ bind address")
    parser.add_argument("--heartbeat", type=float, default=5.0, help="Heartbeat interval in seconds")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start server
    server = ZMQServer(bind_address=args.bind, heartbeat_interval=args.heartbeat)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logging.info("Received interrupt signal")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())