"""
ZeroMQ Message Protocol for AI Hydra.

This module defines the message protocol for communication between the headless
AI agent and external clients (TUI, monitoring tools, etc.).
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import json
import time
from .models import GameBoard, Move, MoveResult


class MessageType(Enum):
    """Types of messages in the protocol."""
    # Commands (Client -> Server)
    START_SIMULATION = "start_simulation"
    STOP_SIMULATION = "stop_simulation"
    PAUSE_SIMULATION = "pause_simulation"
    RESUME_SIMULATION = "resume_simulation"
    GET_STATUS = "get_status"
    UPDATE_CONFIG = "update_config"
    RESET_SIMULATION = "reset_simulation"
    
    # Responses (Server -> Client)
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_STOPPED = "simulation_stopped"
    SIMULATION_PAUSED = "simulation_paused"
    SIMULATION_RESUMED = "simulation_resumed"
    STATUS_RESPONSE = "status_response"
    CONFIG_UPDATED = "config_updated"
    SIMULATION_RESET = "simulation_reset"
    
    # Broadcasts (Server -> All Clients)
    STATUS_UPDATE = "status_update"
    GAME_STATE_UPDATE = "game_state_update"
    DECISION_CYCLE_COMPLETE = "decision_cycle_complete"
    GAME_OVER = "game_over"
    ERROR_OCCURRED = "error_occurred"
    
    # System Messages
    CLIENT_CONNECTED = "client_connected"
    CLIENT_DISCONNECTED = "client_disconnected"
    HEARTBEAT = "heartbeat"


@dataclass
class GameStateData:
    """Current game state information."""
    snake_head: tuple[int, int]
    snake_body: List[tuple[int, int]]
    direction: tuple[int, int]  # dx, dy
    food_position: tuple[int, int]
    score: int
    grid_size: tuple[int, int]
    moves_count: int
    is_game_over: bool
    
    @classmethod
    def from_game_board(cls, board: GameBoard, moves_count: int = 0) -> 'GameStateData':
        """Create GameStateData from GameBoard."""
        return cls(
            snake_head=(board.snake_head.x, board.snake_head.y),
            snake_body=[(pos.x, pos.y) for pos in board.snake_body],
            direction=(board.direction.dx, board.direction.dy),
            food_position=(board.food_position.x, board.food_position.y),
            score=board.score,
            grid_size=board.grid_size,
            moves_count=moves_count,
            is_game_over=False  # Will be set by caller
        )


@dataclass
class PerformanceMetrics:
    """Performance and efficiency metrics."""
    decisions_per_second: float
    budget_utilization: float  # 0.0 to 1.0
    average_tree_depth: float
    neural_network_accuracy: Optional[float]
    memory_usage_mb: float
    cpu_usage_percent: float
    total_decision_cycles: int
    paths_evaluated_per_cycle: float
    
    @classmethod
    def create_empty(cls) -> 'PerformanceMetrics':
        """Create empty metrics for initialization."""
        return cls(
            decisions_per_second=0.0,
            budget_utilization=0.0,
            average_tree_depth=0.0,
            neural_network_accuracy=None,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            total_decision_cycles=0,
            paths_evaluated_per_cycle=0.0
        )


@dataclass
class SimulationConfig:
    """Simulation configuration parameters."""
    grid_size: tuple[int, int] = (20, 20)
    initial_snake_length: int = 3
    move_budget: int = 100
    random_seed: int = 42
    nn_enabled: bool = True
    max_tree_depth: Optional[int] = None
    food_reward: int = 10
    collision_penalty: int = -10
    empty_move_reward: int = 0


@dataclass
class ZMQMessage:
    """Base message structure for ZeroMQ communication."""
    message_type: MessageType
    timestamp: float
    client_id: Optional[str] = None
    request_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        message_dict = {
            "message_type": self.message_type.value,
            "timestamp": self.timestamp,
            "client_id": self.client_id,
            "request_id": self.request_id,
            "data": self.data or {}
        }
        return json.dumps(message_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ZMQMessage':
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls(
            message_type=MessageType(data["message_type"]),
            timestamp=data["timestamp"],
            client_id=data.get("client_id"),
            request_id=data.get("request_id"),
            data=data.get("data", {})
        )
    
    @classmethod
    def create_command(cls, message_type: MessageType, client_id: str, 
                      request_id: str, data: Optional[Dict[str, Any]] = None) -> 'ZMQMessage':
        """Create a command message."""
        return cls(
            message_type=message_type,
            timestamp=time.time(),
            client_id=client_id,
            request_id=request_id,
            data=data
        )
    
    @classmethod
    def create_response(cls, message_type: MessageType, request_id: str,
                       data: Optional[Dict[str, Any]] = None) -> 'ZMQMessage':
        """Create a response message."""
        return cls(
            message_type=message_type,
            timestamp=time.time(),
            request_id=request_id,
            data=data
        )
    
    @classmethod
    def create_broadcast(cls, message_type: MessageType,
                        data: Optional[Dict[str, Any]] = None) -> 'ZMQMessage':
        """Create a broadcast message."""
        return cls(
            message_type=message_type,
            timestamp=time.time(),
            data=data
        )


class MessageBuilder:
    """Helper class for building common message types."""
    
    @staticmethod
    def status_update(game_state: GameStateData, performance: PerformanceMetrics,
                     simulation_status: str) -> ZMQMessage:
        """Build a status update broadcast message."""
        return ZMQMessage.create_broadcast(
            MessageType.STATUS_UPDATE,
            {
                "simulation_status": simulation_status,
                "game_state": asdict(game_state),
                "performance": asdict(performance)
            }
        )
    
    @staticmethod
    def game_state_update(game_state: GameStateData) -> ZMQMessage:
        """Build a game state update broadcast message."""
        return ZMQMessage.create_broadcast(
            MessageType.GAME_STATE_UPDATE,
            {"game_state": asdict(game_state)}
        )
    
    @staticmethod
    def decision_cycle_complete(cycle_number: int, optimal_move: str,
                               budget_used: int, paths_evaluated: int,
                               tree_depth: int) -> ZMQMessage:
        """Build a decision cycle completion broadcast message."""
        return ZMQMessage.create_broadcast(
            MessageType.DECISION_CYCLE_COMPLETE,
            {
                "cycle_number": cycle_number,
                "optimal_move": optimal_move,
                "budget_used": budget_used,
                "paths_evaluated": paths_evaluated,
                "tree_depth": tree_depth
            }
        )
    
    @staticmethod
    def game_over(final_score: int, total_moves: int, duration_seconds: float,
                 decision_cycles: int) -> ZMQMessage:
        """Build a game over broadcast message."""
        return ZMQMessage.create_broadcast(
            MessageType.GAME_OVER,
            {
                "final_score": final_score,
                "total_moves": total_moves,
                "duration_seconds": duration_seconds,
                "decision_cycles": decision_cycles
            }
        )
    
    @staticmethod
    def error_occurred(error_type: str, error_message: str,
                      component: str, recoverable: bool) -> ZMQMessage:
        """Build an error broadcast message."""
        return ZMQMessage.create_broadcast(
            MessageType.ERROR_OCCURRED,
            {
                "error_type": error_type,
                "error_message": error_message,
                "component": component,
                "recoverable": recoverable
            }
        )
    
    @staticmethod
    def simulation_started(config: SimulationConfig) -> ZMQMessage:
        """Build a simulation started response message."""
        return ZMQMessage.create_broadcast(
            MessageType.SIMULATION_STARTED,
            {"config": asdict(config)}
        )
    
    @staticmethod
    def heartbeat(server_status: str, uptime_seconds: float,
                 active_clients: int) -> ZMQMessage:
        """Build a heartbeat broadcast message."""
        return ZMQMessage.create_broadcast(
            MessageType.HEARTBEAT,
            {
                "server_status": server_status,
                "uptime_seconds": uptime_seconds,
                "active_clients": active_clients
            }
        )


class MessageValidator:
    """Validates incoming messages for correctness."""
    
    REQUIRED_FIELDS = {
        MessageType.START_SIMULATION: ["config"],
        MessageType.UPDATE_CONFIG: ["config"],
        MessageType.GET_STATUS: [],
        MessageType.STOP_SIMULATION: [],
        MessageType.PAUSE_SIMULATION: [],
        MessageType.RESUME_SIMULATION: [],
        MessageType.RESET_SIMULATION: []
    }
    
    @classmethod
    def validate_message(cls, message: ZMQMessage) -> tuple[bool, Optional[str]]:
        """
        Validate a message for correctness.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        # Check if message type is supported
        if message.message_type not in cls.REQUIRED_FIELDS:
            return False, f"Unsupported message type: {message.message_type.value}"
        
        # Check required fields
        required_fields = cls.REQUIRED_FIELDS[message.message_type]
        message_data = message.data or {}
        
        for field in required_fields:
            if field not in message_data:
                return False, f"Missing required field: {field}"
        
        # Validate specific message types
        if message.message_type == MessageType.START_SIMULATION:
            return cls._validate_start_simulation(message_data)
        elif message.message_type == MessageType.UPDATE_CONFIG:
            return cls._validate_update_config(message_data)
        
        return True, None
    
    @classmethod
    def _validate_start_simulation(cls, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate start simulation message data."""
        config = data.get("config", {})
        
        # Validate grid size
        grid_size = config.get("grid_size")
        if not isinstance(grid_size, list) or len(grid_size) != 2:
            return False, "grid_size must be a list of two integers"
        
        if not all(isinstance(x, int) and x >= 5 for x in grid_size):
            return False, "grid_size values must be integers >= 5"
        
        # Validate move budget
        move_budget = config.get("move_budget", 100)
        if not isinstance(move_budget, int) or move_budget < 1:
            return False, "move_budget must be a positive integer"
        
        return True, None
    
    @classmethod
    def _validate_update_config(cls, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate update config message data."""
        # Use same validation as start simulation
        return cls._validate_start_simulation(data)