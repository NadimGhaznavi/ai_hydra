"""
AI Hydra - A budget-constrained tree search system for Snake Game AI.

This package implements a sophisticated hybrid neural network + tree search system
for Snake Game AI decision making, combining the speed of neural network predictions
with the reliability of tree search to eliminate collision-based "blunder" moves.
"""

__version__ = "0.2.0"
__author__ = "AI Hydra Team"

# Core components
from .models import Position, Direction, Move, GameBoard, MoveAction, MoveResult
from .game_logic import GameLogic
from .hydra_mgr import HydraMgr
from .master_game import MasterGame
from .config import SimulationConfig, NetworkConfig, LoggingConfig
from .logging_config import SimulationLogger

# ZeroMQ communication components
from .zmq_server import ZMQServer
from .zmq_client_example import ZMQClient
from .zmq_protocol import ZMQMessage, MessageType, MessageBuilder, GameStateData, PerformanceMetrics
from .headless_server import HeadlessServer

__all__ = [
    # Core components
    "Position",
    "Direction", 
    "Move",
    "GameBoard",
    "MoveAction",
    "MoveResult",
    "GameLogic",
    "HydraMgr",
    "MasterGame",
    "SimulationConfig",
    "NetworkConfig", 
    "LoggingConfig",
    "SimulationLogger",
    
    # ZeroMQ components
    "ZMQServer",
    "ZMQClient", 
    "ZMQMessage",
    "MessageType",
    "MessageBuilder",
    "GameStateData",
    "PerformanceMetrics",
    "HeadlessServer",
]