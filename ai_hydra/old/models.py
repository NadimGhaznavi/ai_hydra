"""
Core data models for the AI Hydra.

This module defines the fundamental data structures used throughout the system,
including game state representation, moves, and positions. All models are designed
to be immutable for thread safety and reliable state management.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum
import copy
import random


class MoveAction(Enum):
    """Enumeration of possible move actions in the Snake game."""
    LEFT_TURN = "LEFT_TURN"
    STRAIGHT = "STRAIGHT" 
    RIGHT_TURN = "RIGHT_TURN"


@dataclass(frozen=True)
class Position:
    """Immutable representation of a 2D position on the game grid."""
    x: int
    y: int
    
    def __add__(self, other: 'Position') -> 'Position':
        """Add two positions together."""
        return Position(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Position') -> 'Position':
        """Subtract one position from another."""
        return Position(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class Direction:
    """Immutable representation of a movement direction."""
    dx: int
    dy: int
    
    @classmethod
    def left(cls) -> 'Direction':
        """Create a left-facing direction."""
        return cls(-1, 0)
    
    @classmethod
    def right(cls) -> 'Direction':
        """Create a right-facing direction."""
        return cls(1, 0)
    
    @classmethod
    def up(cls) -> 'Direction':
        """Create an upward-facing direction."""
        return cls(0, -1)
    
    @classmethod
    def down(cls) -> 'Direction':
        """Create a downward-facing direction."""
        return cls(0, 1)
    
    def turn_left(self) -> 'Direction':
        """Return the direction after turning left."""
        # Rotate 90 degrees counter-clockwise: (dx, dy) -> (dy, -dx)
        return Direction(self.dy, -self.dx)
    
    def turn_right(self) -> 'Direction':
        """Return the direction after turning right."""
        # Rotate 90 degrees clockwise: (dx, dy) -> (-dy, dx)
        return Direction(-self.dy, self.dx)


@dataclass(frozen=True)
class Move:
    """Immutable representation of a move action and its resulting direction."""
    action: MoveAction
    resulting_direction: Direction


@dataclass(frozen=True)
class GameBoard:
    """
    Immutable representation of the complete Snake game state.
    
    This class encapsulates all game state data including snake position,
    body segments, direction, food location, score, move count, and random state for
    deterministic behavior.
    """
    snake_head: Position
    snake_body: Tuple[Position, ...]  # Immutable sequence
    direction: Direction
    food_position: Position
    score: int
    move_count: int  # Total moves executed in this game
    random_state: random.Random
    grid_size: Tuple[int, int]
    
    def get_snake_head(self) -> Position:
        """Get the current position of the snake's head."""
        return self.snake_head
    
    def get_snake_body(self) -> List[Position]:
        """Get a list of all snake body segment positions."""
        return list(self.snake_body)
    
    def get_direction(self) -> Direction:
        """Get the current movement direction of the snake."""
        return self.direction
    
    def get_food_position(self) -> Position:
        """Get the current position of the food."""
        return self.food_position
    
    def get_score(self) -> int:
        """Get the current game score."""
        return self.score
    
    def get_move_count(self) -> int:
        """Get the total number of moves executed in this game."""
        return self.move_count
    
    def get_snake_length(self) -> int:
        """Get the current length of the snake (head + body segments)."""
        return 1 + len(self.snake_body)
    
    def get_random_state(self) -> random.Random:
        """Get the current random state for deterministic behavior."""
        return self.random_state
    
    def get_grid_size(self) -> Tuple[int, int]:
        """Get the dimensions of the game grid."""
        return self.grid_size
    
    def clone(self) -> 'GameBoard':
        """
        Create a perfect deep copy of this game board.
        
        This method ensures complete independence between the original
        and cloned boards, including the random state.
        
        Returns:
            GameBoard: A completely independent copy of this board
        """
        return GameBoard(
            snake_head=self.snake_head,
            snake_body=self.snake_body,
            direction=self.direction,
            food_position=self.food_position,
            score=self.score,
            move_count=self.move_count,
            random_state=copy.deepcopy(self.random_state),
            grid_size=self.grid_size
        )
    
    def get_all_snake_positions(self) -> List[Position]:
        """Get all positions occupied by the snake (head + body)."""
        return [self.snake_head] + list(self.snake_body)
    
    def is_position_occupied_by_snake(self, position: Position) -> bool:
        """Check if a given position is occupied by any part of the snake."""
        return position in self.get_all_snake_positions()
    
    def is_position_within_bounds(self, position: Position) -> bool:
        """Check if a given position is within the game grid bounds."""
        width, height = self.grid_size
        return 0 <= position.x < width and 0 <= position.y < height


@dataclass(frozen=True)
class MoveResult:
    """
    Result of executing a move, containing the new game state and outcome.
    
    This class encapsulates the result of applying a move to a game board,
    including the new board state, reward earned, and outcome type.
    """
    new_board: GameBoard
    reward: int
    outcome: str  # "EMPTY", "FOOD", "WALL", "SNAKE"
    is_terminal: bool
    
    def is_collision(self) -> bool:
        """Check if this move result represents a collision (terminal state)."""
        return self.outcome in ["WALL", "SNAKE"]
    
    def is_food_eaten(self) -> bool:
        """Check if this move result represents eating food."""
        return self.outcome == "FOOD"


@dataclass
class GameConfig:
    """Configuration parameters for Snake game simulation."""
    grid_size: Tuple[int, int] = (20, 20)
    initial_snake_length: int = 3
    move_budget: int = 100
    random_seed: int = 42
    max_tree_depth: Optional[int] = None
    max_moves_multiplier: int = 100  # Game terminates when moves > max_moves_multiplier * snake_length
    
    # Reward configuration
    food_reward: int = 10
    collision_penalty: int = -10
    empty_move_reward: int = 0


@dataclass
class ExplorationPath:
    """Represents a complete path through the exploration tree."""
    moves: List[Move]
    cumulative_reward: int
    clone_id: str
    parent_id: Optional[str]
    depth: int
    is_complete: bool


@dataclass
class TreeMetrics:
    """Metrics collected during tree exploration."""
    total_clones_created: int
    max_depth_reached: int
    budget_consumed: int
    paths_evaluated: int
    optimal_path: Optional[ExplorationPath]