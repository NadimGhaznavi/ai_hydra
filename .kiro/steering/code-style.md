# Code Style Guide

## Overview

This document defines the coding standards and style guidelines for the AI Hydra project. Following these standards ensures consistency, readability, and maintainability across the codebase.

## Python Style Guidelines

### 1. PEP 8 Compliance
Follow PEP 8 with these specific configurations:

```python
# Line length: 88 characters (Black formatter default)
# Indentation: 4 spaces (no tabs)
# Imports: Organized in groups with blank lines between

# Good
def calculate_reward(
    move_result: MoveResult, 
    food_reward: int = 10,
    collision_penalty: int = -10
) -> int:
    """Calculate reward based on move result."""
    if move_result.outcome == MoveOutcome.FOOD:
        return food_reward
    elif move_result.outcome in [MoveOutcome.WALL, MoveOutcome.SNAKE]:
        return collision_penalty
    return 0

# Bad
def calculate_reward(move_result,food_reward=10,collision_penalty=-10):
    if move_result.outcome==MoveOutcome.FOOD:return food_reward
    elif move_result.outcome in [MoveOutcome.WALL,MoveOutcome.SNAKE]:return collision_penalty
    return 0
```

### 2. Import Organization
```python
# Standard library imports
import copy
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

# Third-party imports
import torch
import torch.nn as nn
from hydra_zen import make_config
from hypothesis import given, strategies as st

# Local imports
from .models import GameBoard, Move, MoveResult
from .config import SimulationConfig
```

### 3. Naming Conventions
```python
# Classes: PascalCase
class GameBoard:
    pass

class SnakeNet(nn.Module):
    pass

# Functions and variables: snake_case
def execute_move(board: GameBoard, move: Move) -> MoveResult:
    pass

def get_possible_moves() -> List[Move]:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_TREE_DEPTH = 10
DEFAULT_GRID_SIZE = (20, 20)
FOOD_REWARD = 10

# Private methods: leading underscore
def _validate_board_state(self, board: GameBoard) -> bool:
    pass

# Protected attributes: leading underscore
class GameBoard:
    def __init__(self):
        self._random_state = None
        self._internal_cache = {}
```

## Type Annotations

### 1. Comprehensive Type Hints
```python
from typing import Dict, List, Optional, Tuple, Union, Any
import torch

def create_exploration_clones(
    master_board: GameBoard,
    clone_count: int = 3,
    seed_offset: int = 0
) -> List[ExplorationClone]:
    """Create exploration clones from master board."""
    pass

def process_neural_network_batch(
    features: torch.Tensor,
    model: nn.Module,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Process batch of features through neural network."""
    pass

# Complex return types
def execute_decision_cycle(self) -> Tuple[Move, int, List[str]]:
    """Execute decision cycle returning move, budget used, and log entries."""
    pass

# Generic types for containers
def get_clone_results(self) -> Dict[str, MoveResult]:
    """Get results mapped by clone ID."""
    pass
```

### 2. Type Aliases for Clarity
```python
# Define type aliases for complex types
CloneId = str
Reward = int
Budget = int
Seed = int
GridSize = Tuple[int, int]
Position = Tuple[int, int]

# Use aliases in function signatures
def create_clone(
    parent_id: CloneId,
    board: GameBoard,
    seed: Seed
) -> ExplorationClone:
    pass

def calculate_path_reward(path: List[Move]) -> Reward:
    pass
```

### 3. Optional and Union Types
```python
# Use Optional for nullable values
def find_food_position(board: GameBoard) -> Optional[Position]:
    """Find food position, None if no food exists."""
    pass

# Use Union for multiple valid types
def load_config(source: Union[str, dict, SimulationConfig]) -> SimulationConfig:
    """Load configuration from various sources."""
    pass

# Avoid bare Any - be specific
def process_data(data: List[Dict[str, Union[int, str]]]) -> None:
    # Good: Specific structure
    pass

def process_data(data: Any) -> None:
    # Bad: Too generic
    pass
```

## Documentation Standards

### 1. Docstring Format (Google Style)
```python
def execute_move(board: GameBoard, move: Move) -> MoveResult:
    """
    Execute a move on the game board and return the result.
    
    This function applies the specified move to the game board, updating
    the snake position, checking for collisions, and handling food consumption.
    The original board is not modified; a new board state is returned.
    
    Args:
        board: The current game board state
        move: The move to execute (LEFT_TURN, STRAIGHT, RIGHT_TURN)
        
    Returns:
        MoveResult: Contains the new board state, outcome, reward, and
            whether the game has ended
            
    Raises:
        ValueError: If the move is invalid for the current board state
        TypeError: If board or move have incorrect types
        
    Example:
        >>> board = GameBoard.create_initial((10, 10), 3, 42)
        >>> move = Move.STRAIGHT
        >>> result = execute_move(board, move)
        >>> print(f"Reward: {result.reward}, Game Over: {result.is_terminal}")
        
    Note:
        This function maintains immutability - the original board is never
        modified. All random operations use the board's internal random state
        to ensure deterministic behavior.
    """
    pass
```

### 2. Class Documentation
```python
class ExplorationClone:
    """
    Represents a single exploration clone in the tree search.
    
    An exploration clone maintains its own game state and path history,
    allowing the tree search algorithm to explore different move sequences
    concurrently. Each clone tracks its cumulative reward and can generate
    sub-clones for deeper exploration.
    
    Attributes:
        clone_id: Unique identifier for this clone (e.g., "1L", "2SR")
        current_board: Current game board state for this clone
        path_from_root: Sequence of moves from root to current state
        cumulative_reward: Total reward accumulated along this path
        is_terminated: Whether this clone has reached a terminal state
        
    Example:
        >>> master_board = GameBoard.create_initial((10, 10), 3, 42)
        >>> clone = ExplorationClone("1L", master_board, [Move.LEFT_TURN])
        >>> result = clone.execute_move(Move.STRAIGHT)
        >>> print(f"Clone {clone.clone_id} reward: {clone.cumulative_reward}")
    """
    
    def __init__(self, clone_id: str, board: GameBoard, path: List[Move]):
        """
        Initialize exploration clone.
        
        Args:
            clone_id: Unique identifier for this clone
            board: Initial game board state
            path: Move sequence from root to this state
        """
        pass
```

### 3. Module Documentation
```python
"""
Game Logic Module

This module contains the core game mechanics for the Snake game, including
move execution, collision detection, food placement, and reward calculation.
All functions in this module are pure functions that do not modify their
inputs, ensuring deterministic and testable behavior.

The module is designed to work with immutable GameBoard objects and uses
deterministic random number generation for reproducible game behavior.

Key Functions:
    execute_move: Apply a move to a game board
    check_collision: Detect wall and self-collision
    place_food: Generate new food position
    calculate_reward: Determine reward for move outcome

Example:
    >>> from ai_hydra.game_logic import GameLogic
    >>> board = GameBoard.create_initial((10, 10), 3, 42)
    >>> result = GameLogic.execute_move(board, Move.STRAIGHT)
    >>> print(f"New score: {result.new_board.score}")
"""
```

## Error Handling

### 1. Exception Hierarchy
```python
# Define custom exceptions for domain-specific errors
class SnakeGameError(Exception):
    """Base exception for Snake game errors."""
    pass

class InvalidMoveError(SnakeGameError):
    """Raised when an invalid move is attempted."""
    pass

class BoardStateError(SnakeGameError):
    """Raised when board state is invalid or corrupted."""
    pass

class ConfigurationError(SnakeGameError):
    """Raised when configuration is invalid."""
    pass

# Use specific exceptions
def execute_move(board: GameBoard, move: Move) -> MoveResult:
    if not isinstance(board, GameBoard):
        raise TypeError(f"Expected GameBoard, got {type(board)}")
    
    if not isinstance(move, Move):
        raise TypeError(f"Expected Move, got {type(move)}")
    
    if not board.is_valid():
        raise BoardStateError(f"Invalid board state: {board.get_validation_errors()}")
    
    # ... rest of implementation
```

### 2. Error Context and Logging
```python
import logging

logger = logging.getLogger(__name__)

def execute_tree_search(board: GameBoard, budget: int) -> TreeSearchResult:
    """Execute tree search with comprehensive error handling."""
    try:
        logger.info(f"Starting tree search with budget {budget}")
        
        # Validate inputs
        if budget <= 0:
            raise ValueError(f"Budget must be positive, got {budget}")
        
        # Execute search
        result = _perform_tree_search(board, budget)
        
        logger.info(f"Tree search completed: {result.paths_evaluated} paths evaluated")
        return result
        
    except Exception as e:
        logger.error(f"Tree search failed: {e}", exc_info=True)
        # Provide fallback behavior
        return TreeSearchResult.create_fallback(board)
```

### 3. Defensive Programming
```python
def create_sub_clones(parent_clone: ExplorationClone) -> List[ExplorationClone]:
    """Create sub-clones with defensive validation."""
    # Validate preconditions
    assert parent_clone is not None, "Parent clone cannot be None"
    assert not parent_clone.is_terminated, "Cannot create sub-clones from terminated clone"
    assert parent_clone.current_board.is_valid(), "Parent board state is invalid"
    
    try:
        sub_clones = []
        for move in [Move.LEFT_TURN, Move.STRAIGHT, Move.RIGHT_TURN]:
            clone_id = f"{parent_clone.clone_id}{move.abbreviation}"
            sub_clone = ExplorationClone(clone_id, parent_clone.current_board, 
                                       parent_clone.path_from_root + [move])
            sub_clones.append(sub_clone)
        
        # Validate postconditions
        assert len(sub_clones) == 3, f"Expected 3 sub-clones, created {len(sub_clones)}"
        assert all(clone.clone_id.startswith(parent_clone.clone_id) for clone in sub_clones)
        
        return sub_clones
        
    except Exception as e:
        logger.error(f"Failed to create sub-clones for {parent_clone.clone_id}: {e}")
        raise
```

## Code Organization

### 1. Class Structure
```python
class HydraMgr:
    """Main orchestrator for the hybrid neural network + tree search system."""
    
    # Class constants
    DEFAULT_BUDGET = 100
    MAX_DECISION_CYCLES = 1000
    
    def __init__(self, config: SimulationConfig):
        """Initialize with configuration validation."""
        # Public attributes
        self.config = config
        self.master_game = MasterGame(config)
        
        # Private attributes
        self._budget_controller = BudgetController(config.move_budget)
        self._state_manager = StateManager()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self._setup_neural_network()
        self._setup_logging()
    
    # Public interface methods
    def run_simulation(self) -> SimulationResult:
        """Run complete simulation."""
        pass
    
    def execute_decision_cycle(self) -> DecisionResult:
        """Execute single decision cycle."""
        pass
    
    # Private implementation methods
    def _setup_neural_network(self) -> None:
        """Initialize neural network components."""
        pass
    
    def _execute_tree_search(self, initial_move: Move) -> TreeSearchResult:
        """Execute tree search starting from initial move."""
        pass
    
    def _evaluate_paths(self, paths: List[ExplorationPath]) -> Move:
        """Evaluate exploration paths and select optimal move."""
        pass
    
    # Properties for computed values
    @property
    def current_score(self) -> int:
        """Get current game score."""
        return self.master_game.get_score()
    
    @property
    def is_game_over(self) -> bool:
        """Check if game has ended."""
        return self.master_game.is_terminal()
```

### 2. Function Organization
```python
# Group related functions together
def create_initial_board(grid_size: GridSize, snake_length: int, seed: Seed) -> GameBoard:
    """Create initial game board."""
    pass

def clone_board(board: GameBoard) -> GameBoard:
    """Create independent copy of game board."""
    pass

def validate_board(board: GameBoard) -> bool:
    """Validate board state consistency."""
    pass

# Separate pure functions from stateful operations
def calculate_new_position(current_pos: Position, direction: Direction) -> Position:
    """Pure function: calculate new position from current position and direction."""
    return (current_pos[0] + direction.dx, current_pos[1] + direction.dy)

def update_game_state(board: GameBoard, new_position: Position) -> GameBoard:
    """Stateful operation: create new board with updated state."""
    pass
```

### 3. Configuration and Constants
```python
# Configuration constants
class GameConstants:
    """Game-specific constants."""
    MIN_GRID_SIZE = (5, 5)
    MAX_GRID_SIZE = (50, 50)
    MIN_SNAKE_LENGTH = 1
    DEFAULT_FOOD_REWARD = 10
    DEFAULT_COLLISION_PENALTY = -10
    
    # Neural network constants
    FEATURE_VECTOR_SIZE = 19
    HIDDEN_LAYER_SIZE = 200
    OUTPUT_SIZE = 3

# Runtime configuration
@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration derived from user settings."""
    enable_logging: bool
    enable_profiling: bool
    enable_visualization: bool
    performance_mode: bool
    
    @classmethod
    def from_simulation_config(cls, config: SimulationConfig) -> 'RuntimeConfig':
        """Create runtime config from simulation config."""
        return cls(
            enable_logging=config.log_level != "NONE",
            enable_profiling=config.enable_profiling,
            enable_visualization=hasattr(config, 'enable_visualization'),
            performance_mode=config.move_budget > 200
        )
```

## Performance Considerations

### 1. Efficient Data Structures
```python
# Use appropriate data structures
from collections import deque
from typing import NamedTuple

# Good: Use deque for frequent append/pop operations
class SnakeBody:
    def __init__(self):
        self._segments = deque()  # Efficient for head/tail operations
    
    def add_head(self, position: Position) -> None:
        self._segments.appendleft(position)
    
    def remove_tail(self) -> Position:
        return self._segments.pop()

# Good: Use NamedTuple for immutable data
class MoveResult(NamedTuple):
    new_board: GameBoard
    outcome: MoveOutcome
    reward: int
    is_terminal: bool

# Bad: Inefficient list operations
def update_snake_bad(snake_body: List[Position], new_head: Position) -> List[Position]:
    return [new_head] + snake_body[:-1]  # Creates new list every time
```

### 2. Memory Management
```python
# Use generators for large sequences
def generate_all_possible_paths(initial_board: GameBoard, max_depth: int):
    """Generate paths lazily to save memory."""
    def _generate_paths(board, path, depth):
        if depth >= max_depth:
            yield path
            return
        
        for move in get_possible_moves(board):
            result = execute_move(board, move)
            if not result.is_terminal:
                yield from _generate_paths(result.new_board, path + [move], depth + 1)
    
    yield from _generate_paths(initial_board, [], 0)

# Use context managers for resource cleanup
class TreeSearchContext:
    def __init__(self, budget: int):
        self.budget = budget
        self._active_clones = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up all active clones
        for clone in self._active_clones:
            clone.cleanup()
        self._active_clones.clear()
```

### 3. Caching and Memoization
```python
from functools import lru_cache
from typing import Dict

class GameLogic:
    # Cache expensive computations
    @staticmethod
    @lru_cache(maxsize=1000)
    def get_possible_moves(current_direction: Direction) -> Tuple[Move, ...]:
        """Cache possible moves for each direction."""
        # Expensive computation cached
        pass
    
    # Instance-level caching
    def __init__(self):
        self._collision_cache: Dict[Tuple[Position, GridSize], bool] = {}
    
    def check_wall_collision(self, position: Position, grid_size: GridSize) -> bool:
        """Check collision with caching."""
        cache_key = (position, grid_size)
        if cache_key not in self._collision_cache:
            self._collision_cache[cache_key] = (
                position[0] < 0 or position[0] >= grid_size[0] or
                position[1] < 0 or position[1] >= grid_size[1]
            )
        return self._collision_cache[cache_key]
```

## Testing Integration

### 1. Testable Code Design
```python
# Design for testability
class BudgetController:
    def __init__(self, initial_budget: int, clock=None):
        self.initial_budget = initial_budget
        self.remaining_budget = initial_budget
        self._clock = clock or time.time  # Dependency injection for testing
    
    def consume_move(self) -> None:
        """Consume one move from budget."""
        if self.remaining_budget > 0:
            self.remaining_budget -= 1
            self._log_budget_consumption()
    
    def _log_budget_consumption(self) -> None:
        """Log budget consumption with timestamp."""
        timestamp = self._clock()
        logger.info(f"Budget consumed at {timestamp}, remaining: {self.remaining_budget}")

# Test with mock clock
def test_budget_consumption():
    mock_clock = lambda: 1234567890.0
    controller = BudgetController(10, clock=mock_clock)
    controller.consume_move()
    assert controller.remaining_budget == 9
```

### 2. Assertion Messages
```python
def validate_clone_creation(parent: ExplorationClone, children: List[ExplorationClone]):
    """Validate clone creation with descriptive assertions."""
    assert len(children) == 3, (
        f"Expected 3 child clones, got {len(children)} "
        f"for parent {parent.clone_id}"
    )
    
    for i, child in enumerate(children):
        assert child.clone_id.startswith(parent.clone_id), (
            f"Child clone {i} ID '{child.clone_id}' should start with "
            f"parent ID '{parent.clone_id}'"
        )
        
        assert len(child.path_from_root) == len(parent.path_from_root) + 1, (
            f"Child clone {child.clone_id} path length should be "
            f"{len(parent.path_from_root) + 1}, got {len(child.path_from_root)}"
        )
```

## Code Formatting

### 1. Automated Formatting
Use Black formatter with these settings:

```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

### 2. Import Sorting
Use isort with Black compatibility:

```toml
# pyproject.toml
[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["ai_hydra"]
known_third_party = ["torch", "hydra_zen", "hypothesis"]
```

### 3. Linting Configuration
Use flake8 with these settings:

```ini
# .flake8
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist
per-file-ignores = __init__.py:F401
```

This code style guide ensures consistent, readable, and maintainable code across the AI Hydra project.