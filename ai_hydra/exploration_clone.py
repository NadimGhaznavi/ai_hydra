"""
Exploration clone for tree search execution.

This module provides the ExplorationClone class that represents individual
exploration paths in the tree search. Each clone maintains its own game state,
path history, and cumulative reward calculation.
"""

from typing import List, Optional
import logging
from .models import GameBoard, Move, MoveResult
from .game_logic import GameLogic


class ExplorationClone:
    """
    Represents an individual exploration path in the tree search.
    
    Each ExplorationClone maintains its own GameBoard state, tracks the path
    from root to current position, calculates cumulative rewards, and provides
    logging capabilities for debugging and analysis.
    """
    
    def __init__(self, initial_board: GameBoard, clone_id: str, 
                 parent_id: Optional[str] = None):
        """
        Initialize an exploration clone with the given game board.
        
        Args:
            initial_board: The initial game board state for this clone
            clone_id: Unique identifier for this clone (e.g., "1", "1L", "1LL")
            parent_id: ID of the parent clone (None for root clones)
        """
        self.current_board = initial_board.clone()  # Ensure independence
        self.clone_id = clone_id
        self.parent_id = parent_id
        self.path_from_root: List[Move] = []
        self.cumulative_reward = 0
        self.is_terminated_flag = False
        self.termination_reason: Optional[str] = None
        self.step_count = 0
        
        # Set up logging
        self.logger = logging.getLogger(f"ExplorationClone.{clone_id}")
    
    def execute_move(self, move: Move) -> MoveResult:
        """
        Execute a move using Game_Logic integration.
        
        This method applies a move to the current game board using the Game_Logic
        module, updates the path history, calculates cumulative rewards, and
        handles termination conditions.
        
        Args:
            move: The move to execute
            
        Returns:
            MoveResult: The result of executing the move
        """
        if self.is_terminated_flag:
            raise RuntimeError(f"Clone {self.clone_id} is already terminated")
        
        # Execute move using Game_Logic
        result = GameLogic.execute_move(self.current_board, move)
        
        # Update clone state
        self.current_board = result.new_board
        self.path_from_root.append(move)
        self.cumulative_reward += result.reward
        self.step_count += 1
        
        # Handle termination
        if result.is_terminal:
            self.is_terminated_flag = True
            self.termination_reason = result.outcome
        
        # Log the step
        self.log_step(self.step_count, result.reward, result.outcome)
        
        return result
    
    def get_cumulative_reward(self) -> int:
        """
        Get the cumulative reward from all moves executed by this clone.
        
        Returns:
            int: Total cumulative reward
        """
        return self.cumulative_reward
    
    def get_path_from_root(self) -> List[Move]:
        """
        Get the complete path of moves from root to current position.
        
        Returns:
            List[Move]: List of moves from root to current position
        """
        return self.path_from_root.copy()  # Return copy to prevent modification
    
    def is_terminated(self) -> bool:
        """
        Check if this clone has terminated (collision or other terminal condition).
        
        Returns:
            bool: True if clone is terminated
        """
        return self.is_terminated_flag
    
    def get_current_board(self) -> GameBoard:
        """
        Get the current game board state.
        
        Returns:
            GameBoard: Current game board (cloned for safety)
        """
        return self.current_board.clone()
    
    def get_clone_id(self) -> str:
        """
        Get the unique identifier for this clone.
        
        Returns:
            str: Clone identifier
        """
        return self.clone_id
    
    def get_parent_id(self) -> Optional[str]:
        """
        Get the parent clone identifier.
        
        Returns:
            Optional[str]: Parent clone ID, or None for root clones
        """
        return self.parent_id
    
    def get_termination_reason(self) -> Optional[str]:
        """
        Get the reason for termination if the clone is terminated.
        
        Returns:
            Optional[str]: Termination reason ("WALL", "SNAKE", etc.) or None
        """
        return self.termination_reason
    
    def get_step_count(self) -> int:
        """
        Get the number of steps executed by this clone.
        
        Returns:
            int: Number of steps executed
        """
        return self.step_count
    
    def get_depth(self) -> int:
        """
        Get the depth of this clone in the exploration tree.
        
        Returns:
            int: Depth (number of moves from root)
        """
        return len(self.path_from_root)
    
    def log_step(self, step_number: int, reward: int, outcome: str) -> None:
        """
        Log a step execution with clone identifier and results.
        
        This method provides structured logging for debugging and analysis,
        following the format: CLONE_ID RESULT:OUTCOME REWARD:VALUE SCORE:TOTAL_SCORE
        
        Args:
            step_number: The step number in this clone's execution
            reward: The reward received for this step
            outcome: The outcome of the step ("EMPTY", "FOOD", "WALL", "SNAKE")
        """
        current_score = self.current_board.get_score()
        log_message = f"{self.clone_id} RESULT:{outcome} REWARD:{reward} SCORE:{current_score}"
        
        self.logger.info(log_message)
        
        # Also log termination if this step caused termination
        if self.is_terminated_flag:
            termination_message = f"{self.clone_id} TERMINATED: {self.termination_reason} after {step_number} steps"
            self.logger.info(termination_message)
    
    def can_create_sub_clones(self) -> bool:
        """
        Check if this clone can create sub-clones (i.e., it's not terminated).
        
        Returns:
            bool: True if sub-clones can be created
        """
        return not self.is_terminated_flag
    
    def get_possible_moves(self) -> List[Move]:
        """
        Get all possible moves from the current board state.
        
        Returns:
            List[Move]: List of possible moves
        """
        if self.is_terminated_flag:
            return []
        
        return GameLogic.get_possible_moves(self.current_board.get_direction())
    
    def create_sub_clone_board(self) -> GameBoard:
        """
        Create a board suitable for sub-clone creation.
        
        This method returns a cloned board that can be used to initialize
        sub-clones, ensuring complete independence.
        
        Returns:
            GameBoard: Cloned board for sub-clone initialization
        """
        if self.is_terminated_flag:
            raise RuntimeError(f"Cannot create sub-clone board from terminated clone {self.clone_id}")
        
        return self.current_board.clone()
    
    def get_exploration_summary(self) -> dict:
        """
        Get a comprehensive summary of this clone's exploration.
        
        Returns:
            dict: Summary containing all relevant exploration data
        """
        return {
            "clone_id": self.clone_id,
            "parent_id": self.parent_id,
            "depth": self.get_depth(),
            "step_count": self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "is_terminated": self.is_terminated_flag,
            "termination_reason": self.termination_reason,
            "path_length": len(self.path_from_root),
            "current_score": self.current_board.get_score(),
            "can_create_sub_clones": self.can_create_sub_clones()
        }
    
    def __str__(self) -> str:
        """String representation of the exploration clone."""
        status = "TERMINATED" if self.is_terminated_flag else "ACTIVE"
        return f"ExplorationClone({self.clone_id}, {status}, reward={self.cumulative_reward}, depth={self.get_depth()})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the exploration clone."""
        return (f"ExplorationClone(id='{self.clone_id}', parent='{self.parent_id}', "
                f"depth={self.get_depth()}, reward={self.cumulative_reward}, "
                f"terminated={self.is_terminated_flag})")