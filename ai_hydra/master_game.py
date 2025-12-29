"""
Master game management for the AI Hydra.

This module provides the MasterGame class which maintains the authoritative
game state and coordinates with GameLogic for move execution.
"""

from typing import Optional
from .models import GameBoard, Move, MoveResult
from .game_logic import GameLogic
from .logging_config import SimulationLogger
from .config import LoggingConfig


class MasterGame:
    """
    Maintains the authoritative Snake Game state.
    
    This class manages the master game board and coordinates with GameLogic
    for move execution while maintaining immutability principles.
    """
    
    def __init__(self, initial_board: GameBoard, logging_config: Optional[LoggingConfig] = None, 
                 max_moves_multiplier: int = 100):
        """
        Initialize the master game with an initial board state.
        
        Args:
            initial_board: The initial game board state
            logging_config: Logging configuration (optional)
            max_moves_multiplier: Multiplier for max moves calculation
        """
        self._current_board = initial_board
        self.max_moves_multiplier = max_moves_multiplier
        
        # Initialize logger
        self.logger = SimulationLogger("master_game", logging_config)
        
        # Game state tracking
        self.move_history = []
        self.score_history = [initial_board.score]
        self.game_statistics = {
            'total_moves': 0,
            'food_eaten': 0,
            'collisions': 0,
            'max_score': initial_board.score,
            'game_duration': 0.0
        }
        
        self.logger.log_system_event("MasterGame initialized", {
            "initial_score": initial_board.score,
            "grid_size": f"{initial_board.grid_size[0]}x{initial_board.grid_size[1]}",
            "snake_length": len(initial_board.snake_body) + 1,  # +1 for head
            "tracking_enabled": True
        })
    
    def get_current_board(self) -> GameBoard:
        """
        Get the current game board state.
        
        Returns:
            GameBoard: The current board state
        """
        return self._current_board
    
    def apply_move(self, move: Move) -> MoveResult:
        """
        Apply a move to the master game using GameLogic.
        
        Args:
            move: The move to apply
            
        Returns:
            MoveResult: The result of applying the move
        """
        old_score = self._current_board.score
        old_snake_length = len(self._current_board.snake_body) + 1
        
        result = GameLogic.execute_move(self._current_board, move, self.max_moves_multiplier)
        
        # Update the current board to the new state
        self._current_board = result.new_board
        
        # Track move and update statistics
        self.move_history.append({
            'move': move,
            'result': result,
            'move_number': len(self.move_history) + 1,
            'score_before': old_score,
            'score_after': result.new_board.score
        })
        
        self.score_history.append(result.new_board.score)
        self._update_game_statistics(result, old_score, old_snake_length)
        
        # Log move application
        self.logger.log_system_event("Master move applied", {
            "move": move.action.value,
            "outcome": result.outcome,
            "reward": result.reward,
            "score_change": result.new_board.score - old_score,
            "new_score": result.new_board.score,
            "snake_length": len(result.new_board.snake_body) + 1,
            "is_terminal": result.is_terminal,
            "total_moves": len(self.move_history)
        })
        
        # Log terminal state if game ended
        if result.is_terminal:
            self._log_game_completion()
        
        return result
    
    def is_terminal(self) -> bool:
        """
        Check if the game is in a terminal state.
        
        Returns:
            bool: True if the game is over
        """
        return GameLogic.is_game_over(self._current_board, self.max_moves_multiplier)
    
    def get_score(self) -> int:
        """
        Get the current game score.
        
        Returns:
            int: Current score
        """
        return self._current_board.score
    
    def clone_board(self) -> GameBoard:
        """
        Create a perfect clone of the current board for exploration.
        
        Returns:
            GameBoard: A cloned board state
        """
        cloned_board = self._current_board.clone()
        
        self.logger.log_system_event("Board cloned for exploration", {
            "current_score": self._current_board.score,
            "snake_length": len(self._current_board.snake_body) + 1,
            "clone_created": True
        })
        
        return cloned_board
    
    def get_game_statistics(self) -> dict:
        """
        Get comprehensive game statistics.
        
        Returns:
            dict: Dictionary containing game statistics
        """
        current_stats = self.game_statistics.copy()
        current_stats.update({
            'current_score': self._current_board.score,
            'current_snake_length': len(self._current_board.snake_body) + 1,
            'moves_played': len(self.move_history),
            'score_progression': self.score_history.copy(),
            'is_terminal': self.is_terminal(),
            'average_score_per_move': self._current_board.score / max(1, len(self.move_history))
        })
        return current_stats
    
    def log_game_summary(self) -> None:
        """
        Log a comprehensive game summary.
        """
        stats = self.get_game_statistics()
        
        self.logger.log_system_event("Master game summary", {
            "final_score": stats['current_score'],
            "total_moves": stats['moves_played'],
            "food_eaten": stats['food_eaten'],
            "collisions": stats['collisions'],
            "max_score": stats['max_score'],
            "final_snake_length": stats['current_snake_length'],
            "avg_score_per_move": f"{stats['average_score_per_move']:.2f}",
            "game_completed": stats['is_terminal']
        })
    
    def _update_game_statistics(self, result: MoveResult, old_score: int, old_snake_length: int) -> None:
        """Update game statistics based on move result."""
        self.game_statistics['total_moves'] += 1
        
        # Track food consumption
        if result.new_board.score > old_score:
            self.game_statistics['food_eaten'] += 1
        
        # Track collisions
        if result.is_terminal and result.reward < 0:
            self.game_statistics['collisions'] += 1
        
        # Update max score
        if result.new_board.score > self.game_statistics['max_score']:
            self.game_statistics['max_score'] = result.new_board.score
    
    def _log_game_completion(self) -> None:
        """Log game completion details."""
        stats = self.get_game_statistics()
        
        self.logger.log_system_event("Master game completed", {
            "termination_reason": "collision" if self.game_statistics['collisions'] > 0 else "unknown",
            "final_score": stats['current_score'],
            "total_moves": stats['moves_played'],
            "food_consumed": stats['food_eaten'],
            "final_snake_length": stats['current_snake_length'],
            "game_efficiency": f"{stats['food_eaten'] / max(1, stats['moves_played']) * 100:.1f}%"
        })