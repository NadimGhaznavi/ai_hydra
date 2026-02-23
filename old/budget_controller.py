"""
Budget management for tree search exploration.

This module provides budget control for managing computational resources during
tree search exploration. The BudgetController allows round completion even when
budget is exceeded, ensuring fair evaluation of all clones in a round.
"""

from typing import Optional
from .logging_config import SimulationLogger
from .config import LoggingConfig


class BudgetController:
    """
    Budget controller for managing computational resources during tree search.
    
    The controller tracks move budget consumption and allows round completion
    even when budget is exceeded, ensuring all clones in a round get to execute
    their moves before budget enforcement occurs.
    """
    
    def __init__(self, initial_budget: int, logging_config: Optional[LoggingConfig] = None):
        """
        Initialize the budget controller with the given budget.
        
        Args:
            initial_budget: The initial move budget for tree exploration
            logging_config: Logging configuration (optional)
        """
        if initial_budget <= 0:
            raise ValueError("Initial budget must be positive")
        
        self.initial_budget = initial_budget
        self.current_budget = initial_budget
        self.moves_in_current_round = 0
        self.total_moves_consumed = 0
        self.round_number = 0
        
        # Initialize logger
        self.logger = SimulationLogger("budget_controller", logging_config)
        
        # Budget utilization tracking
        self.budget_history = []  # Track budget usage over time
        self.round_history = []   # Track moves per round
        self.efficiency_metrics = {
            'total_rounds': 0,
            'avg_moves_per_round': 0.0,
            'budget_utilization_rate': 0.0,
            'peak_round_size': 0
        }
        
        self.logger.log_system_event("BudgetController initialized", {
            "initial_budget": initial_budget,
            "tracking_enabled": True
        })
    
    def initialize_budget(self, budget: int) -> None:
        """
        Initialize the budget with a new value.
        
        Args:
            budget: The budget value to initialize with
        """
        if budget <= 0:
            raise ValueError("Budget must be positive")
        
        old_budget = self.initial_budget
        self.initial_budget = budget
        self.current_budget = budget
        self.moves_in_current_round = 0
        self.total_moves_consumed = 0
        self.round_number = 0
        
        # Reset tracking
        self.budget_history.clear()
        self.round_history.clear()
        self._reset_efficiency_metrics()
        
        self.logger.log_system_event("Budget reinitialized", {
            "old_budget": old_budget,
            "new_budget": budget,
            "tracking_reset": True
        })
    
    def consume_move(self) -> None:
        """
        Consume one move from the budget with consistency validation.
        
        This method decrements the budget by 1 and tracks moves in the current
        round. It allows budget overrun to complete the current round.
        """
        # Validate budget state before consumption
        if self.current_budget < -1000:  # Sanity check for extreme negative values
            self.logger.log_error("BudgetController", f"Budget corruption detected: {self.current_budget}")
            # Reset to a safe state
            self.current_budget = 0
        
        self.current_budget -= 1
        self.moves_in_current_round += 1
        self.total_moves_consumed += 1
        
        # Validate arithmetic consistency
        expected_consumed = self.initial_budget - self.current_budget
        if expected_consumed != self.total_moves_consumed:
            self.logger.log_error("BudgetController", 
                f"Budget arithmetic inconsistency: expected {expected_consumed}, actual {self.total_moves_consumed}")
            # Correct the inconsistency
            self.total_moves_consumed = expected_consumed
        
        # Log budget consumption at debug level
        self.logger.log_budget_status(self.current_budget, self.initial_budget)
        
        # Track budget utilization patterns
        utilization_rate = (self.initial_budget - self.current_budget) / self.initial_budget
        self.budget_history.append({
            'remaining': self.current_budget,
            'consumed': self.get_budget_consumed(),
            'utilization_rate': utilization_rate,
            'round': self.round_number,
            'moves_in_round': self.moves_in_current_round
        })
        
        # Log budget exhaustion
        if self.current_budget <= 0 and self.current_budget > -1:  # Log only once when exhausted
            self.logger.log_system_event("Budget exhausted", {
                "total_consumed": self.total_moves_consumed,
                "rounds_completed": self.round_number,
                "moves_in_final_round": self.moves_in_current_round
            })
    
    def get_remaining_budget(self) -> int:
        """
        Get the remaining budget.
        
        Returns:
            int: The remaining budget (can be negative if overrun occurred)
        """
        return self.current_budget
    
    def is_budget_exhausted(self) -> bool:
        """
        Check if the budget is exhausted.
        
        Returns:
            bool: True if budget is exhausted (remaining budget <= 0)
        """
        return self.current_budget <= 0
    
    def get_moves_in_current_round(self) -> int:
        """
        Get the number of moves executed in the current round.
        
        Returns:
            int: Number of moves in current round
        """
        return self.moves_in_current_round
    
    def start_new_round(self) -> None:
        """
        Start a new round of exploration.
        
        This resets the moves counter for the current round and increments
        the round number for tracking purposes.
        """
        # Record the completed round
        if self.round_number > 0:  # Don't record the initial state
            self.round_history.append({
                'round_number': self.round_number,
                'moves_in_round': self.moves_in_current_round,
                'budget_at_start': self.current_budget + self.moves_in_current_round,
                'budget_at_end': self.current_budget
            })
            
            # Update efficiency metrics
            self._update_efficiency_metrics()
            
            self.logger.log_system_event(f"Round {self.round_number} completed", {
                "moves_executed": self.moves_in_current_round,
                "budget_remaining": self.current_budget,
                "total_rounds": self.round_number
            })
        
        self.moves_in_current_round = 0
        self.round_number += 1
        
        self.logger.log_system_event(f"Starting round {self.round_number}", {
            "budget_remaining": self.current_budget,
            "total_consumed": self.get_budget_consumed()
        })
    
    def reset_budget(self) -> None:
        """
        Reset the budget to its initial value for a new decision cycle.
        
        This method resets all counters and prepares the controller for
        a new tree exploration cycle.
        """
        # Log decision cycle completion summary
        if self.total_moves_consumed > 0:
            self.logger.log_system_event("Decision cycle budget summary", {
                "total_moves_consumed": self.total_moves_consumed,
                "rounds_completed": self.round_number,
                "budget_utilization": f"{(self.total_moves_consumed / self.initial_budget) * 100:.1f}%",
                "avg_moves_per_round": self.efficiency_metrics['avg_moves_per_round'],
                "peak_round_size": self.efficiency_metrics['peak_round_size']
            })
        
        self.current_budget = self.initial_budget
        self.moves_in_current_round = 0
        self.total_moves_consumed = 0
        self.round_number = 0
        
        # Clear tracking for new cycle but keep efficiency metrics
        self.budget_history.clear()
        self.round_history.clear()
        
        self.logger.log_system_event("Budget reset for new decision cycle", {
            "reset_budget": self.initial_budget
        })
    
    def get_budget_consumed(self) -> int:
        """
        Get the total amount of budget consumed.
        
        Returns:
            int: Total budget consumed since last reset
        """
        return self.initial_budget - self.current_budget
    
    def get_round_number(self) -> int:
        """
        Get the current round number.
        
        Returns:
            int: Current round number (0-based)
        """
        return self.round_number
    
    def get_total_moves_consumed(self) -> int:
        """
        Get the total number of moves consumed across all rounds.
        
        Returns:
            int: Total moves consumed
        """
        return self.total_moves_consumed
    
    def can_continue_exploration(self) -> bool:
        """
        Check if exploration can continue based on budget.
        
        This method considers round completion logic - if budget is exhausted
        but we're in the middle of a round, exploration can continue until
        the round is complete.
        
        Returns:
            bool: True if exploration can continue
        """
        # If budget is not exhausted, exploration can continue
        if not self.is_budget_exhausted():
            return True
        
        # If budget is exhausted but we're in the middle of a round,
        # allow the round to complete
        return self.moves_in_current_round > 0
    
    def get_budget_status(self) -> dict:
        """
        Get comprehensive budget status information.
        
        Returns:
            dict: Dictionary containing budget status information
        """
        return {
            "initial_budget": self.initial_budget,
            "current_budget": self.current_budget,
            "budget_consumed": self.get_budget_consumed(),
            "total_moves_consumed": self.total_moves_consumed,
            "moves_in_current_round": self.moves_in_current_round,
            "round_number": self.round_number,
            "is_exhausted": self.is_budget_exhausted(),
            "can_continue": self.can_continue_exploration(),
            "utilization_rate": (self.get_budget_consumed() / self.initial_budget) * 100 if self.initial_budget > 0 else 0
        }
    
    def get_budget_utilization_patterns(self) -> dict:
        """
        Get detailed budget utilization patterns and efficiency metrics.
        
        Returns:
            dict: Dictionary containing utilization patterns and metrics
        """
        return {
            "efficiency_metrics": self.efficiency_metrics.copy(),
            "budget_history": self.budget_history.copy(),
            "round_history": self.round_history.copy(),
            "current_utilization_rate": (self.get_budget_consumed() / self.initial_budget) * 100 if self.initial_budget > 0 else 0,
            "rounds_completed": len(self.round_history),
            "average_budget_per_round": self.initial_budget / max(1, len(self.round_history)) if self.round_history else self.initial_budget
        }
    
    def log_budget_summary(self) -> None:
        """
        Log a comprehensive budget utilization summary.
        """
        patterns = self.get_budget_utilization_patterns()
        status = self.get_budget_status()
        
        self.logger.log_system_event("Budget utilization summary", {
            "total_budget": self.initial_budget,
            "consumed": status["budget_consumed"],
            "utilization_rate": f"{status['utilization_rate']:.1f}%",
            "rounds_completed": patterns["rounds_completed"],
            "avg_moves_per_round": patterns["efficiency_metrics"]["avg_moves_per_round"],
            "peak_round_size": patterns["efficiency_metrics"]["peak_round_size"],
            "budget_efficiency": patterns["efficiency_metrics"]["budget_utilization_rate"]
        })
    
    def _update_efficiency_metrics(self) -> None:
        """Update efficiency metrics based on current round history."""
        if not self.round_history:
            return
        
        total_moves = sum(r['moves_in_round'] for r in self.round_history)
        total_rounds = len(self.round_history)
        
        self.efficiency_metrics.update({
            'total_rounds': total_rounds,
            'avg_moves_per_round': total_moves / total_rounds if total_rounds > 0 else 0.0,
            'budget_utilization_rate': (self.total_moves_consumed / self.initial_budget) * 100 if self.initial_budget > 0 else 0.0,
            'peak_round_size': max(r['moves_in_round'] for r in self.round_history) if self.round_history else 0
        })
    
    def detect_and_correct_inconsistencies(self) -> bool:
        """
        Detect and correct budget tracking inconsistencies.
        
        Returns:
            bool: True if inconsistencies were found and corrected, False if all consistent
        """
        inconsistencies_found = False
        
        # Check arithmetic consistency
        expected_consumed = self.initial_budget - self.current_budget
        if expected_consumed != self.total_moves_consumed:
            self.logger.log_error("BudgetController", 
                f"Arithmetic inconsistency detected: expected {expected_consumed}, actual {self.total_moves_consumed}")
            self.total_moves_consumed = expected_consumed
            inconsistencies_found = True
        
        # Check for impossible budget values
        if self.current_budget > self.initial_budget:
            self.logger.log_error("BudgetController", 
                f"Impossible budget state: current {self.current_budget} > initial {self.initial_budget}")
            self.current_budget = self.initial_budget
            self.total_moves_consumed = 0
            inconsistencies_found = True
        
        # Check for extreme negative values (beyond reasonable overrun)
        max_reasonable_overrun = self.initial_budget * 2  # Allow 2x overrun as maximum
        if self.current_budget < -max_reasonable_overrun:
            self.logger.log_error("BudgetController", 
                f"Extreme budget overrun detected: {self.current_budget}")
            self.current_budget = -max_reasonable_overrun
            self.total_moves_consumed = self.initial_budget + max_reasonable_overrun
            inconsistencies_found = True
        
        # Check round consistency
        if self.moves_in_current_round < 0:
            self.logger.log_error("BudgetController", 
                f"Negative moves in round: {self.moves_in_current_round}")
            self.moves_in_current_round = 0
            inconsistencies_found = True
        
        if inconsistencies_found:
            self.logger.log_system_event("Budget inconsistencies corrected", {
                "current_budget": self.current_budget,
                "total_consumed": self.total_moves_consumed,
                "moves_in_round": self.moves_in_current_round
            })
        
        return inconsistencies_found
    
    def _reset_efficiency_metrics(self) -> None:
        """Reset efficiency metrics to initial state."""
        self.efficiency_metrics = {
            'total_rounds': 0,
            'avg_moves_per_round': 0.0,
            'budget_utilization_rate': 0.0,
            'peak_round_size': 0
        }