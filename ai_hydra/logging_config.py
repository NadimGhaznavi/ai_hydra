"""
Logging configuration and utilities for the AI Hydra.

This module provides comprehensive logging setup for all system components,
including structured logging for clone steps, decision cycles, and neural
network training progress.
"""

import logging
import logging.config
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from .config import LoggingConfig


class SimulationLogger:
    """
    Specialized logger for Snake game simulation events.
    
    This logger provides structured logging for clone steps, decision cycles,
    neural network predictions, and system events with consistent formatting.
    """
    
    def __init__(self, name: str = "simulation", config: Optional[LoggingConfig] = None):
        """
        Initialize the simulation logger.
        
        Args:
            name: Logger name
            config: Logging configuration (uses default if None)
        """
        self.config = config or LoggingConfig()
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up the logger with the specified configuration."""
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        level = getattr(logging, self.config.level.upper())
        self.logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(self.config.format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def log_clone_step(self, clone_id: str, result: str, reward: int, score: int) -> None:
        """
        Log a single clone step execution.
        
        Args:
            clone_id: Identifier of the clone (e.g., "1L", "2S", "3R")
            result: Step result ("EMPTY", "FOOD", "WALL", "SNAKE")
            reward: Reward earned from this step
            score: Current total score
        """
        if self.config.log_clone_steps:
            self.logger.info(f"{clone_id} RESULT:{result} REWARD:{reward} SCORE:{score}")
    
    def log_decision_cycle(self, cycle_number: int, winning_clone: str, 
                          total_paths: int, budget_used: int) -> None:
        """
        Log the completion of a decision cycle.
        
        Args:
            cycle_number: The decision cycle number
            winning_clone: ID of the clone with the optimal path
            total_paths: Total number of paths evaluated
            budget_used: Amount of budget consumed
        """
        if self.config.log_decision_cycles:
            self.logger.info(
                f"CYCLE:{cycle_number} WINNER:{winning_clone} "
                f"PATHS:{total_paths} BUDGET:{budget_used}"
            )
    
    def log_master_move(self, move: str, new_score: int) -> None:
        """
        Log a move applied to the master game.
        
        Args:
            move: The move applied ("L", "S", "R")
            new_score: The new score after the move
        """
        self.logger.info(f"MASTER MOVE:{move} SCORE:{new_score}")
    
    def log_neural_network_prediction(self, prediction: str, confidence: float, score: int) -> None:
        """
        Log a neural network prediction.
        
        Args:
            prediction: The predicted move ("LEFT", "STRAIGHT", "RIGHT")
            confidence: Confidence score (0.0 to 1.0)
            score: Current game score
        """
        if self.config.log_neural_network:
            self.logger.info(f"NN PREDICTION:{prediction} CONFIDENCE:{confidence:.3f} SCORE:{score}")
    
    def log_oracle_decision(self, nn_move: str, optimal_move: str, final_decision: str, score: int) -> None:
        """
        Log an oracle training decision.
        
        Args:
            nn_move: Neural network predicted move
            optimal_move: Tree search optimal move
            final_decision: Final decision made by the system
            score: Current game score
        """
        if self.config.log_neural_network:
            self.logger.info(
                f"ORACLE: NN={nn_move} OPTIMAL={optimal_move} DECISION={final_decision} SCORE:{score}"
            )
    
    def log_training_sample(self, was_nn_wrong: bool, score: int) -> None:
        """
        Log neural network training sample generation.
        
        Args:
            was_nn_wrong: Whether the NN prediction was incorrect
            score: Current game score
        """
        if self.config.log_neural_network:
            status = "NN_WRONG" if was_nn_wrong else "NN_CORRECT"
            self.logger.info(f"TRAINING: SAMPLE_GENERATED {status} SCORE:{score}")
    
    def log_training_update(self, accuracy: float, samples_processed: int) -> None:
        """
        Log neural network training update.
        
        Args:
            accuracy: Current prediction accuracy
            samples_processed: Number of training samples processed
        """
        if self.config.log_neural_network:
            self.logger.info(
                f"TRAINING: UPDATE accuracy={accuracy:.3f} samples={samples_processed}"
            )
    
    def log_budget_status(self, remaining: int, total: int) -> None:
        """
        Log current budget status.
        
        Args:
            remaining: Remaining budget
            total: Total budget
        """
        self.logger.debug(f"BUDGET: {remaining}/{total} remaining")
    
    def log_tree_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log tree exploration metrics.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        metric_str = " ".join([f"{k}:{v}" for k, v in metrics.items()])
        self.logger.info(f"TREE_METRICS: {metric_str}")
    
    def log_error(self, component: str, error: str, details: Optional[str] = None) -> None:
        """
        Log an error with component context.
        
        Args:
            component: Component where error occurred
            error: Error message
            details: Optional additional details
        """
        message = f"ERROR in {component}: {error}"
        if details:
            message += f" - {details}"
        self.logger.error(message)
    
    def log_warning(self, component: str, warning: str) -> None:
        """
        Log a warning with component context.
        
        Args:
            component: Component issuing the warning
            warning: Warning message
        """
        self.logger.warning(f"WARNING in {component}: {warning}")
    
    def log_system_event(self, event: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a system-level event.
        
        Args:
            event: Event description
            details: Optional event details
        """
        message = f"SYSTEM: {event}"
        if details:
            detail_str = " ".join([f"{k}:{v}" for k, v in details.items()])
            message += f" - {detail_str}"
        self.logger.info(message)


def setup_logging(config: LoggingConfig) -> SimulationLogger:
    """
    Set up the main simulation logger with the given configuration.
    
    Args:
        config: Logging configuration
        
    Returns:
        SimulationLogger: Configured logger instance
    """
    return SimulationLogger("ai_hydra", config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a standard logger for a specific component.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(f"ai_hydra.{name}")


def configure_root_logging(level: str = "INFO") -> None:
    """
    Configure root logging for the entire application.
    
    Args:
        level: Log level to set
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


# Default logger instance
default_logger = SimulationLogger()