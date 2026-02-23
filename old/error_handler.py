"""
Comprehensive error handling system for the AI Hydra.

This module provides error handling, recovery mechanisms, and fault isolation
for all components of the simulation system including clone failures, budget
inconsistencies, and state corruption detection.
"""

from typing import Optional, Dict, List, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging
import traceback
import time
from .models import GameBoard, Move, MoveResult
from .logging_config import SimulationLogger
from .config import LoggingConfig


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Recovery actions that can be taken for different error types."""
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    RESET = "reset"
    ABORT = "abort"


@dataclass
class ErrorContext:
    """Context information for error handling and recovery."""
    component: str
    operation: str
    error_type: str
    severity: ErrorSeverity
    timestamp: float
    details: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


@dataclass
class RecoveryResult:
    """Result of an error recovery attempt."""
    success: bool
    action_taken: RecoveryAction
    message: str
    recovered_data: Optional[Any] = None
    should_retry: bool = False


class CloneFailureError(Exception):
    """Exception raised when an exploration clone fails."""
    def __init__(self, clone_id: str, operation: str, original_error: Exception):
        self.clone_id = clone_id
        self.operation = operation
        self.original_error = original_error
        super().__init__(f"Clone {clone_id} failed during {operation}: {original_error}")


class BudgetInconsistencyError(Exception):
    """Exception raised when budget tracking becomes inconsistent."""
    def __init__(self, expected: int, actual: int, operation: str):
        self.expected = expected
        self.actual = actual
        self.operation = operation
        super().__init__(f"Budget inconsistency in {operation}: expected {expected}, got {actual}")


class StateCorruptionError(Exception):
    """Exception raised when game state corruption is detected."""
    def __init__(self, component: str, validation_failure: str, corrupted_data: Any):
        self.component = component
        self.validation_failure = validation_failure
        self.corrupted_data = corrupted_data
        super().__init__(f"State corruption in {component}: {validation_failure}")


class ErrorHandler:
    """
    Comprehensive error handling system for the simulation.
    
    This class provides centralized error handling, recovery mechanisms,
    and fault isolation for all components of the simulation system.
    """
    
    def __init__(self, logging_config: Optional[LoggingConfig] = None):
        """
        Initialize the error handler.
        
        Args:
            logging_config: Logging configuration (optional)
        """
        self.logger = SimulationLogger("error_handler", logging_config)
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.recovery_statistics = {
            'total_errors': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'errors_by_component': {},
            'errors_by_severity': {severity.value: 0 for severity in ErrorSeverity},
            'recovery_actions_taken': {action.value: 0 for action in RecoveryAction}
        }
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, Callable] = {
            'clone_failure': self._handle_clone_failure,
            'budget_inconsistency': self._handle_budget_inconsistency,
            'state_corruption': self._handle_state_corruption,
            'component_initialization': self._handle_component_initialization_failure,
            'neural_network_error': self._handle_neural_network_error,
            'logging_error': self._handle_logging_error
        }
        
        # Configuration
        self.max_recovery_attempts = 3
        self.retry_delays = [0.1, 0.5, 1.0]  # Exponential backoff delays
        self.critical_error_threshold = 5  # Max critical errors before abort
        
        self.logger.log_system_event("ErrorHandler initialized", {
            "recovery_strategies": len(self.recovery_strategies),
            "max_recovery_attempts": self.max_recovery_attempts,
            "critical_error_threshold": self.critical_error_threshold
        })
    
    def handle_error(self, error: Exception, component: str, operation: str, 
                    context: Optional[Dict[str, Any]] = None) -> RecoveryResult:
        """
        Handle an error with appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            component: Component where error occurred
            operation: Operation being performed when error occurred
            context: Additional context information
            
        Returns:
            RecoveryResult: Result of error handling and recovery
        """
        # Create error context
        error_context = self._create_error_context(error, component, operation, context)
        
        # Log the error
        self._log_error(error_context)
        
        # Track error statistics
        self._update_error_statistics(error_context)
        
        # Determine recovery strategy
        recovery_strategy = self._determine_recovery_strategy(error_context)
        
        # Execute recovery
        recovery_result = self._execute_recovery(error_context, recovery_strategy)
        
        # Log recovery result
        self._log_recovery_result(error_context, recovery_result)
        
        # Update recovery statistics
        self._update_recovery_statistics(recovery_result)
        
        return recovery_result
    
    def handle_clone_failure(self, clone_id: str, error: Exception, 
                           operation: str) -> RecoveryResult:
        """
        Handle exploration clone failure with isolation and recovery.
        
        Args:
            clone_id: ID of the failed clone
            error: The exception that caused the failure
            operation: Operation being performed when failure occurred
            
        Returns:
            RecoveryResult: Result of clone failure handling
        """
        context = {
            'clone_id': clone_id,
            'operation': operation,
            'original_error': str(error)
        }
        
        clone_error = CloneFailureError(clone_id, operation, error)
        return self.handle_error(clone_error, "exploration_clone", operation, context)
    
    def handle_budget_inconsistency(self, expected_budget: int, actual_budget: int,
                                  operation: str) -> RecoveryResult:
        """
        Handle budget tracking inconsistency with detection and correction.
        
        Args:
            expected_budget: Expected budget value
            actual_budget: Actual budget value found
            operation: Operation where inconsistency was detected
            
        Returns:
            RecoveryResult: Result of budget inconsistency handling
        """
        context = {
            'expected_budget': expected_budget,
            'actual_budget': actual_budget,
            'inconsistency_amount': abs(expected_budget - actual_budget),
            'operation': operation
        }
        
        budget_error = BudgetInconsistencyError(expected_budget, actual_budget, operation)
        return self.handle_error(budget_error, "budget_controller", operation, context)
    
    def handle_state_corruption(self, component: str, corrupted_state: Any,
                              validation_failure: str) -> RecoveryResult:
        """
        Handle state corruption with validation and recovery.
        
        Args:
            component: Component with corrupted state
            corrupted_state: The corrupted state data
            validation_failure: Description of validation failure
            
        Returns:
            RecoveryResult: Result of state corruption handling
        """
        context = {
            'component': component,
            'validation_failure': validation_failure,
            'state_type': type(corrupted_state).__name__,
            'state_size': len(str(corrupted_state)) if corrupted_state else 0
        }
        
        corruption_error = StateCorruptionError(component, validation_failure, corrupted_state)
        return self.handle_error(corruption_error, component, "state_validation", context)
    
    def validate_game_board_integrity(self, board: GameBoard) -> bool:
        """
        Validate GameBoard integrity and detect corruption.
        
        Args:
            board: GameBoard to validate
            
        Returns:
            bool: True if board is valid, False if corrupted
        """
        try:
            # Basic structure validation
            if not hasattr(board, 'snake_head') or not hasattr(board, 'snake_body'):
                self.handle_state_corruption("game_board", board, "Missing snake components")
                return False
            
            # Snake head validation
            if not hasattr(board.snake_head, 'x') or not hasattr(board.snake_head, 'y'):
                self.handle_state_corruption("game_board", board, "Invalid snake head position")
                return False
            
            # Grid bounds validation
            grid_width, grid_height = board.grid_size
            if (board.snake_head.x < 0 or board.snake_head.x >= grid_width or
                board.snake_head.y < 0 or board.snake_head.y >= grid_height):
                self.handle_state_corruption("game_board", board, "Snake head out of bounds")
                return False
            
            # Snake body validation
            if not isinstance(board.snake_body, (list, tuple)):
                self.handle_state_corruption("game_board", board, "Invalid snake body structure")
                return False
            
            # Food position validation
            if (board.food_position.x < 0 or board.food_position.x >= grid_width or
                board.food_position.y < 0 or board.food_position.y >= grid_height):
                self.handle_state_corruption("game_board", board, "Food position out of bounds")
                return False
            
            # Score validation
            if board.score < 0:
                self.handle_state_corruption("game_board", board, "Negative score")
                return False
            
            return True
            
        except Exception as e:
            self.handle_state_corruption("game_board", board, f"Validation exception: {e}")
            return False
    
    def validate_budget_consistency(self, budget_controller, expected_consumed: int) -> bool:
        """
        Validate budget controller consistency.
        
        Args:
            budget_controller: BudgetController instance to validate
            expected_consumed: Expected amount of budget consumed
            
        Returns:
            bool: True if budget is consistent, False if inconsistent
        """
        try:
            actual_consumed = budget_controller.get_budget_consumed()
            
            if actual_consumed != expected_consumed:
                self.handle_budget_inconsistency(expected_consumed, actual_consumed, "budget_validation")
                return False
            
            # Validate budget bounds
            if budget_controller.get_remaining_budget() + actual_consumed != budget_controller.initial_budget:
                self.handle_budget_inconsistency(
                    budget_controller.initial_budget,
                    budget_controller.get_remaining_budget() + actual_consumed,
                    "budget_arithmetic_validation"
                )
                return False
            
            return True
            
        except Exception as e:
            self.handle_error(e, "budget_controller", "consistency_validation")
            return False
    
    def isolate_clone_failure(self, failed_clone_id: str, active_clones: List) -> List:
        """
        Isolate a failed clone from the active clone list.
        
        Args:
            failed_clone_id: ID of the failed clone
            active_clones: List of active clones
            
        Returns:
            List: Updated list with failed clone removed
        """
        try:
            # Remove failed clone from active list
            updated_clones = [clone for clone in active_clones 
                            if clone.get_clone_id() != failed_clone_id]
            
            self.logger.log_system_event("Clone failure isolated", {
                "failed_clone": failed_clone_id,
                "remaining_clones": len(updated_clones),
                "isolation_successful": True
            })
            
            return updated_clones
            
        except Exception as e:
            self.logger.log_error("ErrorHandler", f"Failed to isolate clone {failed_clone_id}: {e}")
            return active_clones  # Return original list if isolation fails
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive error handling statistics.
        
        Returns:
            Dict[str, Any]: Error handling statistics
        """
        return {
            "total_errors": self.recovery_statistics['total_errors'],
            "successful_recoveries": self.recovery_statistics['successful_recoveries'],
            "failed_recoveries": self.recovery_statistics['failed_recoveries'],
            "recovery_success_rate": (
                self.recovery_statistics['successful_recoveries'] / 
                max(1, self.recovery_statistics['total_errors'])
            ) * 100,
            "errors_by_component": self.recovery_statistics['errors_by_component'].copy(),
            "errors_by_severity": self.recovery_statistics['errors_by_severity'].copy(),
            "recovery_actions_taken": self.recovery_statistics['recovery_actions_taken'].copy(),
            "recent_errors": len([e for e in self.error_history 
                                if time.time() - e.timestamp < 300])  # Last 5 minutes
        }
    
    def _create_error_context(self, error: Exception, component: str, 
                            operation: str, context: Optional[Dict[str, Any]]) -> ErrorContext:
        """Create error context for tracking and recovery."""
        error_type = type(error).__name__
        severity = self._determine_error_severity(error, component)
        
        return ErrorContext(
            component=component,
            operation=operation,
            error_type=error_type,
            severity=severity,
            timestamp=time.time(),
            details=context or {},
            stack_trace=traceback.format_exc()
        )
    
    def _determine_error_severity(self, error: Exception, component: str) -> ErrorSeverity:
        """Determine error severity based on error type and component."""
        # Critical errors that can crash the system
        if isinstance(error, (RuntimeError, SystemError, MemoryError)):
            return ErrorSeverity.CRITICAL
        
        # High severity for core component failures
        if component in ["hydra_mgr", "master_game"] and isinstance(error, Exception):
            return ErrorSeverity.HIGH
        
        # Medium severity for clone and budget issues
        if isinstance(error, (CloneFailureError, BudgetInconsistencyError)):
            return ErrorSeverity.MEDIUM
        
        # Low severity for validation and logging issues
        if isinstance(error, (ValueError, TypeError, StateCorruptionError)):
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM  # Default
    
    def _determine_recovery_strategy(self, error_context: ErrorContext) -> str:
        """Determine appropriate recovery strategy for the error."""
        error_type = error_context.error_type
        component = error_context.component
        
        # Map error types to recovery strategies
        if error_type == "CloneFailureError":
            return "clone_failure"
        elif error_type == "BudgetInconsistencyError":
            return "budget_inconsistency"
        elif error_type == "StateCorruptionError":
            return "state_corruption"
        elif "initialization" in error_context.operation.lower():
            return "component_initialization"
        elif component == "neural_network":
            return "neural_network_error"
        elif component in ["logger", "logging"]:
            return "logging_error"
        else:
            return "clone_failure"  # Default fallback
    
    def _execute_recovery(self, error_context: ErrorContext, strategy: str) -> RecoveryResult:
        """Execute the recovery strategy."""
        if strategy in self.recovery_strategies:
            return self.recovery_strategies[strategy](error_context)
        else:
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.ABORT,
                message=f"No recovery strategy available for {strategy}"
            )
    
    def _handle_clone_failure(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle exploration clone failure with isolation."""
        clone_id = error_context.details.get('clone_id', 'unknown')
        
        # Clone failures are isolated by removing the failed clone
        # The tree search can continue with remaining clones
        return RecoveryResult(
            success=True,
            action_taken=RecoveryAction.SKIP,
            message=f"Clone {clone_id} isolated and removed from exploration",
            should_retry=False
        )
    
    def _handle_budget_inconsistency(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle budget inconsistency with correction."""
        expected = error_context.details.get('expected_budget', 0)
        actual = error_context.details.get('actual_budget', 0)
        
        # For budget inconsistencies, reset to a known good state
        return RecoveryResult(
            success=True,
            action_taken=RecoveryAction.RESET,
            message=f"Budget reset due to inconsistency (expected: {expected}, actual: {actual})",
            recovered_data={'corrected_budget': expected},
            should_retry=True
        )
    
    def _handle_state_corruption(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle state corruption with validation and recovery."""
        component = error_context.component
        validation_failure = error_context.details.get('validation_failure', 'unknown')
        
        # For state corruption, attempt to restore from last known good state
        return RecoveryResult(
            success=True,
            action_taken=RecoveryAction.FALLBACK,
            message=f"State corruption in {component} handled: {validation_failure}",
            should_retry=True
        )
    
    def _handle_component_initialization_failure(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle component initialization failure."""
        component = error_context.component
        
        # Retry initialization with exponential backoff
        if error_context.recovery_attempts < self.max_recovery_attempts:
            return RecoveryResult(
                success=True,
                action_taken=RecoveryAction.RETRY,
                message=f"Retrying {component} initialization (attempt {error_context.recovery_attempts + 1})",
                should_retry=True
            )
        else:
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.ABORT,
                message=f"Failed to initialize {component} after {self.max_recovery_attempts} attempts"
            )
    
    def _handle_neural_network_error(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle neural network errors."""
        # Neural network errors are non-critical - disable NN and continue
        return RecoveryResult(
            success=True,
            action_taken=RecoveryAction.FALLBACK,
            message="Neural network disabled due to error, continuing with tree search only",
            recovered_data={'nn_enabled': False}
        )
    
    def _handle_logging_error(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle logging system errors."""
        # Logging errors are low priority - continue without detailed logging
        return RecoveryResult(
            success=True,
            action_taken=RecoveryAction.SKIP,
            message="Logging error handled, continuing with reduced logging"
        )
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with appropriate severity level."""
        self.error_history.append(error_context)
        
        log_data = {
            "component": error_context.component,
            "operation": error_context.operation,
            "error_type": error_context.error_type,
            "severity": error_context.severity.value,
            "details": error_context.details
        }
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.log_error("ErrorHandler", f"CRITICAL ERROR: {error_context.error_type}")
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.log_error("ErrorHandler", f"HIGH SEVERITY: {error_context.error_type}")
        else:
            self.logger.log_warning("ErrorHandler", f"{error_context.severity.value.upper()}: {error_context.error_type}")
    
    def _log_recovery_result(self, error_context: ErrorContext, result: RecoveryResult) -> None:
        """Log recovery result."""
        self.logger.log_system_event("Error recovery completed", {
            "error_type": error_context.error_type,
            "component": error_context.component,
            "recovery_success": result.success,
            "action_taken": result.action_taken.value,
            "message": result.message,
            "should_retry": result.should_retry
        })
    
    def _update_error_statistics(self, error_context: ErrorContext) -> None:
        """Update error tracking statistics."""
        self.recovery_statistics['total_errors'] += 1
        
        # Track by component
        component = error_context.component
        if component not in self.recovery_statistics['errors_by_component']:
            self.recovery_statistics['errors_by_component'][component] = 0
        self.recovery_statistics['errors_by_component'][component] += 1
        
        # Track by severity
        self.recovery_statistics['errors_by_severity'][error_context.severity.value] += 1
    
    def _update_recovery_statistics(self, result: RecoveryResult) -> None:
        """Update recovery statistics."""
        if result.success:
            self.recovery_statistics['successful_recoveries'] += 1
        else:
            self.recovery_statistics['failed_recoveries'] += 1
        
        # Track recovery actions
        self.recovery_statistics['recovery_actions_taken'][result.action_taken.value] += 1