"""
Test cases for the comprehensive error handling system.

This module tests error handling, recovery mechanisms, and fault isolation
for all components of the simulation system.
"""

import pytest
from unittest.mock import Mock, patch
from ai_hydra.error_handler import (
    ErrorHandler, ErrorSeverity, RecoveryAction, 
    CloneFailureError, BudgetInconsistencyError, StateCorruptionError
)
from ai_hydra.models import GameBoard, Position, Direction
from ai_hydra.config import LoggingConfig


class TestErrorHandler:
    """Test cases for the ErrorHandler class."""
    
    def test_error_handler_initialization(self):
        """Test that ErrorHandler initializes correctly."""
        handler = ErrorHandler()
        
        assert handler is not None
        assert len(handler.recovery_strategies) > 0
        assert handler.max_recovery_attempts == 3
        assert handler.critical_error_threshold == 5
    
    def test_clone_failure_handling(self):
        """Test handling of exploration clone failures."""
        handler = ErrorHandler()
        
        # Create a mock clone failure
        original_error = RuntimeError("Clone execution failed")
        result = handler.handle_clone_failure("1L", original_error, "move_execution")
        
        assert result.success is True
        assert result.action_taken == RecoveryAction.SKIP
        assert "isolated" in result.message.lower()
    
    def test_budget_inconsistency_handling(self):
        """Test handling of budget inconsistencies."""
        handler = ErrorHandler()
        
        # Test budget inconsistency
        result = handler.handle_budget_inconsistency(100, 85, "budget_validation")
        
        assert result.success is True
        assert result.action_taken == RecoveryAction.RESET
        assert result.should_retry is True
        assert result.recovered_data is not None
    
    def test_state_corruption_handling(self):
        """Test handling of state corruption."""
        handler = ErrorHandler()
        
        # Create corrupted state
        corrupted_board = Mock()
        result = handler.handle_state_corruption(
            "game_board", corrupted_board, "Invalid snake head position"
        )
        
        assert result.success is True
        assert result.action_taken == RecoveryAction.FALLBACK
        assert result.should_retry is True
    
    def test_game_board_integrity_validation(self):
        """Test GameBoard integrity validation."""
        handler = ErrorHandler()
        
        # Create a valid board
        valid_board = GameBoard(
            snake_head=Position(5, 5),
            snake_body=(Position(5, 4), Position(5, 3)),
            direction=Direction.up(),
            food_position=Position(10, 10),
            score=2,
            move_count=0,
            random_state=None,
            grid_size=(20, 20)
        )
        
        # Should pass validation
        assert handler.validate_game_board_integrity(valid_board) is True
    
    def test_game_board_corruption_detection(self):
        """Test detection of corrupted GameBoard."""
        handler = ErrorHandler()
        
        # Create a board with out-of-bounds snake head
        corrupted_board = GameBoard(
            snake_head=Position(-1, 5),  # Out of bounds
            snake_body=(Position(5, 4), Position(5, 3)),
            direction=Direction.up(),
            food_position=Position(10, 10),
            score=2,
            move_count=0,
            random_state=None,
            grid_size=(20, 20)
        )
        
        # Should fail validation
        assert handler.validate_game_board_integrity(corrupted_board) is False
    
    def test_budget_consistency_validation(self):
        """Test budget consistency validation."""
        handler = ErrorHandler()
        
        # Create mock budget controller
        mock_budget = Mock()
        mock_budget.get_budget_consumed.return_value = 50
        mock_budget.get_remaining_budget.return_value = 50
        mock_budget.initial_budget = 100
        
        # Should pass validation
        assert handler.validate_budget_consistency(mock_budget, 50) is True
        
        # Test inconsistency
        assert handler.validate_budget_consistency(mock_budget, 75) is False
    
    def test_clone_failure_isolation(self):
        """Test isolation of failed clones."""
        handler = ErrorHandler()
        
        # Create mock clones
        mock_clone1 = Mock()
        mock_clone1.get_clone_id.return_value = "1L"
        
        mock_clone2 = Mock()
        mock_clone2.get_clone_id.return_value = "2S"
        
        mock_clone3 = Mock()
        mock_clone3.get_clone_id.return_value = "3R"
        
        active_clones = [mock_clone1, mock_clone2, mock_clone3]
        
        # Isolate failed clone
        updated_clones = handler.isolate_clone_failure("2S", active_clones)
        
        assert len(updated_clones) == 2
        assert all(clone.get_clone_id() != "2S" for clone in updated_clones)
    
    def test_error_statistics_tracking(self):
        """Test error statistics tracking."""
        handler = ErrorHandler()
        
        # Generate some errors
        handler.handle_clone_failure("1L", RuntimeError("Test error"), "test_operation")
        handler.handle_budget_inconsistency(100, 90, "test_validation")
        
        stats = handler.get_error_statistics()
        
        assert stats["total_errors"] == 2
        assert stats["successful_recoveries"] >= 0
        assert "exploration_clone" in stats["errors_by_component"]
        assert "budget_controller" in stats["errors_by_component"]
    
    def test_error_severity_determination(self):
        """Test error severity classification."""
        handler = ErrorHandler()
        
        # Test different error types
        runtime_error = RuntimeError("Critical system error")
        severity = handler._determine_error_severity(runtime_error, "hydra_mgr")
        assert severity == ErrorSeverity.CRITICAL
        
        clone_error = CloneFailureError("1L", "move_execution", ValueError("Invalid move"))
        severity = handler._determine_error_severity(clone_error, "exploration_clone")
        assert severity == ErrorSeverity.MEDIUM
        
        value_error = ValueError("Invalid parameter")
        severity = handler._determine_error_severity(value_error, "logging")
        assert severity == ErrorSeverity.LOW
    
    def test_recovery_strategy_selection(self):
        """Test recovery strategy selection."""
        handler = ErrorHandler()
        
        # Test clone failure strategy
        clone_context = Mock()
        clone_context.error_type = "CloneFailureError"
        clone_context.component = "exploration_clone"
        clone_context.operation = "move_execution"
        
        strategy = handler._determine_recovery_strategy(clone_context)
        assert strategy == "clone_failure"
        
        # Test budget inconsistency strategy
        budget_context = Mock()
        budget_context.error_type = "BudgetInconsistencyError"
        budget_context.component = "budget_controller"
        budget_context.operation = "budget_validation"
        
        strategy = handler._determine_recovery_strategy(budget_context)
        assert strategy == "budget_inconsistency"


class TestSpecificErrorTypes:
    """Test cases for specific error types and their handling."""
    
    def test_clone_failure_error(self):
        """Test CloneFailureError creation and properties."""
        original_error = ValueError("Invalid move")
        clone_error = CloneFailureError("1L", "move_execution", original_error)
        
        assert clone_error.clone_id == "1L"
        assert clone_error.operation == "move_execution"
        assert clone_error.original_error == original_error
        assert "Clone 1L failed during move_execution" in str(clone_error)
    
    def test_budget_inconsistency_error(self):
        """Test BudgetInconsistencyError creation and properties."""
        budget_error = BudgetInconsistencyError(100, 85, "budget_validation")
        
        assert budget_error.expected == 100
        assert budget_error.actual == 85
        assert budget_error.operation == "budget_validation"
        assert "expected 100, got 85" in str(budget_error)
    
    def test_state_corruption_error(self):
        """Test StateCorruptionError creation and properties."""
        corrupted_data = {"invalid": "state"}
        corruption_error = StateCorruptionError(
            "game_board", "Invalid snake position", corrupted_data
        )
        
        assert corruption_error.component == "game_board"
        assert corruption_error.validation_failure == "Invalid snake position"
        assert corruption_error.corrupted_data == corrupted_data
        assert "State corruption in game_board" in str(corruption_error)


class TestErrorHandlingIntegration:
    """Integration tests for error handling with other components."""
    
    @patch('ai_hydra.error_handler.SimulationLogger')
    def test_error_handler_logging_integration(self, mock_logger_class):
        """Test that error handler integrates properly with logging."""
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger
        
        handler = ErrorHandler()
        
        # Generate an error
        handler.handle_clone_failure("1L", RuntimeError("Test error"), "test_operation")
        
        # Verify logging was called
        assert mock_logger.log_system_event.called
    
    def test_error_recovery_with_retry_logic(self):
        """Test error recovery with retry logic."""
        handler = ErrorHandler()
        
        # Test component initialization failure with retry
        context = Mock()
        context.error_type = "RuntimeError"
        context.component = "neural_network"
        context.operation = "initialization"
        context.recovery_attempts = 1
        context.details = {}
        
        result = handler._handle_component_initialization_failure(context)
        
        assert result.success is True
        assert result.action_taken == RecoveryAction.RETRY
        assert result.should_retry is True
    
    def test_error_recovery_max_attempts_exceeded(self):
        """Test error recovery when max attempts are exceeded."""
        handler = ErrorHandler()
        
        # Test component initialization failure with max attempts exceeded
        context = Mock()
        context.error_type = "RuntimeError"
        context.component = "neural_network"
        context.operation = "initialization"
        context.recovery_attempts = 3  # Max attempts reached
        context.details = {}
        
        result = handler._handle_component_initialization_failure(context)
        
        assert result.success is False
        assert result.action_taken == RecoveryAction.ABORT
        assert result.should_retry is False


class TestCloneFailureIsolationScenarios:
    """Unit tests for clone failure isolation scenarios."""
    
    def test_single_clone_failure_isolation(self):
        """Test isolation of a single failed clone from active list."""
        handler = ErrorHandler()
        
        # Create mock clones
        mock_clones = []
        for i, clone_id in enumerate(["1L", "2S", "3R", "1LL", "2SL"]):
            mock_clone = Mock()
            mock_clone.get_clone_id.return_value = clone_id
            mock_clones.append(mock_clone)
        
        # Isolate one clone
        updated_clones = handler.isolate_clone_failure("2S", mock_clones)
        
        assert len(updated_clones) == 4
        assert all(clone.get_clone_id() != "2S" for clone in updated_clones)
        assert any(clone.get_clone_id() == "1L" for clone in updated_clones)
        assert any(clone.get_clone_id() == "3R" for clone in updated_clones)
    
    def test_multiple_clone_failure_isolation(self):
        """Test isolation of multiple failed clones sequentially."""
        handler = ErrorHandler()
        
        # Create mock clones
        mock_clones = []
        for clone_id in ["1L", "2S", "3R", "1LL", "2SL", "3RL"]:
            mock_clone = Mock()
            mock_clone.get_clone_id.return_value = clone_id
            mock_clones.append(mock_clone)
        
        # Isolate multiple clones
        updated_clones = handler.isolate_clone_failure("2S", mock_clones)
        updated_clones = handler.isolate_clone_failure("1LL", updated_clones)
        updated_clones = handler.isolate_clone_failure("3RL", updated_clones)
        
        assert len(updated_clones) == 3
        remaining_ids = [clone.get_clone_id() for clone in updated_clones]
        assert "1L" in remaining_ids
        assert "3R" in remaining_ids
        assert "2SL" in remaining_ids
        assert "2S" not in remaining_ids
        assert "1LL" not in remaining_ids
        assert "3RL" not in remaining_ids
    
    def test_clone_failure_isolation_with_empty_list(self):
        """Test clone failure isolation with empty clone list."""
        handler = ErrorHandler()
        
        updated_clones = handler.isolate_clone_failure("1L", [])
        
        assert len(updated_clones) == 0
    
    def test_clone_failure_isolation_nonexistent_clone(self):
        """Test isolation attempt for non-existent clone."""
        handler = ErrorHandler()
        
        # Create mock clones
        mock_clones = []
        for clone_id in ["1L", "2S", "3R"]:
            mock_clone = Mock()
            mock_clone.get_clone_id.return_value = clone_id
            mock_clones.append(mock_clone)
        
        # Try to isolate non-existent clone
        updated_clones = handler.isolate_clone_failure("4X", mock_clones)
        
        # Should return original list unchanged
        assert len(updated_clones) == 3
        assert all(clone.get_clone_id() in ["1L", "2S", "3R"] for clone in updated_clones)
    
    def test_clone_failure_cascade_handling(self):
        """Test handling of cascading clone failures."""
        handler = ErrorHandler()
        
        # Simulate multiple clone failures in sequence
        failures = [
            ("1L", RuntimeError("Network timeout")),
            ("2S", ValueError("Invalid state")),
            ("3R", ConnectionError("Connection lost"))
        ]
        
        for clone_id, error in failures:
            result = handler.handle_clone_failure(clone_id, error, "move_execution")
            assert result.success is True
            assert result.action_taken == RecoveryAction.SKIP
        
        # Verify error statistics track multiple failures
        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 3
        assert stats["errors_by_component"]["exploration_clone"] == 3


class TestBudgetTrackingConsistency:
    """Unit tests for budget tracking consistency scenarios."""
    
    def test_budget_arithmetic_consistency_validation(self):
        """Test validation of budget arithmetic consistency."""
        handler = ErrorHandler()
        
        # Create mock budget controller with consistent state
        mock_budget = Mock()
        mock_budget.get_budget_consumed.return_value = 50
        mock_budget.get_remaining_budget.return_value = 50
        mock_budget.initial_budget = 100
        
        # Should pass validation
        assert handler.validate_budget_consistency(mock_budget, 50) is True
    
    def test_budget_arithmetic_inconsistency_detection(self):
        """Test detection of budget arithmetic inconsistencies."""
        handler = ErrorHandler()
        
        # Create mock budget controller with inconsistent state
        mock_budget = Mock()
        mock_budget.get_budget_consumed.return_value = 60  # Inconsistent
        mock_budget.get_remaining_budget.return_value = 50
        mock_budget.initial_budget = 100
        
        # Should fail validation and trigger error handling
        assert handler.validate_budget_consistency(mock_budget, 45) is False
    
    def test_budget_bounds_validation(self):
        """Test validation of budget bounds."""
        handler = ErrorHandler()
        
        # Create mock budget controller with invalid bounds
        mock_budget = Mock()
        mock_budget.get_budget_consumed.return_value = 30
        mock_budget.get_remaining_budget.return_value = 80  # 30 + 80 = 110 > 100
        mock_budget.initial_budget = 100
        
        # Should fail validation
        assert handler.validate_budget_consistency(mock_budget, 30) is False
    
    def test_negative_budget_handling(self):
        """Test handling of negative budget values."""
        handler = ErrorHandler()
        
        # Test budget inconsistency with negative values
        result = handler.handle_budget_inconsistency(-10, 50, "budget_overrun")
        
        assert result.success is True
        assert result.action_taken == RecoveryAction.RESET
        assert result.should_retry is True
        assert result.recovered_data is not None
    
    def test_budget_consistency_with_overrun(self):
        """Test budget consistency validation with allowed overrun."""
        handler = ErrorHandler()
        
        # Create mock budget controller with overrun (negative remaining)
        mock_budget = Mock()
        mock_budget.get_budget_consumed.return_value = 120  # Overrun
        mock_budget.get_remaining_budget.return_value = -20  # Negative
        mock_budget.initial_budget = 100
        
        # Should still pass validation (overrun is allowed)
        assert handler.validate_budget_consistency(mock_budget, 120) is True
    
    def test_budget_tracking_edge_cases(self):
        """Test budget tracking with edge case values."""
        handler = ErrorHandler()
        
        # Test with zero budget
        result = handler.handle_budget_inconsistency(0, 5, "zero_budget_test")
        assert result.success is True
        
        # Test with very large budget values
        result = handler.handle_budget_inconsistency(10000, 9999, "large_budget_test")
        assert result.success is True
        
        # Test with identical values (should not trigger inconsistency)
        result = handler.handle_budget_inconsistency(100, 100, "identical_values")
        assert result.success is True


class TestStateCorruptionDetection:
    """Unit tests for state corruption detection scenarios."""
    
    def test_game_board_missing_components(self):
        """Test detection of GameBoard with missing components."""
        handler = ErrorHandler()
        
        # Create board with missing snake_head attribute
        corrupted_board = Mock()
        del corrupted_board.snake_head  # Remove attribute
        corrupted_board.snake_body = []
        
        assert handler.validate_game_board_integrity(corrupted_board) is False
    
    def test_game_board_invalid_snake_head(self):
        """Test detection of GameBoard with invalid snake head."""
        handler = ErrorHandler()
        
        # Create board with invalid snake head (missing coordinates)
        corrupted_board = Mock()
        corrupted_board.snake_head = Mock()
        del corrupted_board.snake_head.x  # Remove x coordinate
        corrupted_board.snake_body = []
        
        assert handler.validate_game_board_integrity(corrupted_board) is False
    
    def test_game_board_out_of_bounds_positions(self):
        """Test detection of out-of-bounds positions."""
        handler = ErrorHandler()
        
        # Test snake head out of bounds
        corrupted_board = GameBoard(
            snake_head=Position(25, 5),  # Out of bounds (grid is 20x20)
            snake_body=(Position(5, 4), Position(5, 3)),
            direction=Direction.up(),
            food_position=Position(10, 10),
            score=2,
            move_count=0,
            random_state=None,
            grid_size=(20, 20)
        )
        
        assert handler.validate_game_board_integrity(corrupted_board) is False
        
        # Test food position out of bounds
        corrupted_board2 = GameBoard(
            snake_head=Position(5, 5),
            snake_body=(Position(5, 4), Position(5, 3)),
            direction=Direction.up(),
            food_position=Position(-1, 10),  # Out of bounds
            score=2,
            move_count=0,
            random_state=None,
            grid_size=(20, 20)
        )
        
        assert handler.validate_game_board_integrity(corrupted_board2) is False
    
    def test_game_board_invalid_snake_body(self):
        """Test detection of invalid snake body structure."""
        handler = ErrorHandler()
        
        # Create board with invalid snake body type
        corrupted_board = GameBoard(
            snake_head=Position(5, 5),
            snake_body="invalid_body",  # Should be tuple/list
            direction=Direction.up(),
            food_position=Position(10, 10),
            score=2,
            move_count=0,
            random_state=None,
            grid_size=(20, 20)
        )
        
        assert handler.validate_game_board_integrity(corrupted_board) is False
    
    def test_game_board_negative_score(self):
        """Test detection of negative score."""
        handler = ErrorHandler()
        
        corrupted_board = GameBoard(
            snake_head=Position(5, 5),
            snake_body=(Position(5, 4), Position(5, 3)),
            direction=Direction.up(),
            food_position=Position(10, 10),
            score=-5,  # Invalid negative score
            move_count=0,
            random_state=None,
            grid_size=(20, 20)
        )
        
        assert handler.validate_game_board_integrity(corrupted_board) is False
    
    def test_state_corruption_recovery(self):
        """Test state corruption recovery mechanisms."""
        handler = ErrorHandler()
        
        # Test state corruption handling
        corrupted_data = {"snake_head": None, "invalid": True}
        result = handler.handle_state_corruption(
            "game_board", 
            corrupted_data, 
            "Snake head is None"
        )
        
        assert result.success is True
        assert result.action_taken == RecoveryAction.FALLBACK
        assert result.should_retry is True
        assert "State corruption in game_board handled" in result.message
    
    def test_validation_exception_handling(self):
        """Test handling of exceptions during validation."""
        handler = ErrorHandler()
        
        # Create a board that will cause an exception during validation
        corrupted_board = Mock()
        corrupted_board.snake_head.x = Mock(side_effect=Exception("Validation error"))
        
        # Should handle the exception gracefully
        assert handler.validate_game_board_integrity(corrupted_board) is False
    
    def test_multiple_corruption_types(self):
        """Test detection of multiple types of corruption in sequence."""
        handler = ErrorHandler()
        
        corruption_scenarios = [
            ("game_board", {"head": None}, "Missing snake head"),
            ("budget_controller", {"budget": -1000}, "Extreme negative budget"),
            ("state_manager", {"clones": "corrupted"}, "Invalid clone structure"),
            ("neural_network", {"weights": float('inf')}, "Infinite weight values")
        ]
        
        for component, data, failure in corruption_scenarios:
            result = handler.handle_state_corruption(component, data, failure)
            assert result.success is True
            assert result.action_taken == RecoveryAction.FALLBACK
        
        # Verify all corruptions were tracked
        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 4