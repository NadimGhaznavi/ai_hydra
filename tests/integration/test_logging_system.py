"""
Unit tests for the logging system.

This module tests the SimulationLogger class and its integration with
the logging configuration system, focusing on log format consistency
and completeness.
"""

import pytest
import logging
import io
import sys
from unittest.mock import patch, MagicMock
from ai_hydra.logging_config import SimulationLogger, setup_logging, get_logger
from ai_hydra.config import LoggingConfig


class TestSimulationLogger:
    """Test cases for SimulationLogger class."""
    
    @pytest.fixture
    def logging_config(self):
        """Create a test logging configuration."""
        return LoggingConfig(
            level="INFO",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            log_clone_steps=True,
            log_decision_cycles=True,
            log_neural_network=True,
            log_file=None
        )
    
    @pytest.fixture
    def logger(self, logging_config):
        """Create a SimulationLogger instance for testing."""
        return SimulationLogger("test_logger", logging_config)
    
    @pytest.fixture
    def log_capture(self):
        """Capture log output for testing."""
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter("%(message)s"))
        return log_capture, handler
    
    def test_logger_initialization(self, logging_config):
        """Test logger initialization with configuration."""
        logger = SimulationLogger("test_init", logging_config)
        
        assert logger.config == logging_config
        assert logger.logger.name == "test_init"
        assert logger.logger.level == logging.INFO
        assert len(logger.logger.handlers) > 0
    
    def test_logger_initialization_default_config(self):
        """Test logger initialization with default configuration."""
        logger = SimulationLogger("test_default")
        
        assert logger.config is not None
        assert logger.logger.name == "test_default"
        assert len(logger.logger.handlers) > 0
    
    def test_log_clone_step_format(self, logger, log_capture):
        """Test clone step logging format consistency."""
        log_stream, handler = log_capture
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        # Test various clone step scenarios
        test_cases = [
            ("1L", "EMPTY", 0, 5),
            ("2S", "FOOD", 10, 15),
            ("3R", "WALL", -10, 5),
            ("1LL", "SNAKE", -10, 10),
            ("2SR", "EMPTY", 0, 20)
        ]
        
        for clone_id, result, reward, score in test_cases:
            log_stream.seek(0)
            log_stream.truncate(0)
            
            logger.log_clone_step(clone_id, result, reward, score)
            
            log_output = log_stream.getvalue().strip()
            expected = f"{clone_id} RESULT:{result} REWARD:{reward} SCORE:{score}"
            assert log_output == expected
    
    def test_log_decision_cycle_format(self, logger, log_capture):
        """Test decision cycle logging format consistency."""
        log_stream, handler = log_capture
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        logger.log_decision_cycle(5, "2S", 8, 45)
        
        log_output = log_stream.getvalue().strip()
        expected = "CYCLE:5 WINNER:2S PATHS:8 BUDGET:45"
        assert log_output == expected
    
    def test_log_master_move_format(self, logger, log_capture):
        """Test master move logging format."""
        log_stream, handler = log_capture
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        logger.log_master_move("L", 25)
        
        log_output = log_stream.getvalue().strip()
        expected = "MASTER MOVE:L SCORE:25"
        assert log_output == expected
    
    def test_log_neural_network_prediction_format(self, logger, log_capture):
        """Test neural network prediction logging format."""
        log_stream, handler = log_capture
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        logger.log_neural_network_prediction("LEFT", 0.85, 30)
        
        log_output = log_stream.getvalue().strip()
        expected = "NN PREDICTION:LEFT CONFIDENCE:0.850 SCORE:30"
        assert log_output == expected
    
    def test_log_oracle_decision_format(self, logger, log_capture):
        """Test oracle decision logging format."""
        log_stream, handler = log_capture
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        logger.log_oracle_decision("LEFT", "STRAIGHT", "STRAIGHT", 35)
        
        log_output = log_stream.getvalue().strip()
        expected = "ORACLE: NN=LEFT OPTIMAL=STRAIGHT DECISION=STRAIGHT SCORE:35"
        assert log_output == expected
    
    def test_log_training_sample_format(self, logger, log_capture):
        """Test training sample logging format."""
        log_stream, handler = log_capture
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        # Test NN wrong case
        logger.log_training_sample(True, 40)
        log_output = log_stream.getvalue().strip()
        expected = "TRAINING: SAMPLE_GENERATED NN_WRONG SCORE:40"
        assert log_output == expected
        
        # Test NN correct case
        log_stream.seek(0)
        log_stream.truncate(0)
        logger.log_training_sample(False, 45)
        log_output = log_stream.getvalue().strip()
        expected = "TRAINING: SAMPLE_GENERATED NN_CORRECT SCORE:45"
        assert log_output == expected
    
    def test_log_training_update_format(self, logger, log_capture):
        """Test training update logging format."""
        log_stream, handler = log_capture
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        logger.log_training_update(0.75, 12)
        
        log_output = log_stream.getvalue().strip()
        expected = "TRAINING: UPDATE accuracy=0.750 samples=12"
        assert log_output == expected
    
    def test_logging_configuration_flags(self, logging_config):
        """Test that logging configuration flags are respected."""
        # Test with clone steps disabled
        config_no_clone = LoggingConfig(
            level="INFO",
            log_clone_steps=False,
            log_decision_cycles=True,
            log_neural_network=True
        )
        
        logger = SimulationLogger("test_flags", config_no_clone)
        
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_clone_step("1L", "EMPTY", 0, 5)
            mock_info.assert_not_called()
        
        # Test with neural network logging disabled
        config_no_nn = LoggingConfig(
            level="INFO",
            log_clone_steps=True,
            log_decision_cycles=True,
            log_neural_network=False
        )
        
        logger_no_nn = SimulationLogger("test_no_nn", config_no_nn)
        
        with patch.object(logger_no_nn.logger, 'info') as mock_info:
            logger_no_nn.log_neural_network_prediction("LEFT", 0.85, 30)
            mock_info.assert_not_called()
            
            logger_no_nn.log_oracle_decision("LEFT", "STRAIGHT", "STRAIGHT", 35)
            mock_info.assert_not_called()
            
            logger_no_nn.log_training_sample(True, 40)
            mock_info.assert_not_called()
    
    def test_clone_id_generation_and_tracking(self, logger, log_capture):
        """Test clone ID generation and tracking consistency."""
        log_stream, handler = log_capture
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        # Test various clone ID formats
        clone_ids = [
            "1", "2", "3",           # Initial clones
            "1L", "1S", "1R",        # First level sub-clones
            "2L", "2S", "2R",        # First level sub-clones
            "1LL", "1LS", "1LR",     # Second level sub-clones
            "2SL", "2SS", "2SR",     # Second level sub-clones
            "1LLL", "1LLS", "1LLR"   # Third level sub-clones
        ]
        
        for clone_id in clone_ids:
            log_stream.seek(0)
            log_stream.truncate(0)
            
            logger.log_clone_step(clone_id, "EMPTY", 0, 10)
            
            log_output = log_stream.getvalue().strip()
            assert clone_id in log_output
            assert log_output.startswith(clone_id)
    
    def test_error_and_warning_logging(self, logger, log_capture):
        """Test error and warning logging functionality."""
        log_stream, handler = log_capture
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.WARNING)
        
        # Test error logging
        logger.log_error("TestComponent", "Test error message", "Additional details")
        log_output = log_stream.getvalue()
        assert "ERROR in TestComponent: Test error message - Additional details" in log_output
        
        # Test warning logging
        log_stream.seek(0)
        log_stream.truncate(0)
        logger.log_warning("TestComponent", "Test warning message")
        log_output = log_stream.getvalue()
        assert "WARNING in TestComponent: Test warning message" in log_output
    
    def test_system_event_logging(self, logger, log_capture):
        """Test system event logging functionality."""
        log_stream, handler = log_capture
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        # Test system event without details
        logger.log_system_event("Test system event")
        log_output = log_stream.getvalue()
        assert "SYSTEM: Test system event" in log_output
        
        # Test system event with details
        log_stream.seek(0)
        log_stream.truncate(0)
        details = {"param1": "value1", "param2": 42}
        logger.log_system_event("Test event with details", details)
        log_output = log_stream.getvalue()
        assert "SYSTEM: Test event with details" in log_output
        assert "param1:value1" in log_output
        assert "param2:42" in log_output
    
    def test_tree_metrics_logging(self, logger, log_capture):
        """Test tree metrics logging functionality."""
        log_stream, handler = log_capture
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        metrics = {
            "total_clones": 15,
            "max_depth": 4,
            "budget_used": 85,
            "paths_evaluated": 12
        }
        
        logger.log_tree_metrics(metrics)
        log_output = log_stream.getvalue()
        
        assert "TREE_METRICS:" in log_output
        for key, value in metrics.items():
            assert f"{key}:{value}" in log_output


class TestLoggingUtilities:
    """Test cases for logging utility functions."""
    
    def test_setup_logging(self):
        """Test setup_logging function."""
        config = LoggingConfig(level="DEBUG", log_file="test.log")
        logger = setup_logging(config)
        
        assert isinstance(logger, SimulationLogger)
        assert logger.config == config
        assert logger.logger.name == "ai_hydra"
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test_component")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "ai_hydra.test_component"
    
    def test_configure_root_logging(self):
        """Test configure_root_logging function."""
        from ai_hydra.logging_config import configure_root_logging
        
        # Test that it doesn't raise an exception
        try:
            configure_root_logging("DEBUG")
            # If we get here, the function worked
            assert True
        except Exception as e:
            pytest.fail(f"configure_root_logging raised an exception: {e}")


class TestLoggingIntegration:
    """Test cases for logging system integration."""
    
    def test_logging_with_file_output(self, tmp_path):
        """Test logging with file output."""
        log_file = tmp_path / "test.log"
        config = LoggingConfig(
            level="INFO",
            log_file=str(log_file),
            log_clone_steps=True
        )
        
        logger = SimulationLogger("file_test", config)
        logger.log_clone_step("1L", "EMPTY", 0, 5)
        
        # Force flush handlers
        for handler in logger.logger.handlers:
            handler.flush()
        
        assert log_file.exists()
        content = log_file.read_text()
        assert "1L RESULT:EMPTY REWARD:0 SCORE:5" in content
    
    def test_multiple_logger_instances(self):
        """Test multiple logger instances don't interfere."""
        config1 = LoggingConfig(level="INFO", log_clone_steps=True)
        config2 = LoggingConfig(level="DEBUG", log_clone_steps=False)
        
        logger1 = SimulationLogger("logger1", config1)
        logger2 = SimulationLogger("logger2", config2)
        
        assert logger1.config != logger2.config
        assert logger1.logger.name != logger2.logger.name
        assert logger1.logger.level != logger2.logger.level
    
    def test_log_level_filtering(self):
        """Test that log level filtering works correctly."""
        # Test that different log levels create loggers with different effective levels
        info_config = LoggingConfig(level="INFO")
        warning_config = LoggingConfig(level="WARNING")
        
        info_logger = SimulationLogger("info_test", info_config)
        warning_logger = SimulationLogger("warning_test", warning_config)
        
        assert info_logger.logger.level == logging.INFO
        assert warning_logger.logger.level == logging.WARNING
        assert warning_logger.logger.level > info_logger.logger.level


class TestCloneIdValidation:
    """Test cases for clone ID generation and validation patterns."""
    
    def test_clone_id_hierarchical_patterns(self):
        """Test that clone IDs follow proper hierarchical patterns."""
        logger = SimulationLogger("clone_id_test")
        
        # Test initial clone IDs (single digits)
        initial_ids = ["1", "2", "3"]
        for clone_id in initial_ids:
            assert len(clone_id) == 1
            assert clone_id.isdigit()
        
        # Test first-level sub-clone IDs (digit + letter)
        first_level_ids = ["1L", "1S", "1R", "2L", "2S", "2R", "3L", "3S", "3R"]
        for clone_id in first_level_ids:
            assert len(clone_id) == 2
            assert clone_id[0].isdigit()
            assert clone_id[1] in ["L", "S", "R"]
        
        # Test second-level sub-clone IDs (digit + letter + letter)
        second_level_ids = ["1LL", "1LS", "1LR", "2SL", "2SS", "2SR", "3RL", "3RS", "3RR"]
        for clone_id in second_level_ids:
            assert len(clone_id) == 3
            assert clone_id[0].isdigit()
            assert clone_id[1] in ["L", "S", "R"]
            assert clone_id[2] in ["L", "S", "R"]
    
    def test_clone_id_consistency_in_logs(self):
        """Test that clone IDs are consistently formatted in log messages."""
        logger = SimulationLogger("consistency_test")
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        # Test various clone IDs and verify they appear correctly in logs
        test_cases = [
            ("1", "EMPTY", 0, 5),
            ("2L", "FOOD", 10, 15),
            ("3RS", "WALL", -10, 5),
            ("1LLR", "SNAKE", -10, 10)
        ]
        
        for clone_id, result, reward, score in test_cases:
            log_capture.seek(0)
            log_capture.truncate(0)
            
            logger.log_clone_step(clone_id, result, reward, score)
            log_output = log_capture.getvalue().strip()
            
            # Verify clone ID appears at the start of the log message
            assert log_output.startswith(clone_id)
            # Verify the complete expected format
            expected = f"{clone_id} RESULT:{result} REWARD:{reward} SCORE:{score}"
            assert log_output == expected


class TestLoggingCompleteness:
    """Test cases for logging completeness and format consistency."""
    
    def test_all_result_types_logged(self):
        """Test that all possible result types are properly logged."""
        logger = SimulationLogger("result_test")
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        # Test all possible result types from the design document
        result_types = ["EMPTY", "FOOD", "WALL", "SNAKE"]
        
        for i, result_type in enumerate(result_types):
            log_capture.seek(0)
            log_capture.truncate(0)
            
            clone_id = f"{i+1}"
            reward = 10 if result_type == "FOOD" else (-10 if result_type in ["WALL", "SNAKE"] else 0)
            score = 5 + (i * 5)
            
            logger.log_clone_step(clone_id, result_type, reward, score)
            log_output = log_capture.getvalue().strip()
            
            expected = f"{clone_id} RESULT:{result_type} REWARD:{reward} SCORE:{score}"
            assert log_output == expected
    
    def test_budget_tracking_completeness(self):
        """Test that budget tracking logs contain all required information."""
        logger = SimulationLogger("budget_test")
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.DEBUG)  # Budget logs are at DEBUG level
        
        # Test budget status logging
        logger.log_budget_status(75, 100)
        log_output = log_capture.getvalue().strip()
        
        assert "BUDGET:" in log_output
        assert "75/100" in log_output
        assert "remaining" in log_output
    
    def test_tree_metrics_completeness(self):
        """Test that tree metrics logging includes all required metrics."""
        logger = SimulationLogger("metrics_test")
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        # Test comprehensive tree metrics as specified in requirements 8.1
        metrics = {
            "total_clones_created": 15,
            "max_depth_reached": 4,
            "budget_consumed": 85,
            "paths_evaluated": 12,
            "clone_terminations": 8,
            "cumulative_rewards": 45
        }
        
        logger.log_tree_metrics(metrics)
        log_output = log_capture.getvalue().strip()
        
        assert log_output.startswith("TREE_METRICS:")
        for key, value in metrics.items():
            assert f"{key}:{value}" in log_output
    
    def test_neural_network_logging_completeness(self):
        """Test that neural network logging includes all required information."""
        logger = SimulationLogger("nn_test")
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        # Test NN prediction logging
        log_capture.seek(0)
        log_capture.truncate(0)
        logger.log_neural_network_prediction("LEFT", 0.85, 30)
        log_output = log_capture.getvalue().strip()
        
        assert "NN PREDICTION:" in log_output
        assert "LEFT" in log_output
        assert "CONFIDENCE:0.850" in log_output
        assert "SCORE:30" in log_output
        
        # Test oracle decision logging
        log_capture.seek(0)
        log_capture.truncate(0)
        logger.log_oracle_decision("LEFT", "STRAIGHT", "STRAIGHT", 35)
        log_output = log_capture.getvalue().strip()
        
        assert "ORACLE:" in log_output
        assert "NN=LEFT" in log_output
        assert "OPTIMAL=STRAIGHT" in log_output
        assert "DECISION=STRAIGHT" in log_output
        assert "SCORE:35" in log_output
    
    def test_decision_cycle_logging_completeness(self):
        """Test that decision cycle logging includes all required tracking information."""
        logger = SimulationLogger("cycle_test")
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        # Test decision cycle logging as specified in requirements 8.1 and 8.4
        logger.log_decision_cycle(5, "2S", 8, 45)
        log_output = log_capture.getvalue().strip()
        
        # Verify all required components are present
        assert "CYCLE:5" in log_output
        assert "WINNER:2S" in log_output
        assert "PATHS:8" in log_output
        assert "BUDGET:45" in log_output
        
        # Verify the complete format matches specification
        expected = "CYCLE:5 WINNER:2S PATHS:8 BUDGET:45"
        assert log_output == expected


class TestLoggingErrorHandling:
    """Test cases for logging system error handling and edge cases."""
    
    def test_logging_with_invalid_clone_ids(self):
        """Test logging behavior with edge case clone IDs."""
        logger = SimulationLogger("edge_case_test")
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        # Test with various edge case clone IDs
        edge_case_ids = ["0", "10", "A", "1X", "999"]
        
        for clone_id in edge_case_ids:
            log_capture.seek(0)
            log_capture.truncate(0)
            
            # Should not raise an exception
            logger.log_clone_step(clone_id, "EMPTY", 0, 5)
            log_output = log_capture.getvalue().strip()
            
            # Should still log the message with the provided ID
            expected = f"{clone_id} RESULT:EMPTY REWARD:0 SCORE:5"
            assert log_output == expected
        
        # Test empty clone ID separately (special case)
        log_capture.seek(0)
        log_capture.truncate(0)
        logger.log_clone_step("", "EMPTY", 0, 5)
        log_output = log_capture.getvalue()  # Don't strip to preserve leading space
        # Empty clone ID will result in a leading space, which is expected behavior
        expected = " RESULT:EMPTY REWARD:0 SCORE:5\n"
        assert log_output == expected
    
    def test_logging_with_extreme_values(self):
        """Test logging with extreme reward and score values."""
        logger = SimulationLogger("extreme_test")
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        
        # Test with extreme values
        extreme_cases = [
            ("1", "FOOD", 1000, 9999),
            ("2", "WALL", -1000, 0),
            ("3", "EMPTY", 0, -50)  # Negative score edge case
        ]
        
        for clone_id, result, reward, score in extreme_cases:
            log_capture.seek(0)
            log_capture.truncate(0)
            
            logger.log_clone_step(clone_id, result, reward, score)
            log_output = log_capture.getvalue().strip()
            
            expected = f"{clone_id} RESULT:{result} REWARD:{reward} SCORE:{score}"
            assert log_output == expected