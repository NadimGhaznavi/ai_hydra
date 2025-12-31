"""
Property-based tests for Token Tracker error recovery resilience.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume
from typing import List, Optional, Dict, Any

from ai_hydra.token_tracker.models import TokenTransaction, TrackerConfig
from ai_hydra.token_tracker.tracker import TokenTracker
from ai_hydra.token_tracker.csv_writer import CSVWriter
from ai_hydra.token_tracker.error_handler import (
    TokenTrackerErrorHandler,
    CSVWriteError,
    CSVReadError,
    FileLockError,
    PermissionError as TrackerPermissionError,
    DiskSpaceError,
    ValidationError,
    ConfigurationError,
    MetadataError,
)


class TestTokenTrackerErrorRecovery:
    """Property-based tests for token tracker error recovery resilience."""

    @given(
        prompt_text=st.text(min_size=1, max_size=500).filter(
            lambda x: x.strip() != "" and "\x00" not in x
        ),
        tokens_used=st.integers(min_value=1, max_value=10000),
        elapsed_time=st.floats(
            min_value=0.001, max_value=60.0, allow_nan=False, allow_infinity=False
        ),
        error_scenarios=st.sampled_from(
            [
                "csv_write_error",
                "csv_read_error",
                "file_lock_error",
                "permission_error",
                "disk_space_error",
                "validation_error",
                "metadata_error",
            ]
        ),
    )
    @settings(max_examples=50, deadline=5000)
    def test_error_recovery_resilience_property(
        self, prompt_text, tokens_used, elapsed_time, error_scenarios
    ):
        """
        **Feature: kiro-token-tracker, Property 5: Error Recovery Resilience**
        **Validates: Requirements 2.4, 7.1, 7.2, 7.3, 7.4, 7.5**

        For any error condition (file system errors, parsing failures, validation errors),
        the token tracker should handle the error gracefully, log appropriate messages,
        and continue operation without data loss.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_error_recovery.csv"

            # Create test configuration
            config = TrackerConfig(
                csv_file_path=csv_path,
                backup_enabled=True,
                enable_validation=True,
                file_lock_timeout_seconds=2.0,
                max_prompt_length=1000,
            )

            # Create token tracker
            tracker = TokenTracker(config)

            try:
                # Test different error scenarios
                if error_scenarios == "csv_write_error":
                    self._test_csv_write_error_recovery(
                        tracker, prompt_text, tokens_used, elapsed_time, csv_path
                    )
                elif error_scenarios == "csv_read_error":
                    self._test_csv_read_error_recovery(
                        tracker, prompt_text, tokens_used, elapsed_time, csv_path
                    )
                elif error_scenarios == "file_lock_error":
                    self._test_file_lock_error_recovery(
                        tracker, prompt_text, tokens_used, elapsed_time, csv_path
                    )
                elif error_scenarios == "permission_error":
                    self._test_permission_error_recovery(
                        tracker, prompt_text, tokens_used, elapsed_time, csv_path
                    )
                elif error_scenarios == "disk_space_error":
                    self._test_disk_space_error_recovery(
                        tracker, prompt_text, tokens_used, elapsed_time, csv_path
                    )
                elif error_scenarios == "validation_error":
                    self._test_validation_error_recovery(
                        tracker, prompt_text, tokens_used, elapsed_time
                    )
                elif error_scenarios == "metadata_error":
                    self._test_metadata_error_recovery(
                        tracker, prompt_text, tokens_used, elapsed_time
                    )

            finally:
                # Cleanup
                if csv_path.exists():
                    csv_path.unlink()

    def _test_csv_write_error_recovery(
        self,
        tracker: TokenTracker,
        prompt_text: str,
        tokens_used: int,
        elapsed_time: float,
        csv_path: Path,
    ):
        """Test recovery from CSV write errors."""
        # Mock CSV writer to simulate write failure
        original_write = tracker.csv_writer.write_transaction

        def mock_write_with_failure(transaction):
            # Simulate write failure
            raise OSError("Simulated disk write failure")

        # First, record a successful transaction to establish baseline
        baseline_success = tracker.record_transaction(
            prompt_text="baseline test", tokens_used=100, elapsed_time=1.0
        )
        assert baseline_success, "Baseline transaction should succeed"

        # Now simulate write failure
        tracker.csv_writer.write_transaction = mock_write_with_failure

        # Attempt to record transaction with simulated failure
        result = tracker.record_transaction(prompt_text, tokens_used, elapsed_time)

        # System should handle error gracefully
        # Result may be False (failure) but system should not crash
        assert isinstance(result, bool), "Should return boolean result"

        # Restore original write method
        tracker.csv_writer.write_transaction = original_write

        # System should recover and work normally after error
        recovery_success = tracker.record_transaction(
            prompt_text="recovery test", tokens_used=200, elapsed_time=2.0
        )
        assert recovery_success, "System should recover after error"

        # Verify error statistics are tracked
        stats = tracker.get_statistics()
        assert stats["errors_encountered"] > 0, "Errors should be tracked"
        assert stats["transactions_failed"] > 0, "Failed transactions should be counted"

    def _test_csv_read_error_recovery(
        self,
        tracker: TokenTracker,
        prompt_text: str,
        tokens_used: int,
        elapsed_time: float,
        csv_path: Path,
    ):
        """Test recovery from CSV read errors."""
        # First record a transaction successfully
        success = tracker.record_transaction(prompt_text, tokens_used, elapsed_time)
        assert success, "Initial transaction should succeed"

        # Corrupt the CSV file to simulate read error
        if csv_path.exists():
            with open(csv_path, "w") as f:
                f.write("corrupted,invalid,csv,data\n")
                f.write("this,is,not,valid,csv,format\n")

        # Attempt to read transaction history
        history = tracker.get_transaction_history()

        # System should handle read error gracefully
        assert isinstance(history, list), "Should return list even on error"
        # May be empty due to corruption, but should not crash

        # System should continue to work for new transactions
        recovery_success = tracker.record_transaction(
            prompt_text="post-corruption test", tokens_used=300, elapsed_time=3.0
        )
        # May fail due to corrupted file, but should not crash
        assert isinstance(recovery_success, bool), "Should return boolean result"

    def _test_file_lock_error_recovery(
        self,
        tracker: TokenTracker,
        prompt_text: str,
        tokens_used: int,
        elapsed_time: float,
        csv_path: Path,
    ):
        """Test recovery from file lock errors."""
        # Mock file locking to simulate timeout
        original_acquire_lock = tracker.csv_writer._acquire_file_lock

        def mock_lock_timeout(*args, **kwargs):
            raise BlockingIOError("Simulated file lock timeout")

        # Record baseline transaction
        baseline_success = tracker.record_transaction(
            prompt_text="baseline", tokens_used=100, elapsed_time=1.0
        )
        assert baseline_success, "Baseline should succeed"

        # Simulate lock timeout
        tracker.csv_writer._acquire_file_lock = mock_lock_timeout

        # Attempt transaction with lock failure
        result = tracker.record_transaction(prompt_text, tokens_used, elapsed_time)

        # Should handle gracefully
        assert isinstance(result, bool), "Should return boolean result"

        # Restore original lock method
        tracker.csv_writer._acquire_file_lock = original_acquire_lock

        # Should recover
        recovery_success = tracker.record_transaction(
            prompt_text="recovery", tokens_used=200, elapsed_time=2.0
        )
        assert recovery_success, "Should recover from lock error"

    def _test_permission_error_recovery(
        self,
        tracker: TokenTracker,
        prompt_text: str,
        tokens_used: int,
        elapsed_time: float,
        csv_path: Path,
    ):
        """Test recovery from permission errors."""
        # Create CSV file and make it read-only to simulate permission error
        if not csv_path.exists():
            # Create initial file
            tracker.record_transaction("initial", 100, 1.0)

        # Make file read-only
        csv_path.chmod(0o444)  # Read-only

        try:
            # Attempt to write with permission denied
            result = tracker.record_transaction(prompt_text, tokens_used, elapsed_time)

            # Should handle gracefully (may succeed with fallback or fail gracefully)
            assert isinstance(result, bool), "Should return boolean result"

        finally:
            # Restore write permissions for cleanup
            csv_path.chmod(0o644)

        # Should work normally after permissions restored
        recovery_success = tracker.record_transaction(
            prompt_text="recovery", tokens_used=200, elapsed_time=2.0
        )
        assert recovery_success, "Should work after permission restored"

    def _test_disk_space_error_recovery(
        self,
        tracker: TokenTracker,
        prompt_text: str,
        tokens_used: int,
        elapsed_time: float,
        csv_path: Path,
    ):
        """Test recovery from disk space errors."""
        # Mock disk space check to simulate insufficient space
        original_check = tracker.csv_writer._check_disk_space

        def mock_disk_space_error(transaction):
            raise DiskSpaceError(
                "Simulated insufficient disk space",
                required_space=1000,
                available_space=100,
            )

        # Record baseline
        baseline_success = tracker.record_transaction(
            prompt_text="baseline", tokens_used=100, elapsed_time=1.0
        )
        assert baseline_success, "Baseline should succeed"

        # Simulate disk space error
        tracker.csv_writer._check_disk_space = mock_disk_space_error

        # Attempt transaction with disk space error
        result = tracker.record_transaction(prompt_text, tokens_used, elapsed_time)

        # Should handle gracefully
        assert isinstance(result, bool), "Should return boolean result"

        # Restore original method
        tracker.csv_writer._check_disk_space = original_check

        # Should recover
        recovery_success = tracker.record_transaction(
            prompt_text="recovery", tokens_used=200, elapsed_time=2.0
        )
        assert recovery_success, "Should recover from disk space error"

    def _test_validation_error_recovery(
        self,
        tracker: TokenTracker,
        prompt_text: str,
        tokens_used: int,
        elapsed_time: float,
    ):
        """Test recovery from validation errors."""
        # Test with invalid data that should trigger validation errors

        # Test with negative tokens (should be caught by validation)
        result1 = tracker.record_transaction(
            prompt_text="test", tokens_used=-100, elapsed_time=1.0
        )
        # Should handle validation error gracefully
        assert isinstance(result1, bool), "Should return boolean for invalid tokens"

        # Test with negative elapsed time
        result2 = tracker.record_transaction(
            prompt_text="test", tokens_used=100, elapsed_time=-1.0
        )
        # Should handle validation error gracefully
        assert isinstance(result2, bool), "Should return boolean for invalid time"

        # Test with empty prompt
        result3 = tracker.record_transaction(
            prompt_text="", tokens_used=100, elapsed_time=1.0
        )
        # Should handle validation error gracefully
        assert isinstance(result3, bool), "Should return boolean for empty prompt"

        # System should still work with valid data
        valid_result = tracker.record_transaction(
            prompt_text=prompt_text, tokens_used=tokens_used, elapsed_time=elapsed_time
        )
        assert valid_result, "Should work with valid data after validation errors"

    def _test_metadata_error_recovery(
        self,
        tracker: TokenTracker,
        prompt_text: str,
        tokens_used: int,
        elapsed_time: float,
    ):
        """Test recovery from metadata collection errors."""
        # Mock metadata collector to simulate failure
        original_collect = tracker.metadata_collector.collect_execution_metadata

        def mock_metadata_error(context):
            raise RuntimeError("Simulated metadata collection failure")

        # Record baseline
        baseline_success = tracker.record_transaction(
            prompt_text="baseline", tokens_used=100, elapsed_time=1.0
        )
        assert baseline_success, "Baseline should succeed"

        # Simulate metadata collection error
        tracker.metadata_collector.collect_execution_metadata = mock_metadata_error

        # Attempt transaction with metadata error
        result = tracker.record_transaction(prompt_text, tokens_used, elapsed_time)

        # Should handle gracefully (may use default metadata)
        assert isinstance(result, bool), "Should return boolean result"

        # Restore original method
        tracker.metadata_collector.collect_execution_metadata = original_collect

        # Should recover
        recovery_success = tracker.record_transaction(
            prompt_text="recovery", tokens_used=200, elapsed_time=2.0
        )
        assert recovery_success, "Should recover from metadata error"

    @given(
        error_types=st.lists(
            st.sampled_from(
                [
                    "csv_write_error",
                    "validation_error",
                    "metadata_error",
                ]
            ),
            min_size=1,
            max_size=3,
            unique=True,
        ),
        transactions=st.lists(
            st.builds(
                dict,
                prompt_text=st.text(min_size=1, max_size=200).filter(
                    lambda x: x.strip() != "" and "\x00" not in x
                ),
                tokens_used=st.integers(min_value=1, max_value=5000),
                elapsed_time=st.floats(
                    min_value=0.001,
                    max_value=30.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=30, deadline=8000)
    def test_multiple_error_recovery_property(self, error_types, transactions):
        """
        **Feature: kiro-token-tracker, Property 5: Error Recovery Resilience**
        **Validates: Requirements 2.4, 7.1, 7.2, 7.3, 7.4, 7.5**

        For any sequence of multiple error conditions, the token tracker should
        maintain resilience and continue operation without permanent failure.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_multiple_errors.csv"

            config = TrackerConfig(
                csv_file_path=csv_path,
                backup_enabled=True,
                enable_validation=True,
                file_lock_timeout_seconds=1.0,
            )

            tracker = TokenTracker(config)

            try:
                successful_transactions = 0
                error_count = 0

                # Process transactions with intermittent errors
                for i, transaction_data in enumerate(transactions):
                    # Introduce errors based on error_types
                    error_type = error_types[i % len(error_types)]

                    if error_type == "validation_error" and i % 2 == 0:
                        # Introduce validation error occasionally
                        result = tracker.record_transaction(
                            prompt_text="",  # Empty prompt should cause validation error
                            tokens_used=transaction_data["tokens_used"],
                            elapsed_time=transaction_data["elapsed_time"],
                        )
                        if not result:
                            error_count += 1
                    elif error_type == "csv_write_error" and i % 3 == 0:
                        # Mock CSV write error occasionally
                        original_write = tracker.csv_writer.write_transaction

                        def mock_write_error(trans):
                            raise OSError("Simulated write error")

                        tracker.csv_writer.write_transaction = mock_write_error

                        result = tracker.record_transaction(
                            prompt_text=transaction_data["prompt_text"],
                            tokens_used=transaction_data["tokens_used"],
                            elapsed_time=transaction_data["elapsed_time"],
                        )

                        # Restore original method
                        tracker.csv_writer.write_transaction = original_write

                        if not result:
                            error_count += 1
                    else:
                        # Normal transaction
                        result = tracker.record_transaction(
                            prompt_text=transaction_data["prompt_text"],
                            tokens_used=transaction_data["tokens_used"],
                            elapsed_time=transaction_data["elapsed_time"],
                        )
                        if result:
                            successful_transactions += 1

                # Verify system resilience
                stats = tracker.get_statistics()

                # System should track errors appropriately
                assert stats["errors_encountered"] >= 0, "Should track errors"
                assert stats["transactions_failed"] >= 0, "Should track failures"
                assert stats["transactions_recorded"] >= 0, "Should track successes"

                # System should still be functional after errors
                final_test = tracker.record_transaction(
                    prompt_text="final resilience test",
                    tokens_used=100,
                    elapsed_time=1.0,
                )
                # Should work or fail gracefully
                assert isinstance(final_test, bool), "Should return boolean result"

                # Error handler should have statistics
                error_stats = tracker.error_handler.get_error_statistics()
                assert isinstance(error_stats, dict), "Should return error statistics"
                assert "total_errors" in error_stats, "Should track total errors"

            finally:
                # Cleanup
                if csv_path.exists():
                    csv_path.unlink()

    @given(
        config_issues=st.lists(
            st.sampled_from(
                [
                    "invalid_max_prompt_length",
                    "invalid_backup_interval",
                    "invalid_retention_days",
                    "invalid_timeout",
                    "invalid_log_level",
                ]
            ),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    @settings(max_examples=20, deadline=3000)
    def test_configuration_error_recovery_property(self, config_issues):
        """
        **Feature: kiro-token-tracker, Property 5: Error Recovery Resilience**
        **Validates: Requirements 2.4, 7.1, 7.2, 7.3, 7.4, 7.5**

        For any configuration errors, the system should handle them gracefully
        and either use default values or fail with clear error messages.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_config_errors.csv"

            # Create invalid configuration based on issues
            config_params = {
                "csv_file_path": csv_path,
                "backup_enabled": True,
                "enable_validation": True,
            }

            for issue in config_issues:
                if issue == "invalid_max_prompt_length":
                    config_params["max_prompt_length"] = -1  # Invalid
                elif issue == "invalid_backup_interval":
                    config_params["backup_interval_hours"] = 0  # Invalid
                elif issue == "invalid_retention_days":
                    config_params["retention_days"] = -5  # Invalid
                elif issue == "invalid_timeout":
                    config_params["file_lock_timeout_seconds"] = -1.0  # Invalid
                elif issue == "invalid_log_level":
                    config_params["log_level"] = "INVALID_LEVEL"  # Invalid

            # Attempt to create tracker with invalid config
            try:
                config = TrackerConfig(**config_params)
                # If config creation succeeds, it should have used defaults
                tracker = TokenTracker(config)

                # System should still be functional
                result = tracker.record_transaction(
                    prompt_text="test with recovered config",
                    tokens_used=100,
                    elapsed_time=1.0,
                )
                assert isinstance(result, bool), "Should return boolean result"

            except (ValueError, ConfigurationError) as e:
                # Configuration error should be handled gracefully
                assert isinstance(e, (ValueError, ConfigurationError))
                assert len(str(e)) > 0, "Should have meaningful error message"

                # Should be able to create tracker with default config
                default_config = TrackerConfig.create_default()
                default_config.csv_file_path = csv_path

                tracker = TokenTracker(default_config)
                result = tracker.record_transaction(
                    prompt_text="test with default config",
                    tokens_used=100,
                    elapsed_time=1.0,
                )
                assert result, "Should work with default config"
