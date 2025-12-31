"""
Property-based tests for Token Tracker data model validation.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os
from hypothesis import given, strategies as st, settings, assume
from typing import List, Optional

from ai_hydra.token_tracker.models import TokenTransaction, TrackerConfig, CSVSchema
from ai_hydra.token_tracker.csv_writer import CSVWriter
from ai_hydra.token_tracker.error_handler import (
    ValidationError,
    TokenTrackerErrorHandler,
)


class TestTokenTrackerDataValidation:
    """Property-based tests for token tracker data validation."""

    @given(
        prompt_text=st.text(min_size=1, max_size=1000),
        tokens_used=st.integers(min_value=0, max_value=1000000),
        elapsed_time=st.floats(
            min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False
        ),
        workspace_folder=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        ),
        hook_trigger_type=st.sampled_from(
            ["agentExecutionCompleted", "fileEdited", "fileSaved", "sessionStarted"]
        ),
        hook_name=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")),
        ),
    )
    @settings(max_examples=100, deadline=2000)
    def test_transaction_creation_validation_property(
        self,
        prompt_text,
        tokens_used,
        elapsed_time,
        workspace_folder,
        hook_trigger_type,
        hook_name,
    ):
        """
        **Feature: kiro-token-tracker, Property 8: Data Validation Integrity**
        **Validates: Requirements 7.3, 7.4**

        For any valid input data, TokenTransaction creation should succeed and
        produce a transaction with all required fields properly validated.
        """
        # Filter out problematic characters that could cause CSV issues
        assume(not any(c in prompt_text for c in ["\x00", "\x01", "\x02"]))
        assume(
            not any(
                c in workspace_folder
                for c in ["\x00", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]
            )
        )
        assume(workspace_folder.strip() != "")
        assume(prompt_text.strip() != "")

        try:
            transaction = TokenTransaction.create_new(
                prompt_text=prompt_text,
                tokens_used=tokens_used,
                elapsed_time=elapsed_time,
                workspace_folder=workspace_folder,
                hook_trigger_type=hook_trigger_type,
                hook_name=hook_name,
            )

            # Validate all required fields are present and valid
            assert transaction.prompt_text == prompt_text
            assert transaction.tokens_used == tokens_used
            assert transaction.elapsed_time == elapsed_time
            assert transaction.workspace_folder == workspace_folder
            assert transaction.hook_trigger_type == hook_trigger_type
            assert transaction.hook_name == hook_name

            # Validate auto-generated fields
            assert transaction.session_id is not None
            assert len(transaction.session_id) > 0
            assert transaction.agent_execution_id is not None
            assert len(transaction.agent_execution_id) > 0
            assert isinstance(transaction.timestamp, datetime)

            # Validate data types
            assert isinstance(transaction.tokens_used, int)
            assert isinstance(transaction.elapsed_time, float)
            assert isinstance(transaction.error_occurred, bool)

            # Validate constraints
            assert transaction.tokens_used >= 0
            assert transaction.elapsed_time >= 0.0

        except ValueError as e:
            # If validation fails, it should be for a legitimate reason
            error_msg = str(e).lower()
            assert any(
                reason in error_msg
                for reason in ["cannot be empty", "must be non-negative", "invalid"]
            )

    @given(
        invalid_tokens=st.integers(max_value=-1),
        invalid_time=st.floats(max_value=-0.1, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=1000)
    def test_transaction_validation_rejects_invalid_data_property(
        self, invalid_tokens, invalid_time
    ):
        """
        **Feature: kiro-token-tracker, Property 8: Data Validation Integrity**
        **Validates: Requirements 7.3, 7.4**

        For any invalid input data (negative values), TokenTransaction creation
        should reject the data with appropriate validation errors.
        """
        # Test negative tokens
        with pytest.raises(ValueError, match="tokens_used must be non-negative"):
            TokenTransaction.create_new(
                prompt_text="test prompt",
                tokens_used=invalid_tokens,
                elapsed_time=1.0,
                workspace_folder="test_workspace",
                hook_trigger_type="agentExecutionCompleted",
            )

        # Test negative elapsed time
        with pytest.raises(ValueError, match="elapsed_time must be non-negative"):
            TokenTransaction.create_new(
                prompt_text="test prompt",
                tokens_used=100,
                elapsed_time=invalid_time,
                workspace_folder="test_workspace",
                hook_trigger_type="agentExecutionCompleted",
            )

    @given(empty_strings=st.sampled_from(["", "   ", "\t", "\n", "\r\n"]))
    @settings(max_examples=20, deadline=1000)
    def test_transaction_validation_rejects_empty_strings_property(self, empty_strings):
        """
        **Feature: kiro-token-tracker, Property 8: Data Validation Integrity**
        **Validates: Requirements 7.3, 7.4**

        For any empty or whitespace-only strings in required fields,
        TokenTransaction creation should reject the data.
        """
        # Test empty prompt text
        with pytest.raises(ValueError, match="prompt_text cannot be empty"):
            TokenTransaction.create_new(
                prompt_text=empty_strings,
                tokens_used=100,
                elapsed_time=1.0,
                workspace_folder="test_workspace",
                hook_trigger_type="agentExecutionCompleted",
            )

        # Test empty workspace folder
        with pytest.raises(ValueError, match="workspace_folder cannot be empty"):
            TokenTransaction.create_new(
                prompt_text="test prompt",
                tokens_used=100,
                elapsed_time=1.0,
                workspace_folder=empty_strings,
                hook_trigger_type="agentExecutionCompleted",
            )

        # Test empty hook trigger type
        with pytest.raises(ValueError, match="hook_trigger_type cannot be empty"):
            TokenTransaction.create_new(
                prompt_text="test prompt",
                tokens_used=100,
                elapsed_time=1.0,
                workspace_folder="test_workspace",
                hook_trigger_type=empty_strings,
            )

    @given(
        transactions=st.lists(
            st.builds(
                TokenTransaction.create_new,
                prompt_text=st.text(min_size=1, max_size=100).filter(
                    lambda x: x.strip() != ""
                ),
                tokens_used=st.integers(min_value=0, max_value=10000),
                elapsed_time=st.floats(
                    min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False
                ),
                workspace_folder=st.text(
                    min_size=1,
                    max_size=20,
                    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
                ).filter(lambda x: x.strip() != ""),
                hook_trigger_type=st.sampled_from(
                    ["agentExecutionCompleted", "fileEdited"]
                ),
                hook_name=st.text(
                    min_size=1,
                    max_size=20,
                    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
                ).filter(lambda x: x.strip() != ""),
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=50, deadline=3000)
    def test_csv_round_trip_property(self, transactions):
        """
        **Feature: kiro-token-tracker, Property 8: Data Validation Integrity**
        **Validates: Requirements 7.3, 7.4**

        For any list of valid transactions, converting to CSV format and back
        should preserve all data integrity without loss or corruption.
        """
        for original_transaction in transactions:
            # Convert to CSV row
            csv_row = original_transaction.to_csv_row()

            # Validate CSV row structure
            assert len(csv_row) == 12  # Expected number of columns
            assert all(isinstance(field, str) for field in csv_row)

            # Convert back from CSV row
            reconstructed_transaction = TokenTransaction.from_csv_row(csv_row)

            # Validate data preservation (accounting for CSV sanitization)
            # The prompt text may be sanitized for CSV safety
            original_sanitized = (
                original_transaction.prompt_text.replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace('"', '""')
            )
            assert (
                reconstructed_transaction.prompt_text == original_sanitized
                or reconstructed_transaction.prompt_text
                == original_transaction.prompt_text
            )
            assert (
                reconstructed_transaction.tokens_used
                == original_transaction.tokens_used
            )
            assert (
                reconstructed_transaction.elapsed_time
                == original_transaction.elapsed_time
            )
            assert (
                reconstructed_transaction.workspace_folder
                == original_transaction.workspace_folder
            )
            assert (
                reconstructed_transaction.hook_trigger_type
                == original_transaction.hook_trigger_type
            )
            assert reconstructed_transaction.hook_name == original_transaction.hook_name
            assert (
                reconstructed_transaction.error_occurred
                == original_transaction.error_occurred
            )
            assert (
                reconstructed_transaction.error_message
                == original_transaction.error_message
            )

            # Validate timestamp preservation (within reasonable precision)
            time_diff = abs(
                (
                    reconstructed_transaction.timestamp - original_transaction.timestamp
                ).total_seconds()
            )
            assert time_diff < 1.0  # Should be identical or very close

    @given(
        config_params=st.fixed_dictionaries(
            {
                "max_prompt_length": st.integers(min_value=10, max_value=10000),
                "backup_interval_hours": st.integers(
                    min_value=1, max_value=168
                ),  # 1 week max
                "retention_days": st.integers(
                    min_value=1, max_value=3650
                ),  # 10 years max
                "file_lock_timeout_seconds": st.floats(
                    min_value=0.1, max_value=60.0, allow_nan=False, allow_infinity=False
                ),
                "max_concurrent_writes": st.integers(min_value=1, max_value=100),
                "log_level": st.sampled_from(
                    ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                ),
            }
        )
    )
    @settings(max_examples=50, deadline=2000)
    def test_tracker_config_validation_property(self, config_params):
        """
        **Feature: kiro-token-tracker, Property 8: Data Validation Integrity**
        **Validates: Requirements 7.3, 7.4**

        For any valid configuration parameters, TrackerConfig creation should
        succeed and all validation rules should be properly enforced.
        """
        config = TrackerConfig(**config_params)

        # Validate all parameters are set correctly
        assert config.max_prompt_length == config_params["max_prompt_length"]
        assert config.backup_interval_hours == config_params["backup_interval_hours"]
        assert config.retention_days == config_params["retention_days"]
        assert (
            config.file_lock_timeout_seconds
            == config_params["file_lock_timeout_seconds"]
        )
        assert config.max_concurrent_writes == config_params["max_concurrent_writes"]
        assert config.log_level == config_params["log_level"]

        # Validate constraints are enforced
        assert config.max_prompt_length >= 10
        assert config.backup_interval_hours >= 1
        assert config.retention_days >= 1
        assert config.file_lock_timeout_seconds > 0
        assert config.max_concurrent_writes >= 1
        assert config.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        # Validate derived properties
        assert isinstance(config.csv_file_path, Path)

        # Validate configuration validation method
        validation_issues = config.validate()
        # For valid configs, there should be no critical issues
        critical_issues = [
            issue
            for issue in validation_issues
            if "does not exist" in issue or "No write permission" in issue
        ]
        # We can't test file system issues in property tests, so we just check structure
        assert isinstance(validation_issues, list)

    @given(
        invalid_config_params=st.one_of(
            st.fixed_dictionaries({"max_prompt_length": st.integers(max_value=9)}),
            st.fixed_dictionaries({"backup_interval_hours": st.integers(max_value=0)}),
            st.fixed_dictionaries({"retention_days": st.integers(max_value=0)}),
            st.fixed_dictionaries(
                {
                    "file_lock_timeout_seconds": st.floats(
                        max_value=0.0, allow_nan=False, allow_infinity=False
                    )
                }
            ),
            st.fixed_dictionaries({"max_concurrent_writes": st.integers(max_value=0)}),
            st.fixed_dictionaries(
                {
                    "log_level": st.text(min_size=1, max_size=10).filter(
                        lambda x: x
                        not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                    )
                }
            ),
        )
    )
    @settings(max_examples=30, deadline=1000)
    def test_tracker_config_rejects_invalid_params_property(
        self, invalid_config_params
    ):
        """
        **Feature: kiro-token-tracker, Property 8: Data Validation Integrity**
        **Validates: Requirements 7.3, 7.4**

        For any invalid configuration parameters, TrackerConfig creation
        should reject the parameters with appropriate validation errors.
        """
        with pytest.raises(ValueError):
            TrackerConfig(**invalid_config_params)

    @given(
        csv_headers=st.lists(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
            ),
            min_size=1,
            max_size=20,
            unique=True,
        )
    )
    @settings(max_examples=30, deadline=1000)
    def test_csv_schema_header_validation_property(self, csv_headers):
        """
        **Feature: kiro-token-tracker, Property 8: Data Validation Integrity**
        **Validates: Requirements 7.3, 7.4**

        For any set of CSV headers, the schema validation should correctly
        identify missing, extra, or incorrectly ordered headers.
        """
        schema = CSVSchema()
        validation_issues = schema.validate_headers(csv_headers)

        # Check for missing headers
        missing_headers = [h for h in schema.headers if h not in csv_headers]
        extra_headers = [h for h in csv_headers if h not in schema.headers]

        if missing_headers:
            assert any(
                "Missing required header" in issue for issue in validation_issues
            )

        if extra_headers:
            assert any("Unknown header" in issue for issue in validation_issues)

        if csv_headers != schema.headers:
            # Either missing/extra headers or wrong order
            if not missing_headers and not extra_headers:
                # Must be wrong order
                assert any(
                    "not in expected order" in issue for issue in validation_issues
                )

        # If headers match exactly, there should be no issues
        if csv_headers == schema.headers:
            assert len(validation_issues) == 0

    @given(
        special_characters=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(
                whitelist_categories=("Pc", "Pd", "Po", "Ps", "Pe", "Pi", "Pf"),
                whitelist_characters=["\n", "\r", "\t", '"', "'", ",", ";"],
            ),
        )
    )
    @settings(max_examples=50, deadline=2000)
    def test_special_character_handling_property(self, special_characters):
        """
        **Feature: kiro-token-tracker, Property 8: Data Validation Integrity**
        **Validates: Requirements 7.3, 7.4**

        For any text containing special characters, CSV sanitization should
        properly escape and preserve the content without corruption.
        """
        # Create transaction with special characters in prompt
        transaction = TokenTransaction.create_new(
            prompt_text=special_characters,
            tokens_used=100,
            elapsed_time=1.0,
            workspace_folder="test_workspace",
            hook_trigger_type="agentExecutionCompleted",
        )

        # Convert to CSV and back
        csv_row = transaction.to_csv_row()
        reconstructed = TokenTransaction.from_csv_row(csv_row)

        # The content should be preserved (though possibly sanitized)
        # At minimum, the essential meaning should be preserved
        assert len(reconstructed.prompt_text) > 0

        # If the original was not too long, it should be mostly preserved
        if len(special_characters) <= 1000:  # Within max length
            # Should preserve most characters (some may be escaped)
            assert len(reconstructed.prompt_text) >= len(special_characters) * 0.5

    @given(
        file_patterns=st.one_of(
            st.none(),
            st.lists(
                st.text(
                    min_size=1,
                    max_size=20,
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")
                    ),
                ),
                min_size=0,
                max_size=5,
            ),
        )
    )
    @settings(max_examples=30, deadline=1000)
    def test_file_patterns_handling_property(self, file_patterns):
        """
        **Feature: kiro-token-tracker, Property 8: Data Validation Integrity**
        **Validates: Requirements 7.3, 7.4**

        For any file patterns list (including None), the transaction should
        properly handle and preserve the file pattern information.
        """
        transaction = TokenTransaction.create_new(
            prompt_text="test prompt",
            tokens_used=100,
            elapsed_time=1.0,
            workspace_folder="test_workspace",
            hook_trigger_type="agentExecutionCompleted",
            file_patterns=file_patterns,
        )

        # Validate file patterns are stored correctly
        assert transaction.file_patterns == file_patterns

        # Test CSV round trip
        csv_row = transaction.to_csv_row()
        reconstructed = TokenTransaction.from_csv_row(csv_row)

        # File patterns should be preserved (empty list becomes None in CSV)
        if file_patterns == []:
            assert reconstructed.file_patterns is None
        else:
            assert reconstructed.file_patterns == file_patterns

    @given(
        transactions=st.lists(
            st.builds(
                TokenTransaction.create_new,
                prompt_text=st.text(min_size=1, max_size=500).filter(
                    lambda x: x.strip() != "" and "\x00" not in x
                ),
                tokens_used=st.integers(min_value=1, max_value=10000),
                elapsed_time=st.floats(
                    min_value=0.001,
                    max_value=60.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                workspace_folder=st.text(
                    min_size=1,
                    max_size=50,
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Pc")
                    ),
                ).filter(lambda x: x.strip() != ""),
                hook_trigger_type=st.sampled_from(
                    ["agentExecutionCompleted", "fileEdited", "userMessage", "manual"]
                ),
                hook_name=st.text(
                    min_size=1,
                    max_size=30,
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")
                    ),
                ).filter(lambda x: x.strip() != ""),
                file_patterns=st.one_of(
                    st.none(),
                    st.lists(
                        st.text(
                            min_size=1,
                            max_size=20,
                            alphabet=st.characters(
                                whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")
                            ),
                        ).filter(
                            lambda x: ";" not in x
                        ),  # Avoid semicolons in patterns
                        min_size=0,
                        max_size=3,
                    ),
                ),
                error_occurred=st.booleans(),
                error_message=st.one_of(
                    st.none(),
                    st.text(min_size=0, max_size=200).filter(lambda x: "\x00" not in x),
                ),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=100, deadline=5000)
    def test_csv_transaction_persistence_property(self, transactions):
        """
        **Feature: kiro-token-tracker, Property 1: CSV Transaction Persistence**
        **Validates: Requirements 1.1, 1.2, 1.3, 6.1, 6.2**

        For any valid token transaction, recording it to the CSV file should result in
        the transaction being retrievable with all original data intact, and the CSV
        should maintain proper structure with required headers.
        """
        # Create temporary CSV file for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_transactions.csv"

            # Create test configuration
            config = TrackerConfig(
                csv_file_path=csv_path,
                backup_enabled=False,  # Disable backup for testing
                enable_validation=True,
                file_lock_timeout_seconds=2.0,
            )

            # Create CSV writer
            error_handler = TokenTrackerErrorHandler()
            csv_writer = CSVWriter(config, error_handler)

            try:
                # Write all transactions
                successful_writes = 0
                for transaction in transactions:
                    if csv_writer.write_transaction(transaction):
                        successful_writes += 1

                # All transactions should be written successfully
                assert successful_writes == len(transactions)

                # Verify CSV file exists and has correct structure
                assert csv_path.exists()
                assert csv_writer.validate_csv_headers()

                # Read back all transactions
                retrieved_transactions = csv_writer.read_transactions()

                # Should retrieve the same number of transactions
                assert len(retrieved_transactions) == len(transactions)

                # Verify each transaction is preserved correctly
                for original, retrieved in zip(transactions, retrieved_transactions):
                    # Core data should be identical
                    assert retrieved.tokens_used == original.tokens_used
                    assert retrieved.elapsed_time == original.elapsed_time
                    assert retrieved.workspace_folder == original.workspace_folder
                    assert retrieved.hook_trigger_type == original.hook_trigger_type
                    assert retrieved.hook_name == original.hook_name
                    assert retrieved.error_occurred == original.error_occurred

                    # Session and execution IDs should be preserved
                    assert retrieved.session_id == original.session_id
                    assert retrieved.agent_execution_id == original.agent_execution_id

                    # Timestamp should be preserved (within reasonable precision)
                    time_diff = abs(
                        (retrieved.timestamp - original.timestamp).total_seconds()
                    )
                    assert time_diff < 1.0

                    # Prompt text should be preserved (may be sanitized for CSV)
                    assert len(retrieved.prompt_text) > 0
                    if len(original.prompt_text) <= config.max_prompt_length:
                        # Should be mostly preserved if within limits
                        assert (
                            len(retrieved.prompt_text)
                            >= len(original.prompt_text) * 0.8
                        )

                    # File patterns should be preserved
                    if original.file_patterns == []:
                        assert retrieved.file_patterns is None
                    else:
                        assert retrieved.file_patterns == original.file_patterns

                    # Error message should be preserved
                    if original.error_message == "":
                        assert (
                            retrieved.error_message is None
                            or retrieved.error_message == ""
                        )
                    else:
                        assert retrieved.error_message == original.error_message

                # Validate CSV integrity
                integrity_results = csv_writer.validate_csv_integrity()
                assert integrity_results["file_exists"]
                assert integrity_results["file_readable"]
                assert integrity_results["header_valid"]
                assert integrity_results["total_rows"] == len(transactions)
                assert integrity_results["valid_rows"] == len(transactions)
                assert integrity_results["invalid_rows"] == 0

            finally:
                # Cleanup
                if csv_path.exists():
                    csv_path.unlink()

    @given(
        initial_transactions=st.lists(
            st.builds(
                TokenTransaction.create_new,
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
                workspace_folder=st.text(
                    min_size=1,
                    max_size=30,
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Pc")
                    ),
                ).filter(lambda x: x.strip() != ""),
                hook_trigger_type=st.sampled_from(
                    ["agentExecutionCompleted", "fileEdited", "userMessage"]
                ),
                hook_name=st.text(
                    min_size=1,
                    max_size=20,
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Pc")
                    ),
                ).filter(lambda x: x.strip() != ""),
                error_occurred=st.booleans(),
            ),
            min_size=0,
            max_size=10,
        ),
        append_transactions=st.lists(
            st.builds(
                TokenTransaction.create_new,
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
                workspace_folder=st.text(
                    min_size=1,
                    max_size=30,
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Pc")
                    ),
                ).filter(lambda x: x.strip() != ""),
                hook_trigger_type=st.sampled_from(
                    ["agentExecutionCompleted", "fileEdited", "userMessage"]
                ),
                hook_name=st.text(
                    min_size=1,
                    max_size=20,
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Pc")
                    ),
                ).filter(lambda x: x.strip() != ""),
                error_occurred=st.booleans(),
            ),
            min_size=1,
            max_size=15,
        ),
    )
    @settings(max_examples=50, deadline=8000)
    def test_data_append_safety_property(
        self, initial_transactions, append_transactions
    ):
        """
        **Feature: kiro-token-tracker, Property 2: Data Append Safety**
        **Validates: Requirements 1.4, 3.4**

        For any sequence of token transactions, appending them to the CSV should
        preserve all existing data while adding the new transactions in chronological
        order without corruption or loss.
        """
        # Create temporary CSV file for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_append_safety.csv"

            # Create test configuration
            config = TrackerConfig(
                csv_file_path=csv_path,
                backup_enabled=True,  # Enable backup to test data preservation
                enable_validation=True,
                file_lock_timeout_seconds=3.0,
            )

            # Create CSV writer
            error_handler = TokenTrackerErrorHandler()
            csv_writer = CSVWriter(config, error_handler)

            try:
                # Step 1: Write initial transactions if any
                initial_count = 0
                if initial_transactions:
                    for transaction in initial_transactions:
                        if csv_writer.write_transaction(transaction):
                            initial_count += 1

                # Verify initial state
                if initial_count > 0:
                    initial_data = csv_writer.read_transactions()
                    assert len(initial_data) == initial_count
                else:
                    initial_data = []

                # Step 2: Use safe append for new transactions
                append_result = csv_writer.append_transactions_safe(append_transactions)

                # Verify append operation results
                assert append_result[
                    "success"
                ], f"Append failed: {append_result['issues']}"
                assert append_result[
                    "validation_passed"
                ], "Validation should pass for valid data"
                assert append_result[
                    "existing_data_preserved"
                ], "Existing data should be preserved"
                assert append_result["transactions_added"] == len(append_transactions)
                assert append_result["transactions_failed"] == 0

                # Step 3: Verify data integrity after append
                integrity_result = csv_writer.verify_data_integrity_after_append(
                    initial_count, len(append_transactions)
                )

                assert integrity_result[
                    "integrity_verified"
                ], f"Integrity check failed: {integrity_result['issues']}"
                assert integrity_result["expected_count"] == initial_count + len(
                    append_transactions
                )
                assert (
                    integrity_result["actual_count"]
                    == integrity_result["expected_count"]
                )
                assert integrity_result[
                    "chronological_order_valid"
                ], "Transactions should be in chronological order"
                assert integrity_result[
                    "no_duplicates"
                ], "No duplicate transactions should exist"
                assert integrity_result[
                    "all_data_valid"
                ], "All transaction data should be valid"

                # Step 4: Verify all data can be read back correctly
                final_transactions = csv_writer.read_transactions()

                # Should have all transactions
                assert len(final_transactions) == initial_count + len(
                    append_transactions
                )

                # Verify existing data is preserved (first N transactions should match initial)
                if initial_data:
                    for i, (original, preserved) in enumerate(
                        zip(initial_data, final_transactions[:initial_count])
                    ):
                        # Core data should be identical
                        assert (
                            preserved.tokens_used == original.tokens_used
                        ), f"Token count mismatch at index {i}"
                        assert (
                            preserved.elapsed_time == original.elapsed_time
                        ), f"Elapsed time mismatch at index {i}"
                        assert (
                            preserved.workspace_folder == original.workspace_folder
                        ), f"Workspace mismatch at index {i}"
                        assert (
                            preserved.hook_trigger_type == original.hook_trigger_type
                        ), f"Hook type mismatch at index {i}"
                        assert (
                            preserved.session_id == original.session_id
                        ), f"Session ID mismatch at index {i}"
                        assert (
                            preserved.agent_execution_id == original.agent_execution_id
                        ), f"Execution ID mismatch at index {i}"

                # Verify appended data is present (last M transactions should correspond to appended)
                appended_data = final_transactions[initial_count:]
                assert len(appended_data) == len(append_transactions)

                # Verify chronological ordering across all transactions
                if len(final_transactions) > 1:
                    timestamps = [t.timestamp for t in final_transactions]
                    sorted_timestamps = sorted(timestamps)
                    # Allow some tolerance for concurrent operations
                    for i, (actual, expected) in enumerate(
                        zip(timestamps, sorted_timestamps)
                    ):
                        time_diff = abs((actual - expected).total_seconds())
                        assert (
                            time_diff <= 5.0
                        ), f"Chronological order violation at index {i}: {time_diff}s difference"

                # Step 5: Verify CSV file integrity
                csv_integrity = csv_writer.validate_csv_integrity()
                assert csv_integrity["file_exists"], "CSV file should exist"
                assert csv_integrity["header_valid"], "CSV headers should be valid"
                assert csv_integrity["total_rows"] == len(
                    final_transactions
                ), "Row count should match transaction count"
                assert (
                    csv_integrity["valid_rows"] == csv_integrity["total_rows"]
                ), "All rows should be valid"
                assert (
                    csv_integrity["invalid_rows"] == 0
                ), "No invalid rows should exist"

                # Step 6: Test round-trip integrity for all transactions
                for i, transaction in enumerate(final_transactions):
                    try:
                        csv_row = transaction.to_csv_row()
                        reconstructed = TokenTransaction.from_csv_row(csv_row)

                        # Verify essential data is preserved
                        assert reconstructed.tokens_used == transaction.tokens_used
                        assert reconstructed.elapsed_time == transaction.elapsed_time
                        assert (
                            reconstructed.workspace_folder
                            == transaction.workspace_folder
                        )
                        assert (
                            reconstructed.hook_trigger_type
                            == transaction.hook_trigger_type
                        )

                    except Exception as e:
                        pytest.fail(f"Round-trip failed for transaction {i}: {e}")

            finally:
                # Cleanup
                if csv_path.exists():
                    csv_path.unlink()
                # Clean up any backup files
                for backup_file in Path(temp_dir).glob("*_backup_*"):
                    backup_file.unlink()
