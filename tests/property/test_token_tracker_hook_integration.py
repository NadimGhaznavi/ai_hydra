"""
Property-based tests for Token Tracker hook integration.
"""

import pytest
import tempfile
import os
import uuid
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from ai_hydra.token_tracker.hook import TokenTrackingHook
from ai_hydra.token_tracker.tracker import TokenTracker
from ai_hydra.token_tracker.metadata_collector import MetadataCollector
from ai_hydra.token_tracker.models import TrackerConfig, TokenTransaction
from ai_hydra.token_tracker.error_handler import TokenTrackerErrorHandler


class TestTokenTrackingHookIntegration:
    """Property-based tests for hook-tracker integration."""

    @given(
        execution_context=st.fixed_dictionaries(
            {
                "execution_id": st.text(
                    min_size=10,
                    max_size=50,
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Pd")
                    ),
                ).filter(lambda x: x.strip() != ""),
                "trigger_type": st.sampled_from(
                    [
                        "agentExecutionCompleted",
                        "agentExecutionStarted",
                        "fileEdited",
                        "fileSaved",
                        "userMessage",
                        "manual",
                        "scheduled",
                    ]
                ),
                "workspace_folder": st.text(
                    min_size=1,
                    max_size=50,
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Pc")
                    ),
                ).filter(lambda x: x.strip() != "" and "/" not in x and "\\" not in x),
                "file_patterns": st.one_of(
                    st.none(),
                    st.lists(
                        st.sampled_from(
                            ["*.py", "*.js", "*.ts", "*.md", "*.txt", "*.json"]
                        ),
                        min_size=0,
                        max_size=5,
                    ),
                ),
                "hook_name": st.text(
                    min_size=1,
                    max_size=30,
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Pd")
                    ),
                ).filter(lambda x: x.strip() != ""),
                "session_id": st.text(
                    min_size=10,
                    max_size=50,
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Pd")
                    ),
                ).filter(lambda x: x.strip() != ""),
                "user_message": st.one_of(
                    st.none(),
                    st.text(min_size=1, max_size=200).filter(lambda x: x.strip() != ""),
                ),
            }
        ),
        execution_result=st.fixed_dictionaries(
            {
                "success": st.booleans(),
                "token_usage": st.one_of(
                    st.integers(min_value=1, max_value=10000),
                    st.fixed_dictionaries(
                        {
                            "total_tokens": st.integers(min_value=1, max_value=10000),
                            "prompt": st.text(min_size=1, max_size=1000).filter(
                                lambda x: x.strip() != "" and "\x00" not in x
                            ),
                            "model": st.sampled_from(
                                ["gpt-4", "gpt-3.5-turbo", "claude-3", "test_model"]
                            ),
                            "response": st.text(min_size=0, max_size=500),
                        }
                    ),
                ),
                "duration": st.floats(
                    min_value=0.1, max_value=60.0, allow_nan=False, allow_infinity=False
                ),
                "output": st.text(min_size=0, max_size=500),
                "error": st.one_of(st.none(), st.text(min_size=1, max_size=200)),
            }
        ),
    )
    @settings(max_examples=10, deadline=5000)  # Reduced for testing
    def test_hook_tracker_integration_property(
        self, execution_context, execution_result
    ):
        """
        **Feature: kiro-token-tracker, Property 4: Hook-Tracker Integration**
        **Validates: Requirements 2.1, 2.2, 2.3, 8.1, 8.2, 8.3, 8.4, 8.5**

        For any valid AI agent execution, the agent hook should automatically trigger
        token tracking and successfully record the transaction with complete metadata.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_hook_integration.csv"

            # Create test configuration
            config = TrackerConfig(
                csv_file_path=csv_path,
                backup_enabled=False,
                enable_validation=True,
                file_lock_timeout_seconds=2.0,
                log_level="DEBUG",
            )

            # Create token tracking hook
            hook = TokenTrackingHook(config)

            try:
                # Ensure hook is enabled
                assert hook.is_enabled, "Hook should be enabled by default"

                # Add execution_id to result if not present
                execution_result_with_id = execution_result.copy()
                execution_result_with_id["execution_id"] = execution_context[
                    "execution_id"
                ]

                # Simulate agent execution lifecycle
                hook.on_agent_execution_start(execution_context)

                # Verify execution context was stored
                assert (
                    execution_context["execution_id"] in hook.execution_context
                ), "Execution context should be stored on start"

                stored_context = hook.execution_context[
                    execution_context["execution_id"]
                ]
                assert "start_time" in stored_context, "Start time should be recorded"
                assert "trigger_type" in stored_context, "Trigger type should be stored"
                assert (
                    stored_context["trigger_type"] == execution_context["trigger_type"]
                ), "Trigger type should match context"

                # Simulate execution completion
                hook.on_agent_execution_complete(
                    execution_context, execution_result_with_id
                )

                # Verify execution context was cleaned up
                assert (
                    execution_context["execution_id"] not in hook.execution_context
                ), "Execution context should be cleaned up after completion"

                # Extract expected token usage
                token_usage = execution_result["token_usage"]
                expected_tokens = 0
                expected_prompt = ""

                if isinstance(token_usage, int):
                    expected_tokens = token_usage
                    expected_prompt = execution_result.get(
                        "output", "No prompt text available"
                    )
                elif isinstance(token_usage, dict):
                    expected_tokens = token_usage.get("total_tokens", 0)
                    expected_prompt = token_usage.get("prompt", "")

                # Only verify transaction if tokens were used
                if expected_tokens > 0:
                    # Verify transaction was recorded
                    transactions = hook.tracker.get_transaction_history()
                    assert (
                        len(transactions) >= 1
                    ), "At least one transaction should be recorded"

                    # Get the most recent transaction
                    transaction = transactions[-1]

                    # Verify core transaction data (Requirements 2.1, 2.2, 2.3)
                    assert (
                        transaction.tokens_used == expected_tokens
                    ), f"Token count should match: {transaction.tokens_used} vs {expected_tokens}"

                    assert (
                        transaction.prompt_text is not None
                    ), "Prompt text should be captured"
                    assert (
                        len(transaction.prompt_text) > 0
                    ), "Prompt text should not be empty"

                    assert (
                        transaction.elapsed_time > 0
                    ), "Elapsed time should be positive"
                    assert (
                        transaction.elapsed_time <= execution_result["duration"] + 1.0
                    ), "Elapsed time should be reasonable"

                    # Verify metadata integration (Requirements 8.1, 8.2, 8.3, 8.4, 8.5)

                    # Requirement 8.1: Workspace folder name
                    assert (
                        transaction.workspace_folder is not None
                    ), "Workspace folder should be captured"
                    assert (
                        len(transaction.workspace_folder) > 0
                    ), "Workspace folder should not be empty"

                    # Requirement 8.2: Hook trigger type
                    assert (
                        transaction.hook_trigger_type
                        == execution_context["trigger_type"]
                    ), f"Hook trigger type should match: {transaction.hook_trigger_type} vs {execution_context['trigger_type']}"

                    # Requirement 8.3: Agent execution ID
                    assert (
                        transaction.agent_execution_id
                        == execution_context["execution_id"]
                    ), f"Agent execution ID should match: {transaction.agent_execution_id} vs {execution_context['execution_id']}"

                    # Requirement 8.4: File patterns (when applicable)
                    if execution_context["file_patterns"]:
                        assert (
                            transaction.file_patterns
                            == execution_context["file_patterns"]
                        ), "File patterns should be preserved when provided"

                    # Requirement 8.5: Hook name
                    assert (
                        transaction.hook_name == execution_context["hook_name"]
                    ), f"Hook name should match: {transaction.hook_name} vs {execution_context['hook_name']}"

                    # Verify session ID consistency
                    assert (
                        transaction.session_id == execution_context["session_id"]
                    ), "Session ID should match context"

                    # Verify timestamp is recent and valid
                    time_diff = datetime.now() - transaction.timestamp
                    assert time_diff < timedelta(
                        seconds=30
                    ), f"Timestamp should be recent: {time_diff}"

                    # Verify error handling
                    if execution_result.get("error"):
                        # Transaction should still be recorded even if execution had errors
                        assert (
                            transaction.error_occurred
                            or not execution_result["success"]
                        ), "Error state should be reflected in transaction"

                    # Verify UUID format for generated IDs
                    try:
                        uuid.UUID(transaction.session_id)
                        uuid.UUID(transaction.agent_execution_id)
                    except ValueError:
                        pytest.fail(
                            "Session ID and Agent execution ID should be valid UUIDs"
                        )

                    # Verify CSV integrity
                    csv_integrity = hook.tracker.validate_csv_integrity()
                    assert csv_integrity["file_exists"], "CSV file should exist"
                    assert csv_integrity["header_valid"], "CSV headers should be valid"
                    assert (
                        csv_integrity["total_rows"] >= 1
                    ), "Should have at least one row"

                # Verify hook statistics
                stats = hook.get_statistics()
                assert stats["executions_started"] >= 1, "Should track execution starts"

                if expected_tokens > 0:
                    assert (
                        stats["executions_completed"] >= 1
                    ), "Should track completions"
                    assert (
                        stats["total_tokens_tracked"] >= expected_tokens
                    ), "Should track total tokens"

            finally:
                # Cleanup
                hook.cleanup()
                if csv_path.exists():
                    csv_path.unlink()
