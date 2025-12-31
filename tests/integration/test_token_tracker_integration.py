"""
Comprehensive integration tests for Token Tracker system.

This module tests complete end-to-end token tracking workflows,
hook integration with real Kiro IDE events, error scenarios,
and documentation builds.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
import pytest

from ai_hydra.token_tracker.tracker import TokenTracker
from ai_hydra.token_tracker.hook import TokenTrackingHook
from ai_hydra.token_tracker.metadata_collector import MetadataCollector
from ai_hydra.token_tracker.models import TokenTransaction, TrackerConfig
from ai_hydra.token_tracker.csv_writer import CSVWriter
from ai_hydra.token_tracker.error_handler import TokenTrackerErrorHandler


class TestTokenTrackerEndToEndIntegration:
    """Test complete end-to-end token tracking workflows."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        # Create temporary directory for test files
        self.temp_dir = Path(tempfile.mkdtemp())
        self.csv_file = self.temp_dir / "test_token_transactions.csv"

        # Create test configuration
        self.config = TrackerConfig(
            enabled=True,
            csv_file_path=str(self.csv_file),
            max_prompt_length=1000,
            backup_enabled=True,
            backup_interval_hours=24,
            compression_enabled=False,
            retention_days=365,
            auto_create_directories=True,
            file_lock_timeout_seconds=5.0,
            max_concurrent_writes=10,
            enable_validation=True,
            log_level="DEBUG",
        )

        # Initialize components
        self.tracker = TokenTracker(self.config)
        self.hook = TokenTrackingHook(self.config)

        # Mock execution contexts
        self.mock_contexts = self._create_mock_contexts()

    def teardown_method(self):
        """Cleanup after each test method."""
        # Clean up temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_complete_token_tracking_workflow(self):
        """Test complete workflow from hook trigger to CSV storage."""
        # Simulate agent execution start
        execution_id = str(uuid.uuid4())
        start_context = {
            "execution_id": execution_id,
            "trigger_type": "agentExecutionStarted",
            "workspace_folder": "ai_hydra",
            "file_patterns": ["*.py", "*.md"],
            "hook_name": "token-tracker-hook",
            "session_id": str(uuid.uuid4()),
            "user_message": "Implement token tracking system",
        }

        # Start execution
        self.hook.on_agent_execution_start(start_context)

        # Verify execution context was stored
        assert execution_id in self.hook.execution_context
        stored_context = self.hook.execution_context[execution_id]
        assert stored_context["trigger_type"] == "agentExecutionStarted"
        assert stored_context["workspace_folder"] == "ai_hydra"

        # Simulate agent execution completion
        completion_result = {
            "execution_id": execution_id,
            "success": True,
            "token_usage": {
                "total_tokens": 1250,
                "prompt": "Implement a comprehensive token tracking system for Kiro IDE",
                "model": "gpt-4",
                "response": "I'll help you implement the token tracking system...",
            },
            "duration": 2.34,
            "output": "Token tracking system implementation completed",
        }

        # Complete execution
        self.hook.on_agent_execution_complete(start_context, completion_result)

        # Verify transaction was recorded
        assert self.csv_file.exists()

        # Read and verify CSV content
        transactions = self.tracker.get_transaction_history()
        assert len(transactions) == 1

        transaction = transactions[0]
        assert transaction.tokens_used == 1250
        assert (
            "Implement a comprehensive token tracking system" in transaction.prompt_text
        )
        assert transaction.workspace_folder == "ai_hydra"
        assert transaction.hook_trigger_type == "agentExecutionStarted"
        assert transaction.hook_name == "token-tracker-hook"
        assert not transaction.error_occurred

        # Verify execution context was cleaned up
        assert execution_id not in self.hook.execution_context

    def test_multiple_concurrent_executions(self):
        """Test handling multiple concurrent agent executions."""
        execution_ids = [str(uuid.uuid4()) for _ in range(5)]

        # Start multiple executions
        for i, exec_id in enumerate(execution_ids):
            context = {
                "execution_id": exec_id,
                "trigger_type": "agentExecutionStarted",
                "workspace_folder": f"workspace_{i}",
                "file_patterns": ["*.py"],
                "hook_name": "token-tracker-hook",
                "session_id": str(uuid.uuid4()),
            }
            self.hook.on_agent_execution_start(context)

        # Verify all contexts are stored
        assert len(self.hook.execution_context) == 5

        # Complete executions in different order
        completion_order = [2, 0, 4, 1, 3]
        for i in completion_order:
            exec_id = execution_ids[i]
            context = {"execution_id": exec_id}
            result = {
                "execution_id": exec_id,
                "success": True,
                "token_usage": {
                    "total_tokens": 100 + i * 50,
                    "prompt": f"Task {i} prompt",
                    "model": "gpt-3.5-turbo",
                },
                "duration": 1.0 + i * 0.5,
            }
            self.hook.on_agent_execution_complete(context, result)

        # Verify all transactions were recorded
        transactions = self.tracker.get_transaction_history()
        assert len(transactions) == 5

        # Verify token counts match expected values
        token_counts = sorted([t.tokens_used for t in transactions])
        expected_counts = sorted([100 + i * 50 for i in range(5)])
        assert token_counts == expected_counts

        # Verify all contexts were cleaned up
        assert len(self.hook.execution_context) == 0

    def test_error_recovery_during_execution(self):
        """Test error recovery mechanisms during execution."""
        execution_id = str(uuid.uuid4())

        # Start execution
        start_context = {
            "execution_id": execution_id,
            "trigger_type": "agentExecutionStarted",
            "workspace_folder": "test_workspace",
            "hook_name": "token-tracker-hook",
            "session_id": str(uuid.uuid4()),
        }
        self.hook.on_agent_execution_start(start_context)

        # Simulate CSV write error by making file read-only
        self.csv_file.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file.touch()
        self.csv_file.chmod(0o444)  # Read-only

        try:
            # Complete execution (should handle write error gracefully)
            completion_result = {
                "execution_id": execution_id,
                "success": True,
                "token_usage": {
                    "total_tokens": 500,
                    "prompt": "Test prompt with write error",
                    "model": "gpt-3.5-turbo",
                },
                "duration": 1.5,
            }

            # This should not raise an exception
            self.hook.on_agent_execution_complete(start_context, completion_result)

            # Verify hook statistics reflect the failure
            stats = self.hook.get_statistics()
            assert stats["executions_failed"] > 0

        finally:
            # Restore write permissions for cleanup
            self.csv_file.chmod(0o644)

    def test_metadata_accuracy_across_scenarios(self):
        """Test metadata accuracy across different execution scenarios."""
        test_scenarios = [
            {
                "name": "File Edit Trigger",
                "context": {
                    "trigger_type": "fileEdited",
                    "workspace_folder": "ai_hydra",
                    "file_patterns": ["*.py"],
                    "current_file": "ai_hydra/token_tracker/tracker.py",
                    "modified_files": ["ai_hydra/token_tracker/tracker.py"],
                },
            },
            {
                "name": "User Message Trigger",
                "context": {
                    "trigger_type": "userMessage",
                    "workspace_folder": "docs",
                    "user_message": "Update documentation for token tracking",
                    "file_patterns": ["*.rst", "*.md"],
                },
            },
            {
                "name": "Scheduled Trigger",
                "context": {
                    "trigger_type": "scheduled",
                    "workspace_folder": "tests",
                    "file_patterns": ["test_*.py"],
                    "schedule_info": {
                        "interval": "hourly",
                        "last_run": "2024-01-01T10:00:00Z",
                    },
                },
            },
        ]

        for scenario in test_scenarios:
            execution_id = str(uuid.uuid4())

            # Add required fields to context
            context = scenario["context"].copy()
            context.update(
                {
                    "execution_id": execution_id,
                    "hook_name": "token-tracker-hook",
                    "session_id": str(uuid.uuid4()),
                }
            )

            # Execute workflow
            self.hook.on_agent_execution_start(context)

            result = {
                "execution_id": execution_id,
                "success": True,
                "token_usage": {
                    "total_tokens": 200,
                    "prompt": f"Test prompt for {scenario['name']}",
                    "model": "gpt-3.5-turbo",
                },
                "duration": 1.0,
            }

            self.hook.on_agent_execution_complete(context, result)

        # Verify all transactions were recorded with correct metadata
        transactions = self.tracker.get_transaction_history()
        assert len(transactions) == len(test_scenarios)

        # Check specific metadata for each scenario
        for i, scenario in enumerate(test_scenarios):
            transaction = next(
                t for t in transactions if scenario["name"] in t.prompt_text
            )

            assert transaction.hook_trigger_type == scenario["context"]["trigger_type"]
            assert (
                transaction.workspace_folder == scenario["context"]["workspace_folder"]
            )

            if "file_patterns" in scenario["context"]:
                expected_patterns = ";".join(scenario["context"]["file_patterns"])
                assert transaction.file_patterns == expected_patterns

    def test_configuration_management_integration(self):
        """Test configuration management during runtime."""
        # Start with initial configuration
        initial_stats = self.hook.get_statistics()
        assert initial_stats["tracker_statistics"]["transactions_recorded"] == 0

        # Record a transaction
        self._record_test_transaction("Initial transaction")

        # Verify transaction was recorded
        stats_after_first = self.hook.get_statistics()
        assert stats_after_first["tracker_statistics"]["transactions_recorded"] == 1

        # Update configuration to disable tracking
        new_config = TrackerConfig(
            enabled=False,
            csv_file_path=str(self.csv_file),
            max_prompt_length=500,  # Changed
            backup_enabled=False,  # Changed
            retention_days=180,  # Changed
            log_level="WARNING",  # Changed
        )

        success = self.hook.update_configuration(new_config)
        assert success
        assert not self.hook.is_enabled
        assert self.hook.config.max_prompt_length == 500

        # Try to record another transaction (should be skipped)
        self._record_test_transaction("Disabled transaction")

        # Verify no new transaction was recorded
        stats_after_disable = self.hook.get_statistics()
        assert stats_after_disable["tracker_statistics"]["transactions_recorded"] == 1

        # Re-enable tracking
        enabled_config = TrackerConfig(
            enabled=True,
            csv_file_path=str(self.csv_file),
            max_prompt_length=500,
            backup_enabled=False,
            retention_days=180,
            log_level="WARNING",
        )

        success = self.hook.update_configuration(enabled_config)
        assert success
        assert self.hook.is_enabled

        # Record another transaction
        self._record_test_transaction("Re-enabled transaction")

        # Verify new transaction was recorded
        final_stats = self.hook.get_statistics()
        assert final_stats["tracker_statistics"]["transactions_recorded"] == 2

    def test_system_behavior_under_various_error_conditions(self):
        """Test system behavior under various error conditions."""
        error_scenarios = [
            {
                "name": "Invalid Token Count",
                "token_usage": {
                    "total_tokens": -100,  # Invalid negative value
                    "prompt": "Test prompt",
                    "model": "gpt-3.5-turbo",
                },
                "expected_behavior": "graceful_handling",
            },
            {
                "name": "Missing Prompt",
                "token_usage": {
                    "total_tokens": 150,
                    "model": "gpt-3.5-turbo",
                    # Missing prompt field
                },
                "expected_behavior": "use_fallback_prompt",
            },
            {
                "name": "Malformed Token Usage",
                "token_usage": "invalid_format",  # Not a dictionary
                "expected_behavior": "extract_zero_tokens",
            },
            {
                "name": "Very Long Prompt",
                "token_usage": {
                    "total_tokens": 200,
                    "prompt": "A" * 5000,  # Exceeds max_prompt_length
                    "model": "gpt-3.5-turbo",
                },
                "expected_behavior": "truncate_prompt",
            },
        ]

        for scenario in error_scenarios:
            execution_id = str(uuid.uuid4())

            context = {
                "execution_id": execution_id,
                "trigger_type": "manual_test",
                "workspace_folder": "test_workspace",
                "hook_name": "token-tracker-hook",
                "session_id": str(uuid.uuid4()),
            }

            result = {
                "execution_id": execution_id,
                "success": True,
                "token_usage": scenario["token_usage"],
                "duration": 1.0,
            }

            # Execute workflow (should not raise exceptions)
            self.hook.on_agent_execution_start(context)
            self.hook.on_agent_execution_complete(context, result)

        # Verify system handled all error scenarios gracefully
        transactions = self.tracker.get_transaction_history()

        # Should have recorded transactions for scenarios that could be recovered
        assert len(transactions) >= 2  # At least some scenarios should succeed

        # Verify specific error handling behaviors
        for transaction in transactions:
            # All recorded transactions should have valid data
            assert transaction.tokens_used >= 0
            assert len(transaction.prompt_text) <= self.config.max_prompt_length
            assert transaction.prompt_text.strip()  # Not empty

    def test_performance_under_load(self):
        """Test system performance under load conditions."""
        # Record start time
        start_time = time.time()

        # Simulate high-frequency token tracking
        num_transactions = 50
        execution_ids = []

        for i in range(num_transactions):
            execution_id = str(uuid.uuid4())
            execution_ids.append(execution_id)

            context = {
                "execution_id": execution_id,
                "trigger_type": "agentExecutionStarted",
                "workspace_folder": f"workspace_{i % 5}",
                "file_patterns": ["*.py"],
                "hook_name": "token-tracker-hook",
                "session_id": str(uuid.uuid4()),
            }

            result = {
                "execution_id": execution_id,
                "success": True,
                "token_usage": {
                    "total_tokens": 100 + (i % 200),
                    "prompt": f"Load test prompt {i}",
                    "model": "gpt-3.5-turbo",
                },
                "duration": 0.5 + (i % 10) * 0.1,
            }

            # Execute workflow
            self.hook.on_agent_execution_start(context)
            self.hook.on_agent_execution_complete(context, result)

        # Record end time
        end_time = time.time()
        total_time = end_time - start_time

        # Verify all transactions were recorded
        transactions = self.tracker.get_transaction_history()
        assert len(transactions) == num_transactions

        # Verify performance metrics
        avg_time_per_transaction = total_time / num_transactions
        assert avg_time_per_transaction < 0.1  # Should be fast

        # Verify CSV file integrity
        integrity_result = self.tracker.validate_csv_integrity()
        assert integrity_result["file_exists"]
        assert len(integrity_result.get("validation_issues", [])) == 0

        # Verify statistics are accurate
        stats = self.tracker.get_statistics()
        assert stats["transactions_recorded"] == num_transactions
        assert stats["total_tokens_tracked"] > 0

    def _record_test_transaction(self, prompt_text: str) -> None:
        """Helper method to record a test transaction."""
        execution_id = str(uuid.uuid4())

        context = {
            "execution_id": execution_id,
            "trigger_type": "manual_test",
            "workspace_folder": "test_workspace",
            "hook_name": "token-tracker-hook",
            "session_id": str(uuid.uuid4()),
        }

        result = {
            "execution_id": execution_id,
            "success": True,
            "token_usage": {
                "total_tokens": 150,
                "prompt": prompt_text,
                "model": "gpt-3.5-turbo",
            },
            "duration": 1.0,
        }

        self.hook.on_agent_execution_start(context)
        self.hook.on_agent_execution_complete(context, result)

    def _create_mock_contexts(self) -> List[Dict[str, Any]]:
        """Create mock execution contexts for testing."""
        return [
            {
                "execution_id": str(uuid.uuid4()),
                "trigger_type": "agentExecutionStarted",
                "workspace_folder": "ai_hydra",
                "file_patterns": ["*.py", "*.md"],
                "hook_name": "token-tracker-hook",
                "session_id": str(uuid.uuid4()),
                "user_message": "Implement feature X",
            },
            {
                "execution_id": str(uuid.uuid4()),
                "trigger_type": "fileEdited",
                "workspace_folder": "docs",
                "file_patterns": ["*.rst"],
                "hook_name": "documentation-hook",
                "session_id": str(uuid.uuid4()),
                "current_file": "docs/index.rst",
            },
            {
                "execution_id": str(uuid.uuid4()),
                "trigger_type": "userMessage",
                "workspace_folder": "tests",
                "file_patterns": ["test_*.py"],
                "hook_name": "test-runner-hook",
                "session_id": str(uuid.uuid4()),
                "user_message": "Run integration tests",
            },
        ]


class TestDocumentationIntegration:
    """Test documentation builds and navigation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.docs_dir = Path("docs")
        self.source_dir = self.docs_dir / "_source"
        self.build_dir = self.docs_dir / "_build"

    def test_documentation_build_integrity(self):
        """Test that documentation builds without errors."""
        if not self.docs_dir.exists():
            pytest.skip("Documentation directory not found")

        # Clean previous build
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)

        # Build documentation
        import subprocess

        result = subprocess.run(
            ["make", "html"], cwd=self.docs_dir, capture_output=True, text=True
        )

        # Check build success
        assert result.returncode == 0, f"Documentation build failed: {result.stderr}"

        # Verify HTML files were generated
        html_dir = self.build_dir / "html"
        assert html_dir.exists(), "HTML build directory not created"

        html_files = list(html_dir.glob("*.html"))
        assert len(html_files) >= 5, f"Too few HTML files generated: {len(html_files)}"

    def test_token_tracker_documentation_content(self):
        """Test that token tracker documentation content is present."""
        if not self.source_dir.exists():
            pytest.skip("Documentation source directory not found")

        # Check for token tracker documentation
        runbook_dir = self.source_dir / "runbook"
        if runbook_dir.exists():
            token_tracking_doc = runbook_dir / "token_tracking.rst"
            if token_tracking_doc.exists():
                content = token_tracking_doc.read_text()

                # Verify key sections are present
                required_sections = [
                    "Token Tracking",
                    "Configuration",
                    "Usage",
                    "Troubleshooting",
                ]

                for section in required_sections:
                    assert section in content, f"Missing section: {section}"

    def test_steering_documentation_integration(self):
        """Test that steering documentation is properly integrated."""
        steering_dir = Path(".kiro/steering")
        if not steering_dir.exists():
            pytest.skip("Steering directory not found")

        # Check for token tracking standards
        token_standards_file = steering_dir / "token-tracking-standards.md"
        if token_standards_file.exists():
            content = token_standards_file.read_text()

            # Verify key content is present
            required_content = [
                "Token Tracking Standards",
                "CSV Format Specifications",
                "Agent Hook Configuration",
                "Error Handling Standards",
            ]

            for content_item in required_content:
                assert content_item in content, f"Missing content: {content_item}"


class TestRealKiroIDEIntegration:
    """Test integration with real Kiro IDE events (mocked)."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.csv_file = self.temp_dir / "kiro_token_transactions.csv"

        self.config = TrackerConfig(
            enabled=True,
            csv_file_path=str(self.csv_file),
            max_prompt_length=2000,
            backup_enabled=True,
            log_level="INFO",
        )

        self.hook = TokenTrackingHook(self.config)

    def teardown_method(self):
        """Cleanup after each test."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_kiro_agent_execution_simulation(self):
        """Test simulation of real Kiro IDE agent execution events."""
        # Mock Kiro IDE environment variables
        with patch.dict(
            os.environ,
            {
                "KIRO_SESSION_ID": "test-session-123",
                "KIRO_WORKSPACE_FOLDER": "ai_hydra",
                "KIRO_HOOK_TYPE": "agentExecutionCompleted",
                "KIRO_HOOK_NAME": "token-tracker-hook",
            },
        ):
            # Simulate realistic Kiro IDE execution context
            execution_context = {
                "execution_id": "exec_" + str(uuid.uuid4()),
                "trigger_type": "agentExecutionCompleted",
                "workspace_folder": "ai_hydra",
                "file_patterns": ["*.py", "*.md", "*.rst"],
                "hook_name": "token-tracker-hook",
                "session_id": os.environ["KIRO_SESSION_ID"],
                "timestamp": datetime.now().isoformat(),
                "user_message": "Implement comprehensive token tracking with error handling",
                "agent_config": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 2000,
                },
            }

            # Simulate execution result with realistic token usage
            execution_result = {
                "execution_id": execution_context["execution_id"],
                "success": True,
                "token_usage": {
                    "total_tokens": 1847,
                    "prompt_tokens": 1203,
                    "completion_tokens": 644,
                    "prompt": execution_context["user_message"],
                    "model": "gpt-4",
                    "response": "I'll implement a comprehensive token tracking system...",
                    "finish_reason": "stop",
                },
                "duration": 4.23,
                "output": "Token tracking implementation completed successfully",
                "metadata": {
                    "files_modified": ["ai_hydra/token_tracker/tracker.py"],
                    "tests_run": 15,
                    "tests_passed": 15,
                },
            }

            # Execute the hook workflow
            self.hook.on_agent_execution_start(execution_context)
            self.hook.on_agent_execution_complete(execution_context, execution_result)

            # Verify transaction was recorded correctly
            assert self.csv_file.exists()

            # Read and validate the recorded transaction
            tracker = TokenTracker(self.config)
            transactions = tracker.get_transaction_history()

            assert len(transactions) == 1
            transaction = transactions[0]

            # Verify all expected fields are present and correct
            assert transaction.tokens_used == 1847
            assert transaction.prompt_text == execution_context["user_message"]
            assert transaction.workspace_folder == "ai_hydra"
            assert transaction.hook_trigger_type == "agentExecutionCompleted"
            assert transaction.hook_name == "token-tracker-hook"
            assert transaction.session_id == "test-session-123"
            assert transaction.agent_execution_id == execution_context["execution_id"]
            assert transaction.file_patterns == "*.py;*.md;*.rst"
            assert not transaction.error_occurred
            assert transaction.elapsed_time > 0

    def test_multiple_workspace_scenarios(self):
        """Test token tracking across multiple workspace scenarios."""
        workspaces = [
            {
                "name": "ai_hydra",
                "type": "python",
                "files": ["*.py", "*.toml", "*.md"],
                "typical_prompts": [
                    "Implement neural network training logic",
                    "Add error handling to game logic",
                    "Optimize tree search algorithm",
                ],
            },
            {
                "name": "docs",
                "type": "documentation",
                "files": ["*.rst", "*.md"],
                "typical_prompts": [
                    "Update API documentation",
                    "Add troubleshooting guide",
                    "Create getting started tutorial",
                ],
            },
            {
                "name": "tests",
                "type": "testing",
                "files": ["test_*.py", "*.yaml"],
                "typical_prompts": [
                    "Write integration tests",
                    "Add property-based tests",
                    "Fix failing test cases",
                ],
            },
        ]

        for workspace in workspaces:
            for i, prompt in enumerate(workspace["typical_prompts"]):
                execution_id = f"{workspace['name']}_exec_{i}"

                context = {
                    "execution_id": execution_id,
                    "trigger_type": "agentExecutionStarted",
                    "workspace_folder": workspace["name"],
                    "file_patterns": workspace["files"],
                    "hook_name": "token-tracker-hook",
                    "session_id": str(uuid.uuid4()),
                    "user_message": prompt,
                    "workspace_type": workspace["type"],
                }

                result = {
                    "execution_id": execution_id,
                    "success": True,
                    "token_usage": {
                        "total_tokens": 300 + i * 100,
                        "prompt": prompt,
                        "model": "gpt-3.5-turbo",
                    },
                    "duration": 1.5 + i * 0.5,
                }

                self.hook.on_agent_execution_start(context)
                self.hook.on_agent_execution_complete(context, result)

        # Verify all transactions were recorded
        tracker = TokenTracker(self.config)
        transactions = tracker.get_transaction_history()

        total_expected = sum(len(ws["typical_prompts"]) for ws in workspaces)
        assert len(transactions) == total_expected

        # Verify workspace distribution
        workspace_counts = {}
        for transaction in transactions:
            ws = transaction.workspace_folder
            workspace_counts[ws] = workspace_counts.get(ws, 0) + 1

        for workspace in workspaces:
            expected_count = len(workspace["typical_prompts"])
            actual_count = workspace_counts.get(workspace["name"], 0)
            assert actual_count == expected_count

    def test_error_scenario_integration(self):
        """Test integration with various error scenarios."""
        error_scenarios = [
            {
                "name": "Agent Execution Failure",
                "context": {
                    "execution_id": "error_exec_1",
                    "trigger_type": "agentExecutionCompleted",
                    "workspace_folder": "ai_hydra",
                    "hook_name": "token-tracker-hook",
                },
                "result": {
                    "execution_id": "error_exec_1",
                    "success": False,
                    "error": "Compilation error in generated code",
                    "token_usage": {
                        "total_tokens": 500,
                        "prompt": "Fix the compilation error",
                        "model": "gpt-3.5-turbo",
                    },
                    "duration": 2.1,
                },
            },
            {
                "name": "Timeout Error",
                "context": {
                    "execution_id": "error_exec_2",
                    "trigger_type": "agentExecutionCompleted",
                    "workspace_folder": "ai_hydra",
                    "hook_name": "token-tracker-hook",
                },
                "result": {
                    "execution_id": "error_exec_2",
                    "success": False,
                    "error": "Request timeout after 30 seconds",
                    "token_usage": {
                        "total_tokens": 0,  # No tokens used due to timeout
                        "prompt": "Generate complex algorithm implementation",
                        "model": "gpt-4",
                    },
                    "duration": 30.0,
                },
            },
            {
                "name": "Rate Limit Error",
                "context": {
                    "execution_id": "error_exec_3",
                    "trigger_type": "agentExecutionCompleted",
                    "workspace_folder": "ai_hydra",
                    "hook_name": "token-tracker-hook",
                },
                "result": {
                    "execution_id": "error_exec_3",
                    "success": False,
                    "error": "Rate limit exceeded, please try again later",
                    "token_usage": {
                        "total_tokens": 0,
                        "prompt": "Analyze large codebase",
                        "model": "gpt-4",
                    },
                    "duration": 0.5,
                },
            },
        ]

        for scenario in error_scenarios:
            # Execute the error scenario
            self.hook.on_agent_execution_start(scenario["context"])
            self.hook.on_agent_execution_complete(
                scenario["context"], scenario["result"]
            )

        # Verify transactions were recorded even for failed executions
        tracker = TokenTracker(self.config)
        transactions = tracker.get_transaction_history()

        # Should record transactions for scenarios with token usage > 0
        recorded_transactions = [t for t in transactions if t.tokens_used > 0]
        assert len(recorded_transactions) == 1  # Only the first scenario used tokens

        # Verify error information is captured
        error_transaction = recorded_transactions[0]
        assert error_transaction.tokens_used == 500
        assert "Fix the compilation error" in error_transaction.prompt_text

        # Verify hook statistics reflect the errors
        stats = self.hook.get_statistics()
        assert stats["executions_started"] == 3
        assert stats["executions_completed"] >= 1  # At least one completed successfully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
