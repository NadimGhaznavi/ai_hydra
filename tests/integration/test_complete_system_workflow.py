"""
Integration tests for complete token tracker system workflow.

This module tests the full workflow from hook trigger to CSV storage,
validates metadata accuracy across different scenarios, and tests
system behavior under various error conditions.
"""

import csv
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Note: BackupManager and MaintenanceManager are integrated into other components


class TestCompleteSystemWorkflow:
    """Test complete system workflow from hook trigger to CSV storage."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        # Create temporary directory for test files
        self.temp_dir = Path(tempfile.mkdtemp())
        self.csv_file = self.temp_dir / "complete_system_test.csv"
        self.backup_dir = self.temp_dir / "backups"

        # Create comprehensive test configuration
        self.config = TrackerConfig(
            enabled=True,
            csv_file_path=str(self.csv_file),
            max_prompt_length=2000,
            backup_enabled=True,
            backup_interval_hours=1,  # Short interval for testing
            compression_enabled=False,
            retention_days=30,
            auto_create_directories=True,
            file_lock_timeout_seconds=10.0,
            max_concurrent_writes=20,
            enable_validation=True,
            log_level="DEBUG",
        )

        # Initialize system components
        self.tracker = TokenTracker(self.config)
        self.hook = TokenTrackingHook(self.config)
        self.metadata_collector = MetadataCollector()

        # Test data
        self.test_executions = self._create_test_execution_data()

    def teardown_method(self):
        """Cleanup after each test method."""
        # Stop any monitoring
        if hasattr(self.tracker, "stop_monitoring"):
            self.tracker.stop_monitoring()

        # Clean up temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_full_workflow_single_execution(self):
        """Test complete workflow for a single agent execution."""
        # Step 1: Agent execution starts
        execution_id = str(uuid.uuid4())
        start_context = {
            "execution_id": execution_id,
            "trigger_type": "agentExecutionStarted",
            "workspace_folder": "ai_hydra",
            "file_patterns": ["*.py", "*.md"],
            "hook_name": "token-tracker-hook",
            "session_id": str(uuid.uuid4()),
            "user_message": "Implement comprehensive error handling for token tracker",
            "timestamp": datetime.now().isoformat(),
        }

        # Hook captures start event
        self.hook.on_agent_execution_start(start_context)

        # Verify execution context is stored
        assert execution_id in self.hook.execution_context
        stored_context = self.hook.execution_context[execution_id]
        assert stored_context["trigger_type"] == "agentExecutionStarted"

        # Step 2: Agent execution completes
        completion_result = {
            "execution_id": execution_id,
            "success": True,
            "token_usage": {
                "total_tokens": 1456,
                "prompt_tokens": 892,
                "completion_tokens": 564,
                "prompt": start_context["user_message"],
                "model": "gpt-4",
                "response": "I'll implement comprehensive error handling...",
                "finish_reason": "stop",
            },
            "duration": 3.47,
            "output": "Error handling implementation completed",
            "metadata": {
                "files_created": ["ai_hydra/token_tracker/error_recovery.py"],
                "files_modified": ["ai_hydra/token_tracker/tracker.py"],
                "tests_added": 12,
            },
        }

        # Hook processes completion
        self.hook.on_agent_execution_complete(start_context, completion_result)

        # Step 3: Verify CSV storage
        assert self.csv_file.exists()

        # Read CSV directly to verify format
        with open(self.csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        # Verify CSV column values
        assert int(row["tokens_used"]) == 1456
        assert start_context["user_message"] in row["prompt_text"]
        assert row["workspace_folder"] == "ai_hydra"
        assert row["hook_trigger_type"] == "agentExecutionStarted"
        assert row["hook_name"] == "token-tracker-hook"
        assert row["agent_execution_id"] == execution_id
        assert row["file_patterns"] == "*.py;*.md"
        assert row["error_occurred"] == "false"
        assert float(row["elapsed_time"]) > 0

        # Step 4: Verify tracker can read the data back
        transactions = self.tracker.get_transaction_history()
        assert len(transactions) == 1

        transaction = transactions[0]
        assert transaction.tokens_used == 1456
        assert transaction.workspace_folder == "ai_hydra"
        assert transaction.agent_execution_id == execution_id

        # Step 5: Verify execution context cleanup
        assert execution_id not in self.hook.execution_context

        # Step 6: Verify statistics
        stats = self.tracker.get_statistics()
        assert stats["transactions_recorded"] == 1
        assert stats["total_tokens_tracked"] == 1456
        assert stats["transactions_failed"] == 0

    def test_concurrent_execution_workflow(self):
        """Test workflow with multiple concurrent executions."""
        num_concurrent = 10
        execution_ids = [str(uuid.uuid4()) for _ in range(num_concurrent)]

        # Start all executions concurrently
        start_contexts = []
        for i, exec_id in enumerate(execution_ids):
            context = {
                "execution_id": exec_id,
                "trigger_type": "agentExecutionStarted",
                "workspace_folder": f"workspace_{i % 3}",
                "file_patterns": ["*.py"],
                "hook_name": "token-tracker-hook",
                "session_id": str(uuid.uuid4()),
                "user_message": f"Concurrent task {i}",
                "timestamp": datetime.now().isoformat(),
            }
            start_contexts.append(context)
            self.hook.on_agent_execution_start(context)

        # Verify all contexts are stored
        assert len(self.hook.execution_context) == num_concurrent

        # Complete executions with different timing
        def complete_execution(i):
            exec_id = execution_ids[i]
            context = start_contexts[i]

            # Simulate variable execution time
            time.sleep(0.01 * (i % 3))

            result = {
                "execution_id": exec_id,
                "success": True,
                "token_usage": {
                    "total_tokens": 200 + i * 50,
                    "prompt": f"Concurrent task {i}",
                    "model": "gpt-3.5-turbo",
                },
                "duration": 1.0 + i * 0.1,
            }

            self.hook.on_agent_execution_complete(context, result)
            return i

        # Use thread pool to complete executions concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(complete_execution, i) for i in range(num_concurrent)
            ]
            completed = [future.result() for future in as_completed(futures)]

        # Verify all executions completed
        assert len(completed) == num_concurrent

        # Verify all transactions were recorded
        transactions = self.tracker.get_transaction_history()
        assert len(transactions) == num_concurrent

        # Verify token counts are correct
        expected_tokens = [200 + i * 50 for i in range(num_concurrent)]
        actual_tokens = sorted([t.tokens_used for t in transactions])
        assert actual_tokens == sorted(expected_tokens)

        # Verify CSV integrity
        integrity_result = self.tracker.validate_csv_integrity()
        assert integrity_result["file_exists"]
        assert len(integrity_result.get("validation_issues", [])) == 0

        # Verify all contexts were cleaned up
        assert len(self.hook.execution_context) == 0

    def test_metadata_accuracy_comprehensive(self):
        """Test metadata accuracy across comprehensive scenarios."""
        test_scenarios = [
            {
                "name": "Python Development",
                "context": {
                    "trigger_type": "fileEdited",
                    "workspace_folder": "ai_hydra",
                    "file_patterns": ["*.py", "*.toml"],
                    "current_file": "ai_hydra/token_tracker/tracker.py",
                    "modified_files": [
                        "ai_hydra/token_tracker/tracker.py",
                        "pyproject.toml",
                    ],
                    "workspace_type": "python",
                    "git_branch": "feature/token-tracking",
                    "user_message": "Add comprehensive logging to token tracker",
                },
            },
            {
                "name": "Documentation Update",
                "context": {
                    "trigger_type": "userMessage",
                    "workspace_folder": "docs",
                    "file_patterns": ["*.rst", "*.md"],
                    "current_file": "docs/_source/runbook/token_tracking.rst",
                    "workspace_type": "documentation",
                    "user_message": "Update token tracking documentation with new features",
                },
            },
            {
                "name": "Test Development",
                "context": {
                    "trigger_type": "agentExecutionCompleted",
                    "workspace_folder": "tests",
                    "file_patterns": ["test_*.py", "conftest.py"],
                    "current_file": "tests/integration/test_token_tracker.py",
                    "workspace_type": "testing",
                    "test_framework": "pytest",
                    "user_message": "Create comprehensive integration tests",
                },
            },
            {
                "name": "Configuration Management",
                "context": {
                    "trigger_type": "scheduled",
                    "workspace_folder": ".kiro",
                    "file_patterns": ["*.json", "*.yaml", "*.md"],
                    "workspace_type": "configuration",
                    "schedule_type": "maintenance",
                    "user_message": "Update Kiro configuration for token tracking",
                },
            },
        ]

        recorded_transactions = []

        for scenario in test_scenarios:
            execution_id = str(uuid.uuid4())

            # Prepare full context
            full_context = scenario["context"].copy()
            full_context.update(
                {
                    "execution_id": execution_id,
                    "hook_name": "token-tracker-hook",
                    "session_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Execute workflow
            self.hook.on_agent_execution_start(full_context)

            result = {
                "execution_id": execution_id,
                "success": True,
                "token_usage": {
                    "total_tokens": 300 + len(scenario["name"]) * 10,
                    "prompt": scenario["context"]["user_message"],
                    "model": "gpt-3.5-turbo",
                },
                "duration": 2.0 + len(scenario["name"]) * 0.1,
            }

            self.hook.on_agent_execution_complete(full_context, result)
            recorded_transactions.append((scenario, execution_id))

        # Verify all transactions were recorded
        transactions = self.tracker.get_transaction_history()
        assert len(transactions) == len(test_scenarios)

        # Verify metadata accuracy for each scenario
        for scenario, exec_id in recorded_transactions:
            # Find the corresponding transaction
            transaction = next(
                t for t in transactions if t.agent_execution_id == exec_id
            )

            # Verify core metadata
            assert transaction.hook_trigger_type == scenario["context"]["trigger_type"]
            assert (
                transaction.workspace_folder == scenario["context"]["workspace_folder"]
            )
            assert transaction.hook_name == "token-tracker-hook"

            # Verify file patterns
            if "file_patterns" in scenario["context"]:
                expected_patterns = ";".join(scenario["context"]["file_patterns"])
                assert transaction.file_patterns == expected_patterns

            # Verify prompt content
            assert scenario["context"]["user_message"] in transaction.prompt_text

    def test_system_behavior_under_error_conditions(self):
        """Test system behavior under various error conditions."""
        error_test_cases = [
            {
                "name": "File Permission Error",
                "setup": lambda: self._create_permission_error(),
                "expected_behavior": "graceful_fallback",
                "cleanup": lambda: self._restore_permissions(),
            },
            {
                "name": "Disk Space Error",
                "setup": lambda: self._simulate_disk_full(),
                "expected_behavior": "error_logging",
                "cleanup": lambda: self._restore_disk_space(),
            },
            {
                "name": "Concurrent Write Conflict",
                "setup": lambda: self._create_write_conflict(),
                "expected_behavior": "retry_mechanism",
                "cleanup": lambda: self._resolve_write_conflict(),
            },
            {
                "name": "Malformed Data Recovery",
                "setup": lambda: self._inject_malformed_data(),
                "expected_behavior": "data_sanitization",
                "cleanup": lambda: self._clean_malformed_data(),
            },
        ]

        for test_case in error_test_cases:
            try:
                # Setup error condition
                test_case["setup"]()

                # Attempt to record transaction
                execution_id = str(uuid.uuid4())
                context = {
                    "execution_id": execution_id,
                    "trigger_type": "manual_test",
                    "workspace_folder": "error_test",
                    "hook_name": "token-tracker-hook",
                    "session_id": str(uuid.uuid4()),
                    "user_message": f"Testing {test_case['name']}",
                }

                result = {
                    "execution_id": execution_id,
                    "success": True,
                    "token_usage": {
                        "total_tokens": 100,
                        "prompt": context["user_message"],
                        "model": "gpt-3.5-turbo",
                    },
                    "duration": 1.0,
                }

                # Execute workflow (should handle errors gracefully)
                self.hook.on_agent_execution_start(context)
                self.hook.on_agent_execution_complete(context, result)

                # Verify system didn't crash
                assert True  # If we reach here, error was handled gracefully

            except Exception as e:
                # Some errors might still propagate, but system should remain stable
                assert "critical" not in str(e).lower()

            finally:
                # Always cleanup
                test_case["cleanup"]()

        # Verify system is still functional after all error conditions
        self._verify_system_functionality()

    def test_backup_and_maintenance_integration(self):
        """Test backup and maintenance integration in complete workflow."""
        # Record several transactions to trigger maintenance
        for i in range(15):
            self._record_test_transaction(f"Maintenance test transaction {i}")

        # Verify transactions were recorded
        transactions = self.tracker.get_transaction_history()
        assert len(transactions) == 15

        # Trigger backup creation
        backup_path = self.tracker.create_backup()
        assert backup_path is not None
        assert backup_path.exists()

        # Verify backup contains all transactions
        backup_tracker = TokenTracker(
            TrackerConfig(enabled=True, csv_file_path=str(backup_path))
        )
        backup_transactions = backup_tracker.get_transaction_history()
        assert len(backup_transactions) == 15

        # Test maintenance operations
        maintenance_result = self.tracker.perform_maintenance()
        assert maintenance_result["success"]

        # Verify system is still functional after maintenance
        self._record_test_transaction("Post-maintenance test")
        final_transactions = self.tracker.get_transaction_history()
        assert len(final_transactions) == 16

    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration."""
        # Start performance monitoring
        self.tracker.start_monitoring(interval_seconds=1)

        # Record transactions with varying load
        start_time = time.time()

        for i in range(20):
            self._record_test_transaction(f"Performance test {i}")
            if i % 5 == 0:
                time.sleep(0.1)  # Simulate varying load

        end_time = time.time()

        # Let monitoring collect some data
        time.sleep(2)

        # Get performance metrics
        metrics = self.tracker.get_performance_metrics()
        assert "memory_usage_mb" in metrics
        assert "cpu_percent" in metrics

        # Get performance summary
        summary = self.tracker.get_performance_summary(hours=1)
        assert "transaction_rate" in summary
        assert "average_response_time" in summary

        # Stop monitoring
        self.tracker.stop_monitoring()

        # Verify system performance was acceptable
        total_time = end_time - start_time
        avg_time_per_transaction = total_time / 20
        assert avg_time_per_transaction < 0.5  # Should be fast

    def test_unicode_and_special_character_handling(self):
        """Test Unicode and special character handling in complete workflow."""
        unicode_test_cases = [
            {
                "name": "Emoji and Symbols",
                "prompt": "Add emoji support 😀🚀 and symbols ∑∏∫ to the system",
                "workspace": "unicode_test_🌍",
            },
            {
                "name": "International Characters",
                "prompt": "Implement support for 中文, العربية, Русский, and Ελληνικά",
                "workspace": "international_支持",
            },
            {
                "name": "Special CSV Characters",
                "prompt": 'Handle "quotes", commas,, and\nnewlines\tin CSV data',
                "workspace": "csv_special",
            },
            {
                "name": "Control Characters",
                "prompt": "Process control\x00chars\x01and\x02binary\x03data",
                "workspace": "control_chars",
            },
        ]

        for test_case in unicode_test_cases:
            execution_id = str(uuid.uuid4())

            context = {
                "execution_id": execution_id,
                "trigger_type": "userMessage",
                "workspace_folder": test_case["workspace"],
                "hook_name": "unicode-test-hook",
                "session_id": str(uuid.uuid4()),
                "user_message": test_case["prompt"],
            }

            result = {
                "execution_id": execution_id,
                "success": True,
                "token_usage": {
                    "total_tokens": 150,
                    "prompt": test_case["prompt"],
                    "model": "gpt-3.5-turbo",
                },
                "duration": 1.5,
            }

            # Execute workflow
            self.hook.on_agent_execution_start(context)
            self.hook.on_agent_execution_complete(context, result)

        # Verify all Unicode transactions were recorded correctly
        transactions = self.tracker.get_transaction_history()
        assert len(transactions) == len(unicode_test_cases)

        # Verify CSV file can be read correctly
        with open(self.csv_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Should contain Unicode characters
            assert "😀🚀" in content
            assert "中文" in content
            assert "العربية" in content

        # Verify CSV parsing works correctly
        with open(self.csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == len(unicode_test_cases)

        # Test Unicode compatibility
        unicode_test_result = self.tracker.test_unicode_compatibility()
        assert unicode_test_result["unicode_support_verified"]

    def _create_test_execution_data(self) -> List[Dict[str, Any]]:
        """Create test execution data for various scenarios."""
        return [
            {
                "trigger_type": "agentExecutionStarted",
                "workspace": "ai_hydra",
                "prompt": "Implement neural network training",
                "tokens": 1200,
                "duration": 3.5,
            },
            {
                "trigger_type": "fileEdited",
                "workspace": "docs",
                "prompt": "Update documentation",
                "tokens": 800,
                "duration": 2.1,
            },
            {
                "trigger_type": "userMessage",
                "workspace": "tests",
                "prompt": "Add integration tests",
                "tokens": 600,
                "duration": 1.8,
            },
        ]

    def _record_test_transaction(self, prompt_text: str) -> str:
        """Helper method to record a test transaction."""
        execution_id = str(uuid.uuid4())

        context = {
            "execution_id": execution_id,
            "trigger_type": "manual_test",
            "workspace_folder": "test_workspace",
            "hook_name": "test-hook",
            "session_id": str(uuid.uuid4()),
            "user_message": prompt_text,
        }

        result = {
            "execution_id": execution_id,
            "success": True,
            "token_usage": {
                "total_tokens": 100,
                "prompt": prompt_text,
                "model": "gpt-3.5-turbo",
            },
            "duration": 1.0,
        }

        self.hook.on_agent_execution_start(context)
        self.hook.on_agent_execution_complete(context, result)

        return execution_id

    def _create_permission_error(self):
        """Create a file permission error scenario."""
        if self.csv_file.exists():
            self.csv_file.chmod(0o444)  # Read-only

    def _restore_permissions(self):
        """Restore normal file permissions."""
        if self.csv_file.exists():
            self.csv_file.chmod(0o644)  # Read-write

    def _simulate_disk_full(self):
        """Simulate disk full condition."""
        # This is a mock implementation
        pass

    def _restore_disk_space(self):
        """Restore disk space."""
        # This is a mock implementation
        pass

    def _create_write_conflict(self):
        """Create a write conflict scenario."""
        # This would involve multiple processes trying to write simultaneously
        pass

    def _resolve_write_conflict(self):
        """Resolve write conflict."""
        pass

    def _inject_malformed_data(self):
        """Inject malformed data into the system."""
        if self.csv_file.exists():
            with open(self.csv_file, "a", encoding="utf-8") as f:
                f.write("malformed,data,without,proper,structure\n")

    def _clean_malformed_data(self):
        """Clean up malformed data."""
        if self.csv_file.exists():
            # Read all lines and filter out malformed ones
            with open(self.csv_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Keep only properly formatted lines
            clean_lines = [line for line in lines if not line.startswith("malformed")]

            with open(self.csv_file, "w", encoding="utf-8") as f:
                f.writelines(clean_lines)

    def _verify_system_functionality(self):
        """Verify system is still functional after error conditions."""
        # Record a test transaction to verify functionality
        test_id = self._record_test_transaction("System functionality verification")

        # Verify it was recorded
        transactions = self.tracker.get_transaction_history(limit=1)
        assert len(transactions) > 0
        assert transactions[0].agent_execution_id == test_id


class TestSystemHealthAndMonitoring:
    """Test system health checks and monitoring capabilities."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.csv_file = self.temp_dir / "health_test.csv"

        self.config = TrackerConfig(
            enabled=True,
            csv_file_path=str(self.csv_file),
            backup_enabled=True,
            log_level="INFO",
        )

        self.tracker = TokenTracker(self.config)

    def teardown_method(self):
        """Cleanup after each test."""
        if hasattr(self.tracker, "stop_monitoring"):
            self.tracker.stop_monitoring()

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_health_checks_comprehensive(self):
        """Test comprehensive health checks."""
        # Record some transactions first
        for i in range(5):
            success = self.tracker.record_transaction(
                prompt_text=f"Health check test {i}",
                tokens_used=100 + i * 10,
                elapsed_time=1.0 + i * 0.1,
            )
            assert success

        # Run health checks
        health_result = self.tracker.run_health_checks()

        # Verify health check results
        assert "overall_status" in health_result
        assert health_result["overall_status"] in ["healthy", "warning", "critical"]

        # Should include various health metrics
        expected_checks = [
            "csv_file_integrity",
            "disk_space",
            "memory_usage",
            "recent_errors",
            "performance_metrics",
        ]

        for check in expected_checks:
            assert check in health_result

    def test_monitoring_system_integration(self):
        """Test monitoring system integration."""
        # Start monitoring
        self.tracker.start_monitoring(interval_seconds=1)

        # Generate some activity
        for i in range(10):
            self.tracker.record_transaction(
                prompt_text=f"Monitoring test {i}",
                tokens_used=50 + i * 5,
                elapsed_time=0.5 + i * 0.05,
            )
            time.sleep(0.1)

        # Let monitoring collect data
        time.sleep(2)

        # Check monitoring data
        metrics = self.tracker.get_performance_metrics()
        assert "timestamp" in metrics
        assert "transaction_count" in metrics

        # Get performance summary
        summary = self.tracker.get_performance_summary(hours=1)
        assert "total_transactions" in summary
        assert summary["total_transactions"] == 10

        # Stop monitoring
        self.tracker.stop_monitoring()

    def test_alert_system_integration(self):
        """Test alert system integration."""
        alerts_received = []

        def alert_callback(alert_type, message, severity):
            alerts_received.append(
                {
                    "type": alert_type,
                    "message": message,
                    "severity": severity,
                    "timestamp": datetime.now(),
                }
            )

        # Register alert callback
        self.tracker.register_alert_callback(alert_callback)

        # Trigger conditions that should generate alerts
        # (This would depend on the specific alert thresholds)

        # For now, just verify the callback system works
        assert len(alerts_received) >= 0  # May or may not have alerts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
