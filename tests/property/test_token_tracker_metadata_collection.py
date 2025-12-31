"""
Property-based tests for Token Tracker metadata collection completeness.
"""

import pytest
import tempfile
import os
import platform
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, Any, Optional, List

from ai_hydra.token_tracker.metadata_collector import MetadataCollector
from ai_hydra.token_tracker.error_handler import TokenTrackerErrorHandler
from ai_hydra.token_tracker.tracker import TokenTracker
from ai_hydra.token_tracker.models import TrackerConfig


class TestTokenTrackerMetadataCollection:
    """Property-based tests for metadata collection completeness."""

    @given(
        context_data=st.one_of(
            st.none(),
            st.fixed_dictionaries(
                {
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
                    "hook_name": st.text(
                        min_size=1,
                        max_size=50,
                        alphabet=st.characters(
                            whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")
                        ),
                    ).filter(lambda x: x.strip() != ""),
                    "file_patterns": st.one_of(
                        st.none(),
                        st.lists(
                            st.text(
                                min_size=1,
                                max_size=20,
                                alphabet=st.characters(
                                    whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")
                                ),
                            ).filter(lambda x: ";" not in x),
                            min_size=0,
                            max_size=5,
                        ),
                    ),
                    "event": st.text(
                        min_size=1,
                        max_size=30,
                        alphabet=st.characters(
                            whitelist_categories=("Lu", "Ll", "Nd", "Pc")
                        ),
                    ).filter(lambda x: x.strip() != ""),
                    "config": st.dictionaries(
                        st.text(
                            min_size=1,
                            max_size=20,
                            alphabet=st.characters(
                                whitelist_categories=("Lu", "Ll", "Nd")
                            ),
                        ),
                        st.one_of(
                            st.text(min_size=0, max_size=50),
                            st.integers(min_value=0, max_value=1000),
                            st.booleans(),
                        ),
                        min_size=0,
                        max_size=5,
                    ),
                    "modified_files": st.lists(
                        st.text(
                            min_size=1,
                            max_size=100,
                            alphabet=st.characters(
                                whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")
                            ),
                        ).filter(
                            lambda x: x.strip() != "" and "/" not in x and "\\" not in x
                        ),
                        min_size=0,
                        max_size=10,
                    ),
                    "current_file": st.one_of(
                        st.none(),
                        st.text(
                            min_size=1,
                            max_size=100,
                            alphabet=st.characters(
                                whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")
                            ),
                        ).filter(
                            lambda x: x.strip() != "" and "/" not in x and "\\" not in x
                        ),
                    ),
                    "file_extension": st.one_of(
                        st.none(),
                        st.sampled_from(
                            [
                                ".py",
                                ".js",
                                ".ts",
                                ".md",
                                ".txt",
                                ".json",
                                ".yaml",
                                ".toml",
                            ]
                        ),
                    ),
                }
            ),
        )
    )
    @settings(max_examples=100, deadline=3000)
    def test_metadata_capture_completeness_property(self, context_data):
        """
        **Feature: kiro-token-tracker, Property 4: Hook-Tracker Integration (metadata portion)**
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**

        For any valid hook context, the metadata collector should capture all
        available metadata fields and handle missing or unavailable metadata gracefully.
        """
        # Create metadata collector
        error_handler = TokenTrackerErrorHandler()
        collector = MetadataCollector(error_handler)

        try:
            # Collect execution metadata
            metadata = collector.collect_execution_metadata(context_data)

            # Verify metadata is a dictionary
            assert isinstance(metadata, dict), "Metadata should be a dictionary"
            assert len(metadata) > 0, "Metadata should not be empty"

            # Verify required fields are always present (Requirements 8.1, 8.2, 8.3, 8.4, 8.5)
            required_fields = [
                "timestamp",
                "session_id",
                "agent_execution_id",
                "workspace_folder",
                "hook_trigger_type",
                "hook_name",
            ]

            for field in required_fields:
                assert (
                    field in metadata
                ), f"Required field '{field}' missing from metadata"
                assert (
                    metadata[field] is not None
                ), f"Required field '{field}' should not be None"
                assert isinstance(
                    metadata[field], str
                ), f"Field '{field}' should be a string"
                assert len(metadata[field]) > 0, f"Field '{field}' should not be empty"

            # Verify workspace folder name is captured (Requirement 8.1)
            workspace_folder = metadata["workspace_folder"]
            assert isinstance(
                workspace_folder, str
            ), "Workspace folder should be string"
            assert len(workspace_folder) > 0, "Workspace folder should not be empty"

            # Verify hook trigger type is captured (Requirement 8.2)
            hook_trigger_type = metadata["hook_trigger_type"]
            assert isinstance(
                hook_trigger_type, str
            ), "Hook trigger type should be string"
            # Should be either from context or default "unknown"
            if context_data and "trigger_type" in context_data:
                assert (
                    hook_trigger_type == context_data["trigger_type"]
                ), "Should use context trigger type"
            else:
                assert (
                    hook_trigger_type == "unknown"
                ), "Should default to 'unknown' when no context"

            # Verify agent execution ID is captured (Requirement 8.3)
            agent_execution_id = metadata["agent_execution_id"]
            assert isinstance(
                agent_execution_id, str
            ), "Agent execution ID should be string"
            assert len(agent_execution_id) > 0, "Agent execution ID should not be empty"
            # Should be a valid UUID format
            import uuid

            try:
                uuid.UUID(agent_execution_id)
            except ValueError:
                pytest.fail(
                    f"Agent execution ID should be valid UUID: {agent_execution_id}"
                )

            # Verify file patterns are captured when applicable (Requirement 8.4)
            if (
                context_data
                and "file_patterns" in context_data
                and context_data["file_patterns"]
            ):
                assert (
                    "file_patterns" in metadata
                ), "File patterns should be captured when provided"
                assert (
                    metadata["file_patterns"] == context_data["file_patterns"]
                ), "File patterns should match context"
            else:
                # File patterns may be None or not present when not provided
                if "file_patterns" in metadata:
                    assert metadata["file_patterns"] is None or isinstance(
                        metadata["file_patterns"], list
                    )

            # Verify hook name is captured (Requirement 8.5)
            hook_name = metadata["hook_name"]
            assert isinstance(hook_name, str), "Hook name should be string"
            if context_data and "hook_name" in context_data:
                assert (
                    hook_name == context_data["hook_name"]
                ), "Should use context hook name"
            else:
                assert (
                    hook_name == "unknown"
                ), "Should default to 'unknown' when no context"

            # Verify session ID consistency
            session_id = metadata["session_id"]
            assert isinstance(session_id, str), "Session ID should be string"
            assert len(session_id) > 0, "Session ID should not be empty"

            # Verify timestamp format
            timestamp = metadata["timestamp"]
            assert isinstance(timestamp, str), "Timestamp should be string"
            # Should be valid ISO format
            from datetime import datetime

            try:
                datetime.fromisoformat(timestamp)
            except ValueError:
                pytest.fail(f"Timestamp should be valid ISO format: {timestamp}")

            # Verify additional metadata fields are present
            expected_additional_fields = [
                "current_directory",
                "workspace_type",
                "platform",
                "username",
            ]

            for field in expected_additional_fields:
                if field in metadata:
                    assert isinstance(
                        metadata[field], str
                    ), f"Field '{field}' should be string when present"

            # Verify numeric fields are properly typed
            numeric_fields = [
                "process_id",
                "thread_id",
                "memory_usage_mb",
                "cpu_percent",
            ]
            for field in numeric_fields:
                if field in metadata:
                    assert isinstance(
                        metadata[field], (int, float)
                    ), f"Field '{field}' should be numeric when present"
                    if field in ["process_id", "thread_id"]:
                        assert (
                            metadata[field] > 0
                        ), f"Field '{field}' should be positive"

            # Verify boolean fields are properly typed
            boolean_fields = ["is_git_repo"]
            for field in boolean_fields:
                if field in metadata:
                    assert isinstance(
                        metadata[field], bool
                    ), f"Field '{field}' should be boolean when present"

        except Exception as e:
            # Metadata collection should not raise exceptions - should handle errors gracefully
            pytest.fail(f"Metadata collection should not raise exceptions: {e}")

    @given(
        workspace_scenarios=st.sampled_from(
            [
                "python_project",
                "node_project",
                "rust_project",
                "go_project",
                "java_project",
                "kiro_project",
                "empty_directory",
                "git_repository",
                "non_git_directory",
            ]
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_workspace_info_collection_property(self, workspace_scenarios):
        """
        **Feature: kiro-token-tracker, Property 4: Hook-Tracker Integration (metadata portion)**
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**

        For any workspace type, the metadata collector should properly detect
        and capture workspace information including project type and structure.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir)

            # Set up workspace based on scenario
            if workspace_scenarios == "python_project":
                (workspace_path / "pyproject.toml").write_text(
                    """
[project]
name = "test-project"
version = "1.0.0"
description = "Test project"
"""
                )
                (workspace_path / "requirements.txt").write_text("pytest>=7.0.0\n")

            elif workspace_scenarios == "node_project":
                (workspace_path / "package.json").write_text(
                    """
{
  "name": "test-project",
  "version": "1.0.0",
  "description": "Test project"
}
"""
                )

            elif workspace_scenarios == "rust_project":
                (workspace_path / "Cargo.toml").write_text(
                    """
[package]
name = "test-project"
version = "1.0.0"
"""
                )

            elif workspace_scenarios == "go_project":
                (workspace_path / "go.mod").write_text(
                    """
module test-project

go 1.21
"""
                )

            elif workspace_scenarios == "java_project":
                (workspace_path / "pom.xml").write_text(
                    """
<?xml version="1.0" encoding="UTF-8"?>
<project>
    <groupId>com.test</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0.0</version>
</project>
"""
                )

            elif workspace_scenarios == "kiro_project":
                kiro_dir = workspace_path / ".kiro"
                kiro_dir.mkdir()
                (kiro_dir / "config.json").write_text('{"version": "1.0"}')

            elif workspace_scenarios == "git_repository":
                git_dir = workspace_path / ".git"
                git_dir.mkdir()
                (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
                (git_dir / "config").write_text(
                    """
[core]
    repositoryformatversion = 0
[remote "origin"]
    url = https://github.com/test/repo.git
"""
                )

            # Change to the test workspace
            original_cwd = os.getcwd()
            try:
                os.chdir(workspace_path)

                # Create metadata collector
                error_handler = TokenTrackerErrorHandler()
                collector = MetadataCollector(error_handler)

                # Collect workspace information
                workspace_info = collector.get_workspace_info()

                # Verify workspace info is collected
                assert isinstance(
                    workspace_info, dict
                ), "Workspace info should be a dictionary"
                assert len(workspace_info) > 0, "Workspace info should not be empty"

                # Verify required workspace fields
                required_fields = [
                    "workspace_folder",
                    "current_directory",
                    "workspace_type",
                ]
                for field in required_fields:
                    assert (
                        field in workspace_info
                    ), f"Required field '{field}' missing from workspace info"
                    assert (
                        workspace_info[field] is not None
                    ), f"Field '{field}' should not be None"
                    assert isinstance(
                        workspace_info[field], str
                    ), f"Field '{field}' should be string"

                # Verify workspace folder name
                workspace_folder = workspace_info["workspace_folder"]
                assert len(workspace_folder) > 0, "Workspace folder should not be empty"
                assert (
                    workspace_folder == workspace_path.name
                ), "Should match directory name"

                # Verify current directory
                current_directory = workspace_info["current_directory"]
                assert (
                    len(current_directory) > 0
                ), "Current directory should not be empty"
                assert (
                    str(workspace_path) in current_directory
                ), "Should contain workspace path"

                # Verify workspace type detection
                workspace_type = workspace_info["workspace_type"]
                if workspace_scenarios == "python_project":
                    assert (
                        workspace_type == "python"
                    ), f"Should detect Python project, got {workspace_type}"
                elif workspace_scenarios == "node_project":
                    assert (
                        workspace_type == "node"
                    ), f"Should detect Node project, got {workspace_type}"
                elif workspace_scenarios == "rust_project":
                    assert (
                        workspace_type == "rust"
                    ), f"Should detect Rust project, got {workspace_type}"
                elif workspace_scenarios == "go_project":
                    assert (
                        workspace_type == "go"
                    ), f"Should detect Go project, got {workspace_type}"
                elif workspace_scenarios == "java_project":
                    assert (
                        workspace_type == "java"
                    ), f"Should detect Java project, got {workspace_type}"
                elif workspace_scenarios == "kiro_project":
                    assert (
                        workspace_type == "kiro"
                    ), f"Should detect Kiro project, got {workspace_type}"
                else:
                    # For empty directory or other cases, should default to "unknown"
                    assert (
                        workspace_type == "unknown"
                    ), f"Should default to unknown, got {workspace_type}"

                # Verify project files are listed when present
                if "project_files" in workspace_info:
                    project_files = workspace_info["project_files"]
                    assert isinstance(
                        project_files, list
                    ), "Project files should be a list"

                    if workspace_scenarios == "python_project":
                        assert any(
                            "pyproject.toml" in f for f in project_files
                        ), "Should list pyproject.toml"
                    elif workspace_scenarios == "node_project":
                        assert any(
                            "package.json" in f for f in project_files
                        ), "Should list package.json"

                # Verify Git information when applicable
                if workspace_scenarios == "git_repository":
                    assert (
                        "is_git_repo" in workspace_info
                    ), "Should detect Git repository"
                    assert (
                        workspace_info["is_git_repo"] is True
                    ), "Should be marked as Git repo"
                    if "branch" in workspace_info:
                        assert isinstance(
                            workspace_info["branch"], str
                        ), "Branch should be string"
                        assert (
                            len(workspace_info["branch"]) > 0
                        ), "Branch should not be empty"

            finally:
                os.chdir(original_cwd)

    @given(
        system_environment=st.fixed_dictionaries(
            {
                "mock_platform": st.sampled_from(["Linux", "Darwin", "Windows"]),
                "mock_username": st.text(
                    min_size=1,
                    max_size=20,
                    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
                ).filter(lambda x: x.strip() != ""),
                "mock_hostname": st.text(
                    min_size=1,
                    max_size=30,
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Pd")
                    ),
                ).filter(lambda x: x.strip() != ""),
                "mock_python_version": st.text(
                    min_size=5,
                    max_size=10,
                    alphabet=st.characters(whitelist_categories=("Nd", "Po")),
                ).filter(lambda x: "." in x),
            }
        )
    )
    @settings(max_examples=30, deadline=3000)
    def test_system_info_collection_property(self, system_environment):
        """
        **Feature: kiro-token-tracker, Property 4: Hook-Tracker Integration (metadata portion)**
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**

        For any system environment, the metadata collector should capture
        system information reliably and handle variations gracefully.
        """
        # Create metadata collector
        error_handler = TokenTrackerErrorHandler()
        collector = MetadataCollector(error_handler)

        # Mock system information
        with (
            patch("platform.system", return_value=system_environment["mock_platform"]),
            patch("platform.node", return_value=system_environment["mock_hostname"]),
            patch(
                "platform.python_version",
                return_value=system_environment["mock_python_version"],
            ),
            patch.dict(
                os.environ, {"USER": system_environment["mock_username"]}, clear=False
            ),
        ):

            # Collect system information
            system_info = collector.get_system_info()

            # Verify system info is collected
            assert isinstance(system_info, dict), "System info should be a dictionary"
            assert len(system_info) > 0, "System info should not be empty"

            # Verify required system fields
            required_fields = ["platform", "hostname", "python_version", "username"]
            for field in required_fields:
                assert (
                    field in system_info
                ), f"Required field '{field}' missing from system info"
                assert (
                    system_info[field] is not None
                ), f"Field '{field}' should not be None"
                assert isinstance(
                    system_info[field], str
                ), f"Field '{field}' should be string"
                assert (
                    len(system_info[field]) > 0
                ), f"Field '{field}' should not be empty"

            # Verify mocked values are captured correctly
            assert system_info["platform"] == system_environment["mock_platform"]
            assert system_info["hostname"] == system_environment["mock_hostname"]
            assert (
                system_info["python_version"]
                == system_environment["mock_python_version"]
            )
            assert system_info["username"] == system_environment["mock_username"]

            # Verify additional fields are present and properly typed
            optional_fields = ["platform_version", "architecture"]
            for field in optional_fields:
                if field in system_info:
                    assert isinstance(
                        system_info[field], str
                    ), f"Field '{field}' should be string when present"
                    assert (
                        len(system_info[field]) > 0
                    ), f"Field '{field}' should not be empty when present"

    @given(
        performance_scenario=st.sampled_from(
            [
                "normal_load",
                "high_memory_usage",
                "high_cpu_usage",
                "many_threads",
                "low_resources",
            ]
        )
    )
    @settings(max_examples=20, deadline=4000)
    def test_performance_info_collection_property(self, performance_scenario):
        """
        **Feature: kiro-token-tracker, Property 4: Hook-Tracker Integration (metadata portion)**
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**

        For any system performance state, the metadata collector should capture
        performance metrics without causing performance issues itself.
        """
        # Create metadata collector
        error_handler = TokenTrackerErrorHandler()
        collector = MetadataCollector(error_handler)

        # Mock performance data based on scenario
        mock_memory_mb = 100.0
        mock_cpu_percent = 5.0
        mock_threads = 10
        mock_system_cpu = 15.0
        mock_system_memory = 60.0
        mock_disk_usage = 45.0

        if performance_scenario == "high_memory_usage":
            mock_memory_mb = 2048.0
            mock_system_memory = 85.0
        elif performance_scenario == "high_cpu_usage":
            mock_cpu_percent = 95.0
            mock_system_cpu = 90.0
        elif performance_scenario == "many_threads":
            mock_threads = 150
        elif performance_scenario == "low_resources":
            mock_memory_mb = 50.0
            mock_system_memory = 95.0
            mock_disk_usage = 90.0

        # Mock psutil for consistent testing
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = int(mock_memory_mb * 1024 * 1024)
        mock_process.cpu_percent.return_value = mock_cpu_percent
        mock_process.num_threads.return_value = mock_threads
        mock_process.create_time.return_value = 1640995200.0  # Fixed timestamp

        mock_virtual_memory = Mock()
        mock_virtual_memory.percent = mock_system_memory

        mock_disk_usage_result = Mock()
        mock_disk_usage_result.percent = mock_disk_usage

        with (
            patch("psutil.Process", return_value=mock_process),
            patch("psutil.cpu_percent", return_value=mock_system_cpu),
            patch("psutil.virtual_memory", return_value=mock_virtual_memory),
            patch("psutil.disk_usage", return_value=mock_disk_usage_result),
        ):

            # Collect performance information
            performance_info = collector.get_performance_info()

            # Verify performance info is collected
            assert isinstance(
                performance_info, dict
            ), "Performance info should be a dictionary"
            assert len(performance_info) > 0, "Performance info should not be empty"

            # Verify required performance fields
            required_fields = ["memory_usage_mb", "cpu_percent", "num_threads"]
            for field in required_fields:
                assert (
                    field in performance_info
                ), f"Required field '{field}' missing from performance info"
                assert (
                    performance_info[field] is not None
                ), f"Field '{field}' should not be None"
                assert isinstance(
                    performance_info[field], (int, float)
                ), f"Field '{field}' should be numeric"

            # Verify performance values are reasonable
            memory_usage = performance_info["memory_usage_mb"]
            assert memory_usage >= 0, "Memory usage should be non-negative"
            assert (
                memory_usage == mock_memory_mb
            ), f"Memory usage should match mock: {memory_usage} vs {mock_memory_mb}"

            cpu_percent = performance_info["cpu_percent"]
            assert cpu_percent >= 0, "CPU percent should be non-negative"
            assert cpu_percent <= 100, "CPU percent should not exceed 100%"
            assert (
                cpu_percent == mock_cpu_percent
            ), f"CPU percent should match mock: {cpu_percent} vs {mock_cpu_percent}"

            num_threads = performance_info["num_threads"]
            assert num_threads >= 1, "Thread count should be at least 1"
            assert (
                num_threads == mock_threads
            ), f"Thread count should match mock: {num_threads} vs {mock_threads}"

            # Verify system performance fields
            system_fields = [
                "system_cpu_percent",
                "system_memory_percent",
                "system_disk_usage_percent",
            ]
            for field in system_fields:
                if field in performance_info:
                    value = performance_info[field]
                    assert isinstance(
                        value, (int, float)
                    ), f"Field '{field}' should be numeric"
                    assert (
                        0 <= value <= 100
                    ), f"Field '{field}' should be percentage (0-100): {value}"

            # Verify process creation time
            if "process_create_time" in performance_info:
                create_time = performance_info["process_create_time"]
                assert isinstance(
                    create_time, (int, float)
                ), "Process create time should be numeric"
                assert create_time > 0, "Process create time should be positive"

    @given(
        error_conditions=st.sampled_from(
            [
                "workspace_access_error",
                "system_info_error",
                "performance_info_error",
                "git_info_error",
                "project_file_error",
                "multiple_errors",
            ]
        )
    )
    @settings(max_examples=30, deadline=4000)
    def test_metadata_error_handling_property(self, error_conditions):
        """
        **Feature: kiro-token-tracker, Property 4: Hook-Tracker Integration (metadata portion)**
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**

        For any error condition during metadata collection, the system should
        handle errors gracefully and provide fallback metadata without crashing.
        """
        # Create metadata collector
        error_handler = TokenTrackerErrorHandler()
        collector = MetadataCollector(error_handler)

        # Simulate different error conditions
        if error_conditions == "workspace_access_error":
            with patch("pathlib.Path.cwd", side_effect=OSError("Access denied")):
                metadata = collector.collect_execution_metadata()

        elif error_conditions == "system_info_error":
            with patch(
                "platform.system", side_effect=RuntimeError("System info unavailable")
            ):
                metadata = collector.collect_execution_metadata()

        elif error_conditions == "performance_info_error":
            with patch(
                "psutil.Process", side_effect=ImportError("psutil not available")
            ):
                metadata = collector.collect_execution_metadata()

        elif error_conditions == "git_info_error":
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch(
                    "pathlib.Path.read_text",
                    side_effect=PermissionError("Git access denied"),
                ),
            ):
                metadata = collector.collect_execution_metadata()

        elif error_conditions == "project_file_error":
            with patch("pathlib.Path.glob", side_effect=OSError("File system error")):
                metadata = collector.collect_execution_metadata()

        elif error_conditions == "multiple_errors":
            with (
                patch("pathlib.Path.cwd", side_effect=OSError("Access denied")),
                patch("platform.system", side_effect=RuntimeError("System error")),
                patch("psutil.Process", side_effect=ImportError("psutil error")),
            ):
                metadata = collector.collect_execution_metadata()
        else:
            # Normal case for comparison
            metadata = collector.collect_execution_metadata()

        # Verify metadata collection doesn't crash and provides fallback data
        assert isinstance(metadata, dict), "Should return dictionary even on errors"
        assert len(metadata) > 0, "Should provide some metadata even on errors"

        # Verify essential fields are always present (with fallback values)
        essential_fields = [
            "timestamp",
            "session_id",
            "agent_execution_id",
            "workspace_folder",
            "hook_trigger_type",
            "hook_name",
        ]
        for field in essential_fields:
            assert (
                field in metadata
            ), f"Essential field '{field}' should always be present"
            assert (
                metadata[field] is not None
            ), f"Essential field '{field}' should not be None"
            assert isinstance(
                metadata[field], str
            ), f"Essential field '{field}' should be string"
            assert (
                len(metadata[field]) > 0
            ), f"Essential field '{field}' should not be empty"

        # Verify fallback values are reasonable
        if error_conditions != "normal_case":
            # Some fields may have fallback values like "unknown"
            fallback_fields = ["workspace_folder", "hook_trigger_type", "hook_name"]
            for field in fallback_fields:
                if metadata[field] == "unknown":
                    # This is acceptable fallback behavior
                    pass
                else:
                    # Should still be valid data
                    assert (
                        len(metadata[field]) > 0
                    ), f"Field '{field}' should not be empty"

        # Verify timestamp is always valid
        timestamp = metadata["timestamp"]
        from datetime import datetime

        try:
            datetime.fromisoformat(timestamp)
        except ValueError:
            pytest.fail(
                f"Timestamp should be valid ISO format even on errors: {timestamp}"
            )

        # Verify session and execution IDs are always valid UUIDs
        import uuid

        for id_field in ["session_id", "agent_execution_id"]:
            try:
                uuid.UUID(metadata[id_field])
            except ValueError:
                pytest.fail(
                    f"Field '{id_field}' should be valid UUID even on errors: {metadata[id_field]}"
                )

    @given(
        integration_context=st.fixed_dictionaries(
            {
                "prompt_text": st.text(min_size=1, max_size=500).filter(
                    lambda x: x.strip() != "" and "\x00" not in x
                ),
                "tokens_used": st.integers(min_value=1, max_value=10000),
                "elapsed_time": st.floats(
                    min_value=0.001,
                    max_value=60.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                "hook_context": st.one_of(
                    st.none(),
                    st.fixed_dictionaries(
                        {
                            "trigger_type": st.sampled_from(
                                ["agentExecutionCompleted", "fileEdited", "userMessage"]
                            ),
                            "hook_name": st.text(min_size=1, max_size=30).filter(
                                lambda x: x.strip() != ""
                            ),
                            "file_patterns": st.one_of(
                                st.none(),
                                st.lists(
                                    st.text(min_size=1, max_size=20),
                                    min_size=0,
                                    max_size=3,
                                ),
                            ),
                        }
                    ),
                ),
            }
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_full_integration_metadata_property(self, integration_context):
        """
        **Feature: kiro-token-tracker, Property 4: Hook-Tracker Integration**
        **Validates: Requirements 2.1, 2.2, 2.3, 8.1, 8.2, 8.3, 8.4, 8.5**

        For any valid AI agent execution, the agent hook should automatically trigger
        token tracking and successfully record the transaction with complete metadata.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_integration.csv"

            # Create test configuration
            config = TrackerConfig(
                csv_file_path=csv_path,
                backup_enabled=False,
                enable_validation=True,
                file_lock_timeout_seconds=2.0,
            )

            # Create token tracker (which includes metadata collector)
            tracker = TokenTracker(config)

            try:
                # Record transaction with metadata collection
                result = tracker.record_transaction(
                    prompt_text=integration_context["prompt_text"],
                    tokens_used=integration_context["tokens_used"],
                    elapsed_time=integration_context["elapsed_time"],
                    context=integration_context["hook_context"],
                )

                # Verify transaction was recorded successfully
                assert result is True, "Transaction should be recorded successfully"

                # Verify metadata was collected and integrated
                transactions = tracker.get_transaction_history()
                assert len(transactions) == 1, "Should have one recorded transaction"

                transaction = transactions[0]

                # Verify core transaction data
                assert (
                    transaction.prompt_text is not None
                ), "Prompt text should be preserved"
                assert (
                    transaction.tokens_used == integration_context["tokens_used"]
                ), "Token count should match"
                assert (
                    transaction.elapsed_time == integration_context["elapsed_time"]
                ), "Elapsed time should match"

                # Verify metadata integration (Requirements 8.1, 8.2, 8.3, 8.4, 8.5)
                assert (
                    transaction.workspace_folder is not None
                ), "Workspace folder should be captured"
                assert (
                    len(transaction.workspace_folder) > 0
                ), "Workspace folder should not be empty"

                assert (
                    transaction.hook_trigger_type is not None
                ), "Hook trigger type should be captured"
                assert (
                    len(transaction.hook_trigger_type) > 0
                ), "Hook trigger type should not be empty"

                assert (
                    transaction.agent_execution_id is not None
                ), "Agent execution ID should be captured"
                assert (
                    len(transaction.agent_execution_id) > 0
                ), "Agent execution ID should not be empty"

                assert (
                    transaction.session_id is not None
                ), "Session ID should be captured"
                assert len(transaction.session_id) > 0, "Session ID should not be empty"

                assert transaction.hook_name is not None, "Hook name should be captured"
                assert len(transaction.hook_name) > 0, "Hook name should not be empty"

                # Verify context-specific metadata
                if integration_context["hook_context"]:
                    hook_context = integration_context["hook_context"]

                    if "trigger_type" in hook_context:
                        assert (
                            transaction.hook_trigger_type
                            == hook_context["trigger_type"]
                        ), "Should use context trigger type"

                    if "hook_name" in hook_context:
                        assert (
                            transaction.hook_name == hook_context["hook_name"]
                        ), "Should use context hook name"

                    if (
                        "file_patterns" in hook_context
                        and hook_context["file_patterns"]
                    ):
                        assert (
                            transaction.file_patterns == hook_context["file_patterns"]
                        ), "Should preserve file patterns"

                # Verify UUID format for IDs
                import uuid

                try:
                    uuid.UUID(transaction.session_id)
                    uuid.UUID(transaction.agent_execution_id)
                except ValueError:
                    pytest.fail(
                        "Session ID and Agent execution ID should be valid UUIDs"
                    )

                # Verify timestamp is recent and valid
                from datetime import datetime, timedelta

                time_diff = datetime.now() - transaction.timestamp
                assert time_diff < timedelta(seconds=10), "Timestamp should be recent"

                # Verify CSV integrity after metadata integration
                csv_integrity = tracker.validate_csv_integrity()
                assert csv_integrity["file_exists"], "CSV file should exist"
                assert csv_integrity["header_valid"], "CSV headers should be valid"
                assert csv_integrity["total_rows"] == 1, "Should have one row"
                assert csv_integrity["valid_rows"] == 1, "Row should be valid"

            finally:
                # Cleanup
                if csv_path.exists():
                    csv_path.unlink()
