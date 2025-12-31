"""
Property-based tests for token tracker configuration state management.

This module contains property-based tests that validate the configuration
management functionality of the TokenTrackingHook class.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
from hypothesis import given, strategies as st, settings, assume

from ai_hydra.token_tracker.hook import TokenTrackingHook
from ai_hydra.token_tracker.models import TrackerConfig


class TestConfigurationStateManagement:
    """
    Property-based tests for configuration state management.

    **Property 6: Configuration State Management**
    **Validates: Requirements 2.5**

    For any valid configuration changes, the hook should maintain
    consistent state and allow runtime configuration updates.
    """

    @given(
        enabled=st.booleans(),
        max_prompt_length=st.integers(min_value=10, max_value=5000),
        backup_enabled=st.booleans(),
        backup_interval_hours=st.integers(min_value=1, max_value=168),
        retention_days=st.integers(min_value=1, max_value=365),
        file_lock_timeout=st.floats(min_value=0.1, max_value=30.0),
        max_concurrent_writes=st.integers(min_value=1, max_value=50),
        log_level=st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    )
    @settings(max_examples=50, deadline=3000)
    def test_configuration_state_consistency_property(
        self,
        enabled: bool,
        max_prompt_length: int,
        backup_enabled: bool,
        backup_interval_hours: int,
        retention_days: int,
        file_lock_timeout: float,
        max_concurrent_writes: int,
        log_level: str,
    ):
        """
        **Property 6: Configuration State Management**
        **Validates: Requirements 2.5**

        For any valid configuration parameters, the hook should:
        1. Accept the configuration update
        2. Maintain consistent internal state
        3. Reflect changes in subsequent operations
        4. Preserve configuration across enable/disable cycles
        """
        # Create initial hook with default configuration
        hook = TokenTrackingHook()
        initial_enabled = hook.is_enabled

        try:
            # Create new configuration with test parameters
            new_config = TrackerConfig(
                enabled=enabled,
                max_prompt_length=max_prompt_length,
                backup_enabled=backup_enabled,
                backup_interval_hours=backup_interval_hours,
                retention_days=retention_days,
                file_lock_timeout_seconds=file_lock_timeout,
                max_concurrent_writes=max_concurrent_writes,
                log_level=log_level,
            )

            # Property 1: Configuration update should succeed for valid parameters
            update_success = hook.update_configuration(new_config)
            assert (
                update_success
            ), f"Configuration update failed for valid parameters: {new_config.to_dict()}"

            # Property 2: Hook state should reflect the new configuration
            current_config = hook.get_configuration()
            assert (
                current_config["enabled"] == enabled
            ), "Hook enabled state should match configuration"
            assert (
                current_config["max_prompt_length"] == max_prompt_length
            ), "Max prompt length should match"
            assert (
                current_config["backup_enabled"] == backup_enabled
            ), "Backup enabled should match"
            assert (
                current_config["backup_interval_hours"] == backup_interval_hours
            ), "Backup interval should match"
            assert (
                current_config["retention_days"] == retention_days
            ), "Retention days should match"
            assert (
                current_config["file_lock_timeout_seconds"] == file_lock_timeout
            ), "File lock timeout should match"
            assert (
                current_config["max_concurrent_writes"] == max_concurrent_writes
            ), "Max concurrent writes should match"
            assert current_config["log_level"] == log_level, "Log level should match"

            # Property 3: Hook enabled state should match configuration enabled state
            assert (
                hook.is_enabled == enabled
            ), "Hook is_enabled should match configuration enabled"

            # Property 4: Configuration should be consistent across multiple reads
            config_read_1 = hook.get_configuration()
            config_read_2 = hook.get_configuration()
            assert (
                config_read_1 == config_read_2
            ), "Configuration should be consistent across multiple reads"

            # Property 5: Enable/disable operations should preserve other configuration
            if enabled:
                hook.disable()
                assert (
                    not hook.is_enabled
                ), "Hook should be disabled after disable() call"
                disabled_config = hook.get_configuration()
                # All other settings should remain the same
                assert disabled_config["max_prompt_length"] == max_prompt_length
                assert disabled_config["backup_enabled"] == backup_enabled
                assert disabled_config["log_level"] == log_level

                hook.enable()
                assert hook.is_enabled, "Hook should be enabled after enable() call"
                enabled_config = hook.get_configuration()
                # All other settings should still be preserved
                assert enabled_config["max_prompt_length"] == max_prompt_length
                assert enabled_config["backup_enabled"] == backup_enabled
                assert enabled_config["log_level"] == log_level
            else:
                hook.enable()
                assert hook.is_enabled, "Hook should be enabled after enable() call"
                hook.disable()
                assert (
                    not hook.is_enabled
                ), "Hook should be disabled after disable() call"

            # Property 6: Configuration validation should work correctly
            validation_result = hook.validate_configuration()
            assert isinstance(
                validation_result, dict
            ), "Validation should return a dictionary"
            assert (
                "valid" in validation_result
            ), "Validation result should have 'valid' key"
            assert (
                "issues" in validation_result
            ), "Validation result should have 'issues' key"
            assert isinstance(
                validation_result["issues"], list
            ), "Issues should be a list"

        finally:
            # Cleanup
            hook.cleanup()

    @given(
        config_changes=st.dictionaries(
            keys=st.sampled_from(
                [
                    "enabled",
                    "max_prompt_length",
                    "backup_enabled",
                    "backup_interval_hours",
                    "retention_days",
                    "log_level",
                ]
            ),
            values=st.one_of(
                st.booleans(),
                st.integers(min_value=1, max_value=1000),
                st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR"]),
            ),
            min_size=1,
            max_size=4,
        )
    )
    @settings(max_examples=30, deadline=3000)
    def test_partial_configuration_changes_property(
        self, config_changes: Dict[str, Any]
    ):
        """
        **Property 6: Configuration State Management (Partial Updates)**
        **Validates: Requirements 2.5**

        For any partial configuration changes, the hook should:
        1. Apply only the specified changes
        2. Preserve unchanged configuration values
        3. Maintain system consistency
        """
        hook = TokenTrackingHook()

        try:
            # Get initial configuration
            initial_config = hook.get_configuration()

            # Filter out invalid combinations
            filtered_changes = {}
            for key, value in config_changes.items():
                if (
                    key == "max_prompt_length"
                    and isinstance(value, int)
                    and value >= 10
                ):
                    filtered_changes[key] = value
                elif (
                    key == "backup_interval_hours"
                    and isinstance(value, int)
                    and value >= 1
                ):
                    filtered_changes[key] = value
                elif key == "retention_days" and isinstance(value, int) and value >= 1:
                    filtered_changes[key] = value
                elif key in ["enabled", "backup_enabled"] and isinstance(value, bool):
                    filtered_changes[key] = value
                elif key == "log_level" and isinstance(value, str):
                    filtered_changes[key] = value

            # Skip if no valid changes
            assume(len(filtered_changes) > 0)

            # Apply partial changes
            success = hook.apply_configuration_changes(filtered_changes)
            assert (
                success
            ), f"Partial configuration changes should succeed: {filtered_changes}"

            # Verify changes were applied
            updated_config = hook.get_configuration()
            for key, expected_value in filtered_changes.items():
                assert (
                    updated_config[key] == expected_value
                ), f"Configuration key '{key}' should be updated to {expected_value}"

            # Verify unchanged values are preserved
            for key, initial_value in initial_config.items():
                if key not in filtered_changes and key not in [
                    "hook_enabled",
                    "hook_name",
                ]:
                    assert (
                        updated_config[key] == initial_value
                    ), f"Unchanged configuration key '{key}' should be preserved"

        finally:
            hook.cleanup()

    @given(
        config_data=st.fixed_dictionaries(
            {
                "enabled": st.booleans(),
                "max_prompt_length": st.integers(min_value=10, max_value=2000),
                "backup_enabled": st.booleans(),
                "backup_interval_hours": st.integers(min_value=1, max_value=72),
                "retention_days": st.integers(min_value=1, max_value=180),
                "log_level": st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR"]),
            }
        )
    )
    @settings(max_examples=20, deadline=4000)
    def test_configuration_file_persistence_property(self, config_data: Dict[str, Any]):
        """
        **Property 6: Configuration State Management (File Persistence)**
        **Validates: Requirements 2.5**

        For any valid configuration, the hook should:
        1. Save configuration to file correctly
        2. Reload configuration from file correctly
        3. Maintain configuration consistency across save/load cycles
        """
        hook = TokenTrackingHook()

        try:
            # Create temporary file for configuration
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as temp_file:
                config_file_path = Path(temp_file.name)

            try:
                # Apply configuration
                new_config = TrackerConfig(
                    csv_file_path=Path("test_transactions.csv"),  # Use test path
                    **config_data,
                )

                update_success = hook.update_configuration(new_config)
                assert update_success, "Configuration update should succeed"

                # Save configuration to file
                save_success = hook.save_configuration_to_file(config_file_path)
                assert save_success, "Configuration save should succeed"
                assert (
                    config_file_path.exists()
                ), "Configuration file should exist after save"

                # Verify file contents
                with open(config_file_path, "r") as f:
                    saved_data = json.load(f)

                # Check that key configuration values are saved
                for key, expected_value in config_data.items():
                    assert (
                        saved_data[key] == expected_value
                    ), f"Saved configuration should contain {key}={expected_value}"

                # Create new hook and reload configuration
                new_hook = TokenTrackingHook()
                reload_success = new_hook.reload_configuration_from_file(
                    config_file_path
                )
                assert reload_success, "Configuration reload should succeed"

                # Verify reloaded configuration matches original
                reloaded_config = new_hook.get_configuration()
                for key, expected_value in config_data.items():
                    assert (
                        reloaded_config[key] == expected_value
                    ), f"Reloaded configuration should contain {key}={expected_value}"

                # Verify hook state matches reloaded configuration
                assert (
                    new_hook.is_enabled == config_data["enabled"]
                ), "Hook enabled state should match reloaded configuration"

                new_hook.cleanup()

            finally:
                # Clean up temporary file
                if config_file_path.exists():
                    config_file_path.unlink()

        finally:
            hook.cleanup()

    @given(
        initial_enabled=st.booleans(),
        operations=st.lists(
            st.sampled_from(["enable", "disable", "reset", "validate"]),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=25, deadline=3000)
    def test_configuration_state_transitions_property(
        self, initial_enabled: bool, operations: list
    ):
        """
        **Property 6: Configuration State Management (State Transitions)**
        **Validates: Requirements 2.5**

        For any sequence of configuration operations, the hook should:
        1. Maintain consistent state throughout all transitions
        2. Handle enable/disable operations correctly
        3. Preserve configuration during state changes
        4. Support configuration reset operations
        """
        # Create initial configuration
        initial_config = TrackerConfig(enabled=initial_enabled)
        hook = TokenTrackingHook(config=initial_config)

        try:
            # Track expected state
            expected_enabled = initial_enabled

            for operation in operations:
                if operation == "enable":
                    hook.enable()
                    expected_enabled = True
                elif operation == "disable":
                    hook.disable()
                    expected_enabled = False
                elif operation == "reset":
                    reset_success = hook.reset_to_default_configuration()
                    assert reset_success, "Configuration reset should succeed"
                    expected_enabled = True  # Default configuration has enabled=True
                elif operation == "validate":
                    validation_result = hook.validate_configuration()
                    assert isinstance(
                        validation_result, dict
                    ), "Validation should return a dictionary"

                # Verify state consistency after each operation
                assert (
                    hook.is_enabled == expected_enabled
                ), f"Hook enabled state should be {expected_enabled} after {operation}"

                current_config = hook.get_configuration()
                assert (
                    current_config["enabled"] == expected_enabled
                ), f"Configuration enabled should be {expected_enabled} after {operation}"

                # Verify configuration is still valid
                validation_result = hook.validate_configuration()
                assert isinstance(
                    validation_result, dict
                ), "Configuration should remain valid after operations"

        finally:
            hook.cleanup()

    def test_configuration_error_handling_property(self):
        """
        **Property 6: Configuration State Management (Error Handling)**
        **Validates: Requirements 2.5**

        The hook should handle configuration errors gracefully and maintain
        consistent state even when invalid configurations are provided.
        """
        hook = TokenTrackingHook()

        try:
            # Get initial valid configuration
            initial_config = hook.get_configuration()

            # Test invalid configuration scenarios
            invalid_configs = [
                {"max_prompt_length": -1},  # Negative value
                {"backup_interval_hours": 0},  # Zero value
                {"retention_days": -5},  # Negative value
                {"file_lock_timeout_seconds": -1.0},  # Negative timeout
                {"max_concurrent_writes": 0},  # Zero concurrent writes
                {"log_level": "INVALID_LEVEL"},  # Invalid log level
            ]

            for invalid_change in invalid_configs:
                # Attempt to apply invalid configuration
                success = hook.apply_configuration_changes(invalid_change)

                # Configuration update should fail for invalid values
                assert (
                    not success
                ), f"Invalid configuration should be rejected: {invalid_change}"

                # Hook should maintain previous valid configuration
                current_config = hook.get_configuration()
                for key, value in initial_config.items():
                    if key not in invalid_change:
                        assert (
                            current_config[key] == value
                        ), f"Valid configuration should be preserved after invalid update attempt"

                # Hook should remain functional
                validation_result = hook.validate_configuration()
                assert isinstance(
                    validation_result, dict
                ), "Hook should remain functional after invalid configuration attempt"

        finally:
            hook.cleanup()
