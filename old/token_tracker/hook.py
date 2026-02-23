"""
Token Tracking Hook for Kiro IDE Integration.

This module provides the TokenTrackingHook class that integrates with the Kiro IDE
hook system to automatically track token usage during AI agent executions.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from .tracker import TokenTracker
from .metadata_collector import MetadataCollector
from .models import TrackerConfig
from .error_handler import TokenTrackerErrorHandler, HookExecutionError


class TokenTrackingHook:
    """
    Agent hook for automatic token tracking in Kiro IDE.

    This class implements the Kiro IDE hook interface to automatically capture
    token usage data from AI agent executions and store it using the TokenTracker.
    """

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the token tracking hook.

        Args:
            config: Tracker configuration (uses default if not provided)
            logger: Logger instance (creates default if not provided)
        """
        self.config = config or TrackerConfig.create_default()
        self.logger = logger or self._setup_logging()

        # Initialize components
        self.error_handler = TokenTrackerErrorHandler(self.logger)
        self.metadata_collector = MetadataCollector(self.error_handler, self.logger)
        self.tracker = TokenTracker(self.config, self.logger)

        # Hook state
        self.hook_name = "token-tracking-hook"
        self.is_enabled = self.config.enabled
        self.execution_context: Dict[str, Any] = {}

        # Performance tracking
        self.hook_statistics = {
            "executions_started": 0,
            "executions_completed": 0,
            "executions_failed": 0,
            "total_tokens_tracked": 0,
            "total_execution_time": 0.0,
            "last_execution_time": None,
        }

        self.logger.info(f"TokenTrackingHook initialized: enabled={self.is_enabled}")

    def on_agent_execution_start(self, context: Dict[str, Any]) -> None:
        """
        Triggered when AI agent execution begins.

        This method captures the start context and prepares for token tracking.

        Args:
            context: Execution context from Kiro IDE containing:
                - execution_id: Unique identifier for this execution
                - trigger_type: Type of event that triggered execution
                - workspace_folder: Current workspace folder name
                - file_patterns: File patterns that triggered the execution
                - hook_name: Name of the hook that triggered execution
                - timestamp: Execution start timestamp
                - user_message: User message if applicable
        """
        if not self.is_enabled:
            self.logger.debug("Token tracking is disabled, skipping execution start")
            return

        try:
            execution_id = context.get("execution_id", str(uuid.uuid4()))

            # Store execution context for later use
            self.execution_context[execution_id] = {
                "start_time": time.time(),
                "start_timestamp": datetime.now(),
                "trigger_type": context.get("trigger_type", "unknown"),
                "workspace_folder": context.get("workspace_folder", "unknown"),
                "file_patterns": context.get("file_patterns", []),
                "hook_name": context.get("hook_name", "unknown"),
                "user_message": context.get("user_message"),
                "session_id": context.get("session_id", str(uuid.uuid4())),
                "metadata": self.metadata_collector.collect_execution_metadata(context),
            }

            self.hook_statistics["executions_started"] += 1

            self.logger.debug(
                f"Agent execution started: {execution_id}, "
                f"trigger: {context.get('trigger_type', 'unknown')}"
            )

        except Exception as e:
            self.hook_statistics["executions_failed"] += 1
            recovery_result = self.error_handler.handle_hook_execution_error(
                "execution_start", e, context
            )

            if not recovery_result.success:
                self.logger.error(f"Failed to handle agent execution start: {e}")

    def on_agent_execution_complete(
        self, context: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """
        Triggered when AI agent execution completes.

        This method extracts token usage data and records the transaction.

        Args:
            context: Execution context from start
            result: Execution result containing:
                - execution_id: Unique identifier for this execution
                - success: Whether execution was successful
                - token_usage: Token usage information
                - output: Agent output/response
                - error: Error information if execution failed
                - duration: Execution duration in seconds
        """
        if not self.is_enabled:
            self.logger.debug("Token tracking is disabled, skipping execution complete")
            return

        try:
            execution_id = result.get("execution_id") or context.get("execution_id")
            if not execution_id:
                self.logger.warning("No execution_id found in context or result")
                return

            # Get stored execution context
            stored_context = self.execution_context.get(execution_id)
            if not stored_context:
                self.logger.warning(
                    f"No stored context found for execution {execution_id}"
                )
                # Create minimal context from available data
                stored_context = {
                    "start_time": time.time() - result.get("duration", 0),
                    "trigger_type": context.get("trigger_type", "unknown"),
                    "workspace_folder": context.get("workspace_folder", "unknown"),
                    "file_patterns": context.get("file_patterns", []),
                    "hook_name": context.get("hook_name", "unknown"),
                    "session_id": context.get("session_id", str(uuid.uuid4())),
                    "metadata": {},
                }

            # Extract token usage information
            token_usage = self.extract_token_usage(result)

            if token_usage["tokens_used"] > 0:
                # Calculate execution time
                end_time = time.time()
                execution_time = end_time - stored_context["start_time"]

                # Prepare context for tracker
                tracking_context = {
                    "trigger_type": stored_context["trigger_type"],
                    "hook_name": stored_context["hook_name"],
                    "file_patterns": stored_context["file_patterns"],
                    "session_id": stored_context["session_id"],
                    "agent_execution_id": execution_id,
                    "workspace_folder": stored_context["workspace_folder"],
                    "execution_success": result.get("success", True),
                    "execution_error": result.get("error"),
                }

                # Record the transaction
                success = self.tracker.record_transaction(
                    prompt_text=token_usage["prompt_text"],
                    tokens_used=token_usage["tokens_used"],
                    elapsed_time=execution_time,
                    context=tracking_context,
                )

                if success:
                    self.hook_statistics["executions_completed"] += 1
                    self.hook_statistics["total_tokens_tracked"] += token_usage[
                        "tokens_used"
                    ]
                    self.hook_statistics["total_execution_time"] += execution_time
                    self.hook_statistics["last_execution_time"] = datetime.now()

                    self.logger.debug(
                        f"Token transaction recorded: {token_usage['tokens_used']} tokens, "
                        f"{execution_time:.2f}s execution time"
                    )
                else:
                    self.hook_statistics["executions_failed"] += 1
                    self.logger.warning(
                        f"Failed to record token transaction for {execution_id}"
                    )
            else:
                self.logger.debug(f"No tokens used in execution {execution_id}")

            # Clean up stored context
            if execution_id in self.execution_context:
                del self.execution_context[execution_id]

        except Exception as e:
            self.hook_statistics["executions_failed"] += 1
            recovery_result = self.error_handler.handle_hook_execution_error(
                "execution_complete", e, {"context": context, "result": result}
            )

            if not recovery_result.success:
                self.logger.error(f"Failed to handle agent execution complete: {e}")

    def extract_token_usage(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract token usage information from execution result.

        This method parses the execution result to extract token usage data
        in a format suitable for the TokenTracker.

        Args:
            result: Execution result from Kiro IDE

        Returns:
            Dict[str, Any]: Token usage data containing:
                - tokens_used: Number of tokens consumed
                - prompt_text: The prompt text sent to AI
                - model_name: AI model used (if available)
                - response_text: AI response (if available)
        """
        try:
            # Initialize default values
            token_data = {
                "tokens_used": 0,
                "prompt_text": "",
                "model_name": "unknown",
                "response_text": "",
            }

            # Extract token usage from various possible locations in result
            token_usage = result.get("token_usage", {})

            if isinstance(token_usage, dict):
                # Standard token usage format
                token_data["tokens_used"] = token_usage.get("total_tokens", 0)
                token_data["prompt_text"] = token_usage.get("prompt", "")
                token_data["model_name"] = token_usage.get("model", "unknown")
                token_data["response_text"] = token_usage.get("response", "")
            elif isinstance(token_usage, (int, float)):
                # Simple token count
                token_data["tokens_used"] = int(token_usage)

            # Try alternative locations for token data
            if token_data["tokens_used"] == 0:
                # Check for tokens in other common locations
                token_data["tokens_used"] = (
                    result.get("tokens", 0)
                    or result.get("total_tokens", 0)
                    or result.get("usage", {}).get("total_tokens", 0)
                )

            # Extract prompt text from various locations
            if not token_data["prompt_text"]:
                token_data["prompt_text"] = (
                    result.get("prompt", "")
                    or result.get("input", "")
                    or result.get("user_message", "")
                    or result.get("request", "")
                    or "No prompt text available"
                )

            # Extract response text
            if not token_data["response_text"]:
                token_data["response_text"] = (
                    result.get("output", "")
                    or result.get("response", "")
                    or result.get("completion", "")
                    or ""
                )

            # Validate and sanitize extracted data
            token_data["tokens_used"] = max(0, int(token_data["tokens_used"]))
            token_data["prompt_text"] = str(token_data["prompt_text"])[
                : self.config.max_prompt_length
            ]

            self.logger.debug(
                f"Extracted token usage: {token_data['tokens_used']} tokens"
            )

            return token_data

        except Exception as e:
            self.logger.error(f"Failed to extract token usage: {e}")
            return {
                "tokens_used": 0,
                "prompt_text": "Error extracting prompt",
                "model_name": "unknown",
                "response_text": "",
            }

    def on_file_edited(self, context: Dict[str, Any]) -> None:
        """
        Triggered when a file is edited (if configured to track file edits).

        Args:
            context: File edit context containing file path, changes, etc.
        """
        if not self.is_enabled:
            return

        try:
            # This could be used to track file-based AI interactions
            # For now, we'll just log the event
            file_path = context.get("file_path", "unknown")
            self.logger.debug(f"File edited: {file_path}")

            # Could potentially trigger token tracking for file-based AI operations
            # This would depend on the specific Kiro IDE integration requirements

        except Exception as e:
            self.logger.error(f"Failed to handle file edit event: {e}")

    def on_user_message(self, context: Dict[str, Any]) -> None:
        """
        Triggered when user sends a message (if configured to track user messages).

        Args:
            context: User message context
        """
        if not self.is_enabled:
            return

        try:
            message = context.get("message", "")
            self.logger.debug(f"User message received: {len(message)} characters")

            # This could be used to track direct user-AI interactions
            # Implementation would depend on Kiro IDE message handling

        except Exception as e:
            self.logger.error(f"Failed to handle user message event: {e}")

    def enable(self) -> None:
        """Enable the token tracking hook."""
        self.is_enabled = True
        self.config = TrackerConfig(
            enabled=True,
            csv_file_path=self.config.csv_file_path,
            max_prompt_length=self.config.max_prompt_length,
            backup_enabled=self.config.backup_enabled,
            backup_interval_hours=self.config.backup_interval_hours,
            compression_enabled=self.config.compression_enabled,
            retention_days=self.config.retention_days,
            auto_create_directories=self.config.auto_create_directories,
            file_lock_timeout_seconds=self.config.file_lock_timeout_seconds,
            max_concurrent_writes=self.config.max_concurrent_writes,
            enable_validation=self.config.enable_validation,
            log_level=self.config.log_level,
        )
        # Update tracker configuration
        self.tracker.config = self.config
        self.logger.info("Token tracking hook enabled")

    def disable(self) -> None:
        """Disable the token tracking hook."""
        self.is_enabled = False
        self.config = TrackerConfig(
            enabled=False,
            csv_file_path=self.config.csv_file_path,
            max_prompt_length=self.config.max_prompt_length,
            backup_enabled=self.config.backup_enabled,
            backup_interval_hours=self.config.backup_interval_hours,
            compression_enabled=self.config.compression_enabled,
            retention_days=self.config.retention_days,
            auto_create_directories=self.config.auto_create_directories,
            file_lock_timeout_seconds=self.config.file_lock_timeout_seconds,
            max_concurrent_writes=self.config.max_concurrent_writes,
            enable_validation=self.config.enable_validation,
            log_level=self.config.log_level,
        )
        # Update tracker configuration
        self.tracker.config = self.config
        self.logger.info("Token tracking hook disabled")

    def update_configuration(self, new_config: TrackerConfig) -> bool:
        """
        Update the hook configuration at runtime.

        Args:
            new_config: New configuration to apply

        Returns:
            bool: True if configuration was updated successfully, False otherwise
        """
        try:
            # Validate new configuration
            config_issues = new_config.validate()
            if config_issues:
                self.logger.error(f"Configuration validation failed: {config_issues}")
                return False

            # Store old configuration for rollback
            old_config = self.config
            old_enabled = self.is_enabled

            try:
                # Apply new configuration
                self.config = new_config
                self.is_enabled = new_config.enabled

                # Update tracker configuration
                self.tracker.config = new_config

                # Update metadata collector if needed
                if hasattr(self.metadata_collector, "config"):
                    self.metadata_collector.config = new_config

                # Update error handler if needed
                if hasattr(self.error_handler, "config"):
                    self.error_handler.config = new_config

                self.logger.info(
                    f"Configuration updated successfully: enabled={self.is_enabled}"
                )
                return True

            except Exception as update_error:
                # Rollback on failure
                self.config = old_config
                self.is_enabled = old_enabled
                self.tracker.config = old_config

                self.logger.error(
                    f"Configuration update failed, rolled back: {update_error}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current hook configuration.

        Returns:
            Dict[str, Any]: Current configuration as dictionary
        """
        config_dict = self.config.to_dict()
        config_dict["hook_enabled"] = self.is_enabled
        config_dict["hook_name"] = self.hook_name
        return config_dict

    def reload_configuration_from_file(
        self, config_file_path: Optional[Path] = None
    ) -> bool:
        """
        Reload configuration from a file.

        Args:
            config_file_path: Path to configuration file (uses default if not provided)

        Returns:
            bool: True if configuration was reloaded successfully, False otherwise
        """
        try:
            if config_file_path is None:
                # Use default configuration file path
                config_file_path = Path(".kiro/token_tracker_config.json")

            if not config_file_path.exists():
                self.logger.warning(f"Configuration file not found: {config_file_path}")
                return False

            # Load configuration from file
            import json

            with open(config_file_path, "r") as f:
                config_data = json.load(f)

            # Extract hook-specific fields before creating TrackerConfig
            hook_enabled = config_data.pop("hook_enabled", True)
            hook_name = config_data.pop("hook_name", self.hook_name)

            # Create new configuration from remaining data
            new_config = TrackerConfig.from_dict(config_data)

            # Update configuration
            success = self.update_configuration(new_config)

            # Update hook-specific settings if configuration update succeeded
            if success:
                self.is_enabled = hook_enabled
                self.hook_name = hook_name
                self.logger.info(
                    f"Hook configuration reloaded: enabled={self.is_enabled}"
                )

            return success

        except Exception as e:
            self.logger.error(f"Failed to reload configuration from file: {e}")
            return False

    def save_configuration_to_file(
        self, config_file_path: Optional[Path] = None
    ) -> bool:
        """
        Save current configuration to a file.

        Args:
            config_file_path: Path to save configuration file (uses default if not provided)

        Returns:
            bool: True if configuration was saved successfully, False otherwise
        """
        try:
            if config_file_path is None:
                # Use default configuration file path
                config_file_path = Path(".kiro/token_tracker_config.json")

            # Ensure directory exists
            config_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save configuration to file
            import json

            config_data = self.get_configuration()

            with open(config_file_path, "w") as f:
                json.dump(config_data, f, indent=2, default=str)

            self.logger.info(f"Configuration saved to: {config_file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save configuration to file: {e}")
            return False

    def reset_to_default_configuration(self) -> bool:
        """
        Reset hook configuration to default values.

        Returns:
            bool: True if configuration was reset successfully, False otherwise
        """
        try:
            default_config = TrackerConfig.create_default()
            return self.update_configuration(default_config)
        except Exception as e:
            self.logger.error(f"Failed to reset to default configuration: {e}")
            return False

    def apply_configuration_changes(self, changes: Dict[str, Any]) -> bool:
        """
        Apply partial configuration changes.

        Args:
            changes: Dictionary of configuration changes to apply

        Returns:
            bool: True if changes were applied successfully, False otherwise
        """
        try:
            # Get current configuration as dictionary
            current_config = self.config.to_dict()

            # Apply changes
            for key, value in changes.items():
                if key in current_config:
                    current_config[key] = value
                else:
                    self.logger.warning(f"Unknown configuration key: {key}")

            # Create new configuration
            new_config = TrackerConfig.from_dict(current_config)

            # Update configuration
            return self.update_configuration(new_config)

        except Exception as e:
            self.logger.error(f"Failed to apply configuration changes: {e}")
            return False

    def get_configuration_schema(self) -> Dict[str, Any]:
        """
        Get the configuration schema for validation.

        Returns:
            Dict[str, Any]: Configuration schema
        """
        return {
            "enabled": {
                "type": "boolean",
                "description": "Enable or disable token tracking",
                "default": True,
            },
            "csv_file_path": {
                "type": "string",
                "description": "Path to CSV file for storing transactions",
                "default": ".kiro/token_transactions.csv",
            },
            "max_prompt_length": {
                "type": "integer",
                "description": "Maximum length of prompt text to store",
                "minimum": 10,
                "maximum": 10000,
                "default": 1000,
            },
            "backup_enabled": {
                "type": "boolean",
                "description": "Enable automatic backups",
                "default": True,
            },
            "backup_interval_hours": {
                "type": "integer",
                "description": "Hours between automatic backups",
                "minimum": 1,
                "maximum": 168,
                "default": 24,
            },
            "compression_enabled": {
                "type": "boolean",
                "description": "Enable compression for archived files",
                "default": False,
            },
            "retention_days": {
                "type": "integer",
                "description": "Days to retain transaction data",
                "minimum": 1,
                "maximum": 3650,
                "default": 365,
            },
            "auto_create_directories": {
                "type": "boolean",
                "description": "Automatically create directories if they don't exist",
                "default": True,
            },
            "file_lock_timeout_seconds": {
                "type": "number",
                "description": "Timeout for file locking operations",
                "minimum": 0.1,
                "maximum": 60.0,
                "default": 5.0,
            },
            "max_concurrent_writes": {
                "type": "integer",
                "description": "Maximum number of concurrent write operations",
                "minimum": 1,
                "maximum": 100,
                "default": 10,
            },
            "enable_validation": {
                "type": "boolean",
                "description": "Enable data validation before writing",
                "default": True,
            },
            "log_level": {
                "type": "string",
                "description": "Logging level",
                "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "default": "INFO",
            },
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get hook execution statistics.

        Returns:
            Dict[str, Any]: Hook statistics
        """
        stats = self.hook_statistics.copy()

        # Add derived statistics
        if stats["executions_completed"] > 0:
            stats["average_tokens_per_execution"] = (
                stats["total_tokens_tracked"] / stats["executions_completed"]
            )
            stats["average_execution_time"] = (
                stats["total_execution_time"] / stats["executions_completed"]
            )
        else:
            stats["average_tokens_per_execution"] = 0
            stats["average_execution_time"] = 0

        stats["success_rate"] = stats["executions_completed"] / max(
            1, stats["executions_started"]
        )

        # Add tracker statistics
        stats["tracker_statistics"] = self.tracker.get_statistics()

        return stats

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the hook configuration.

        Returns:
            Dict[str, Any]: Validation results
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": [],
        }

        try:
            # Validate tracker configuration
            config_issues = self.config.validate()
            if config_issues:
                validation_result["issues"].extend(config_issues)
                validation_result["valid"] = False

            # Validate tracker functionality
            tracker_integrity = self.tracker.validate_csv_integrity()
            if not tracker_integrity.get("file_exists", True):
                validation_result["warnings"].append("CSV file does not exist yet")

            # Check permissions
            csv_path = self.config.get_csv_file_path()
            if csv_path.exists():
                try:
                    with open(csv_path, "a") as f:
                        pass
                except PermissionError:
                    validation_result["issues"].append(
                        f"No write permission for {csv_path}"
                    )
                    validation_result["valid"] = False

        except Exception as e:
            validation_result["issues"].append(f"Configuration validation failed: {e}")
            validation_result["valid"] = False

        return validation_result

    def test_hook_functionality(self) -> Dict[str, Any]:
        """
        Test hook functionality with a mock execution.

        Returns:
            Dict[str, Any]: Test results
        """
        test_result = {
            "test_passed": False,
            "test_details": {},
            "errors": [],
        }

        try:
            # Create mock execution context
            test_execution_id = f"test_{uuid.uuid4()}"
            mock_context = {
                "execution_id": test_execution_id,
                "trigger_type": "manual_test",
                "workspace_folder": "test_workspace",
                "file_patterns": ["*.py"],
                "hook_name": "test_hook",
                "session_id": str(uuid.uuid4()),
            }

            # Test execution start
            self.on_agent_execution_start(mock_context)
            test_result["test_details"]["execution_start"] = "passed"

            # Create mock execution result
            mock_result = {
                "execution_id": test_execution_id,
                "success": True,
                "token_usage": {
                    "total_tokens": 150,
                    "prompt": "Test prompt for hook functionality",
                    "model": "test_model",
                    "response": "Test response",
                },
                "duration": 2.5,
            }

            # Test execution complete
            self.on_agent_execution_complete(mock_context, mock_result)
            test_result["test_details"]["execution_complete"] = "passed"

            # Verify transaction was recorded
            recent_transactions = self.tracker.get_transaction_history(limit=1)
            if recent_transactions and recent_transactions[0].tokens_used == 150:
                test_result["test_details"]["transaction_recorded"] = "passed"
                test_result["test_passed"] = True
            else:
                test_result["errors"].append("Transaction was not recorded correctly")

        except Exception as e:
            test_result["errors"].append(f"Hook test failed: {e}")
            self.logger.error(f"Hook functionality test failed: {e}")

        return test_result

    def cleanup(self) -> None:
        """Clean up hook resources."""
        try:
            # Clear any pending execution contexts
            self.execution_context.clear()

            # Cleanup tracker resources
            if hasattr(self.tracker, "cleanup"):
                self.tracker.cleanup()

            self.logger.info("TokenTrackingHook cleanup completed")

        except Exception as e:
            self.logger.error(f"Hook cleanup failed: {e}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the hook."""
        logger = logging.getLogger(f"{__name__}.TokenTrackingHook")

        # Set log level based on config
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)

        # Add handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
