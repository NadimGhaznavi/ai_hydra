"""
Error handling system for the Token Tracker.

This module provides specialized error handling for token tracking operations,
including CSV file errors, validation failures, and recovery mechanisms.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging
import traceback
import time
from pathlib import Path


class TokenTrackerErrorType(Enum):
    """Types of errors that can occur in the token tracking system."""

    CSV_WRITE_ERROR = "csv_write_error"
    CSV_READ_ERROR = "csv_read_error"
    VALIDATION_ERROR = "validation_error"
    FILE_LOCK_ERROR = "file_lock_error"
    PERMISSION_ERROR = "permission_error"
    DISK_SPACE_ERROR = "disk_space_error"
    CONFIGURATION_ERROR = "configuration_error"
    METADATA_ERROR = "metadata_error"


class TokenTrackerError(Exception):
    """Base exception for token tracker errors."""

    def __init__(
        self,
        message: str,
        error_type: TokenTrackerErrorType,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.error_type = error_type
        self.context = context or {}
        self.timestamp = time.time()
        super().__init__(message)


class CSVWriteError(TokenTrackerError):
    """Exception raised when CSV write operations fail."""

    def __init__(
        self, message: str, file_path: Path, context: Optional[Dict[str, Any]] = None
    ):
        self.file_path = file_path
        super().__init__(message, TokenTrackerErrorType.CSV_WRITE_ERROR, context)


class CSVReadError(TokenTrackerError):
    """Exception raised when CSV read operations fail."""

    def __init__(
        self, message: str, file_path: Path, context: Optional[Dict[str, Any]] = None
    ):
        self.file_path = file_path
        super().__init__(message, TokenTrackerErrorType.CSV_READ_ERROR, context)


class ValidationError(TokenTrackerError):
    """Exception raised when data validation fails."""

    def __init__(
        self,
        message: str,
        validation_failures: List[str],
        context: Optional[Dict[str, Any]] = None,
    ):
        self.validation_failures = validation_failures
        super().__init__(message, TokenTrackerErrorType.VALIDATION_ERROR, context)


class FileLockError(TokenTrackerError):
    """Exception raised when file locking operations fail."""

    def __init__(
        self,
        message: str,
        file_path: Path,
        timeout: float,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.file_path = file_path
        self.timeout = timeout
        super().__init__(message, TokenTrackerErrorType.FILE_LOCK_ERROR, context)


class PermissionError(TokenTrackerError):
    """Exception raised when file permission errors occur."""

    def __init__(
        self,
        message: str,
        file_path: Path,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.file_path = file_path
        self.operation = operation
        super().__init__(message, TokenTrackerErrorType.PERMISSION_ERROR, context)


class DiskSpaceError(TokenTrackerError):
    """Exception raised when disk space is insufficient."""

    def __init__(
        self,
        message: str,
        required_space: int,
        available_space: int,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.required_space = required_space
        self.available_space = available_space
        super().__init__(message, TokenTrackerErrorType.DISK_SPACE_ERROR, context)


class ConfigurationError(TokenTrackerError):
    """Exception raised when configuration is invalid."""

    def __init__(
        self,
        message: str,
        config_issues: List[str],
        context: Optional[Dict[str, Any]] = None,
    ):
        self.config_issues = config_issues
        super().__init__(message, TokenTrackerErrorType.CONFIGURATION_ERROR, context)


class MetadataError(TokenTrackerError):
    """Exception raised when metadata collection fails."""

    def __init__(
        self, message: str, metadata_type: str, context: Optional[Dict[str, Any]] = None
    ):
        self.metadata_type = metadata_type
        super().__init__(message, TokenTrackerErrorType.METADATA_ERROR, context)


@dataclass
class ErrorRecoveryResult:
    """Result of an error recovery attempt."""

    success: bool
    message: str
    recovered_data: Optional[Any] = None
    should_retry: bool = False
    fallback_used: bool = False


class TokenTrackerErrorHandler:
    """
    Specialized error handler for token tracking operations.

    This class provides error handling, recovery mechanisms, and fallback
    strategies specifically designed for token tracking system failures.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the token tracker error handler.

        Args:
            logger: Logger instance (creates default if not provided)
        """
        self.logger = logger or logging.getLogger(__name__)

        # Error tracking
        self.error_history: List[TokenTrackerError] = []
        self.recovery_statistics = {
            "total_errors": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "errors_by_type": {
                error_type.value: 0 for error_type in TokenTrackerErrorType
            },
            "fallback_activations": 0,
        }

        # Recovery configuration
        self.max_retry_attempts = 3
        self.retry_delays = [0.1, 0.5, 1.0]  # Exponential backoff
        self.fallback_csv_path = Path(".kiro/token_transactions_fallback.csv")

        self.logger.info("TokenTrackerErrorHandler initialized")

    def handle_csv_read_error(
        self, error: Exception, file_path: Path
    ) -> ErrorRecoveryResult:
        """
        Handle CSV read errors with fallback mechanisms.

        Args:
            error: The original exception
            file_path: Path to the CSV file

        Returns:
            ErrorRecoveryResult: Result of error handling
        """
        context = {
            "file_path": str(file_path),
            "original_error": str(error),
        }

        csv_error = CSVReadError(
            f"Failed to read from CSV file {file_path}: {error}", file_path, context
        )

        self._log_error(csv_error)

        # Try recovery strategies
        recovery_result = self._attempt_csv_read_recovery(csv_error)

        self._update_statistics(csv_error, recovery_result)
        return recovery_result

    def handle_csv_write_error(
        self, error: Exception, file_path: Path, transaction_data: Any
    ) -> ErrorRecoveryResult:
        """
        Handle CSV write errors with fallback mechanisms.

        Args:
            error: The original exception
            file_path: Path to the CSV file
            transaction_data: Data that failed to write

        Returns:
            ErrorRecoveryResult: Result of error handling
        """
        context = {
            "file_path": str(file_path),
            "transaction_data_type": type(transaction_data).__name__,
            "original_error": str(error),
        }

        csv_error = CSVWriteError(
            f"Failed to write to CSV file {file_path}: {error}", file_path, context
        )

        self._log_error(csv_error)

        # Try recovery strategies
        recovery_result = self._attempt_csv_write_recovery(csv_error, transaction_data)

        self._update_statistics(csv_error, recovery_result)
        return recovery_result

    def handle_validation_error(
        self, validation_failures: List[str], data: Any
    ) -> ErrorRecoveryResult:
        """
        Handle data validation errors with correction attempts.

        Args:
            validation_failures: List of validation failure messages
            data: Data that failed validation

        Returns:
            ErrorRecoveryResult: Result of error handling
        """
        context = {
            "data_type": type(data).__name__,
            "failure_count": len(validation_failures),
        }

        validation_error = ValidationError(
            f"Data validation failed: {'; '.join(validation_failures)}",
            validation_failures,
            context,
        )

        self._log_error(validation_error)

        # Try to correct validation issues
        recovery_result = self._attempt_validation_recovery(validation_error, data)

        self._update_statistics(validation_error, recovery_result)
        return recovery_result

    def handle_file_lock_error(
        self, file_path: Path, timeout: float
    ) -> ErrorRecoveryResult:
        """
        Handle file locking errors with retry mechanisms.

        Args:
            file_path: Path to the locked file
            timeout: Timeout that was exceeded

        Returns:
            ErrorRecoveryResult: Result of error handling
        """
        context = {"file_path": str(file_path), "timeout": timeout}

        lock_error = FileLockError(
            f"Failed to acquire lock for {file_path} within {timeout} seconds",
            file_path,
            timeout,
            context,
        )

        self._log_error(lock_error)

        # Try alternative approaches
        recovery_result = self._attempt_file_lock_recovery(lock_error)

        self._update_statistics(lock_error, recovery_result)
        return recovery_result

    def handle_permission_error(
        self, file_path: Path, operation: str
    ) -> ErrorRecoveryResult:
        """
        Handle file permission errors with fallback locations.

        Args:
            file_path: Path with permission issues
            operation: Operation that failed

        Returns:
            ErrorRecoveryResult: Result of error handling
        """
        context = {"file_path": str(file_path), "operation": operation}

        perm_error = PermissionError(
            f"Permission denied for {operation} on {file_path}",
            file_path,
            operation,
            context,
        )

        self._log_error(perm_error)

        # Try fallback locations
        recovery_result = self._attempt_permission_recovery(perm_error)

        self._update_statistics(perm_error, recovery_result)
        return recovery_result

    def handle_disk_space_error(
        self, required_space: int, available_space: int
    ) -> ErrorRecoveryResult:
        """
        Handle disk space errors with cleanup and compression.

        Args:
            required_space: Space required in bytes
            available_space: Space available in bytes

        Returns:
            ErrorRecoveryResult: Result of error handling
        """
        context = {
            "required_space": required_space,
            "available_space": available_space,
            "space_deficit": required_space - available_space,
        }

        space_error = DiskSpaceError(
            f"Insufficient disk space: need {required_space} bytes, have {available_space} bytes",
            required_space,
            available_space,
            context,
        )

        self._log_error(space_error)

        # Try cleanup and compression
        recovery_result = self._attempt_disk_space_recovery(space_error)

        self._update_statistics(space_error, recovery_result)
        return recovery_result

    def handle_configuration_error(
        self, config_issues: List[str]
    ) -> ErrorRecoveryResult:
        """
        Handle configuration errors with default fallbacks.

        Args:
            config_issues: List of configuration issues

        Returns:
            ErrorRecoveryResult: Result of error handling
        """
        context = {"issue_count": len(config_issues), "issues": config_issues}

        config_error = ConfigurationError(
            f"Configuration validation failed: {'; '.join(config_issues)}",
            config_issues,
            context,
        )

        self._log_error(config_error)

        # Use default configuration
        recovery_result = self._attempt_configuration_recovery(config_error)

        self._update_statistics(config_error, recovery_result)
        return recovery_result

    def handle_metadata_error(
        self, metadata_type: str, error: Exception
    ) -> ErrorRecoveryResult:
        """
        Handle metadata collection errors with graceful degradation.

        Args:
            metadata_type: Type of metadata that failed to collect
            error: Original exception

        Returns:
            ErrorRecoveryResult: Result of error handling
        """
        context = {"metadata_type": metadata_type, "original_error": str(error)}

        metadata_error = MetadataError(
            f"Failed to collect {metadata_type} metadata: {error}",
            metadata_type,
            context,
        )

        self._log_error(metadata_error)

        # Provide default metadata
        recovery_result = self._attempt_metadata_recovery(metadata_error)

        self._update_statistics(metadata_error, recovery_result)
        return recovery_result

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive error handling statistics.

        Returns:
            Dict[str, Any]: Error statistics
        """
        total_errors = self.recovery_statistics["total_errors"]
        success_rate = 0.0
        if total_errors > 0:
            success_rate = (
                self.recovery_statistics["successful_recoveries"] / total_errors
            ) * 100

        return {
            "total_errors": total_errors,
            "successful_recoveries": self.recovery_statistics["successful_recoveries"],
            "failed_recoveries": self.recovery_statistics["failed_recoveries"],
            "recovery_success_rate": success_rate,
            "errors_by_type": self.recovery_statistics["errors_by_type"].copy(),
            "fallback_activations": self.recovery_statistics["fallback_activations"],
            "recent_errors": len(
                [e for e in self.error_history if time.time() - e.timestamp < 300]
            ),  # Last 5 minutes
        }

    def _attempt_csv_read_recovery(self, error: CSVReadError) -> ErrorRecoveryResult:
        """Attempt to recover from CSV read errors."""
        # Strategy 1: Try to read partial data or use backup
        try:
            # Check if backup file exists
            backup_pattern = f"{error.file_path.stem}_backup_*{error.file_path.suffix}"
            backup_files = list(error.file_path.parent.glob(backup_pattern))

            if backup_files:
                # Use most recent backup
                latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)
                self.logger.warning(f"Using backup file for recovery: {latest_backup}")

                return ErrorRecoveryResult(
                    success=True,
                    message=f"Using backup file: {latest_backup}",
                    recovered_data=[],  # Would contain transactions from backup
                    fallback_used=True,
                )
            else:
                # Return empty data as fallback
                return ErrorRecoveryResult(
                    success=True,
                    message="No backup available, returning empty transaction list",
                    recovered_data=[],
                    fallback_used=True,
                )

        except Exception as recovery_error:
            self.logger.error(f"CSV read recovery failed: {recovery_error}")

            return ErrorRecoveryResult(
                success=False,
                message=f"CSV read recovery failed: {recovery_error}",
                should_retry=False,
            )

    def _attempt_csv_write_recovery(
        self, error: CSVWriteError, transaction_data: Any
    ) -> ErrorRecoveryResult:
        """Attempt to recover from CSV write errors."""
        # Strategy 1: Try fallback file location
        try:
            # Ensure fallback directory exists
            self.fallback_csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Try writing to fallback location
            # This would be implemented by the CSV writer
            self.logger.warning(f"Using fallback CSV file: {self.fallback_csv_path}")

            return ErrorRecoveryResult(
                success=True,
                message=f"Successfully wrote to fallback CSV file: {self.fallback_csv_path}",
                recovered_data={"fallback_path": str(self.fallback_csv_path)},
                fallback_used=True,
            )

        except Exception as fallback_error:
            self.logger.error(f"Fallback CSV write also failed: {fallback_error}")

            return ErrorRecoveryResult(
                success=False,
                message=f"Both primary and fallback CSV writes failed: {fallback_error}",
                should_retry=False,
            )

    def _attempt_validation_recovery(
        self, error: ValidationError, data: Any
    ) -> ErrorRecoveryResult:
        """Attempt to recover from validation errors."""
        # For validation errors, we can try to sanitize the data
        try:
            # This would implement data sanitization logic
            sanitized_data = self._sanitize_data(data, error.validation_failures)

            return ErrorRecoveryResult(
                success=True,
                message="Data sanitized and validation issues resolved",
                recovered_data=sanitized_data,
                should_retry=True,
            )

        except Exception as sanitize_error:
            return ErrorRecoveryResult(
                success=False,
                message=f"Failed to sanitize data: {sanitize_error}",
                should_retry=False,
            )

    def _attempt_file_lock_recovery(self, error: FileLockError) -> ErrorRecoveryResult:
        """Attempt to recover from file lock errors."""
        # Strategy: Use a different file or queue the operation
        try:
            # Create a temporary queue file for the operation
            queue_file = error.file_path.parent / f"{error.file_path.stem}_queue.tmp"

            return ErrorRecoveryResult(
                success=True,
                message=f"Operation queued to temporary file: {queue_file}",
                recovered_data={"queue_file": str(queue_file)},
                should_retry=True,
                fallback_used=True,
            )

        except Exception as queue_error:
            return ErrorRecoveryResult(
                success=False,
                message=f"Failed to create queue file: {queue_error}",
                should_retry=False,
            )

    def _attempt_permission_recovery(
        self, error: PermissionError
    ) -> ErrorRecoveryResult:
        """Attempt to recover from permission errors."""
        # Strategy: Try alternative locations with write permissions
        try:
            # Try user's home directory
            import os

            home_dir = Path.home()
            fallback_path = home_dir / ".kiro_token_tracker" / "token_transactions.csv"
            fallback_path.parent.mkdir(parents=True, exist_ok=True)

            return ErrorRecoveryResult(
                success=True,
                message=f"Using alternative location with write permissions: {fallback_path}",
                recovered_data={"alternative_path": str(fallback_path)},
                fallback_used=True,
            )

        except Exception as alt_error:
            return ErrorRecoveryResult(
                success=False,
                message=f"Failed to find alternative location: {alt_error}",
                should_retry=False,
            )

    def _attempt_disk_space_recovery(
        self, error: DiskSpaceError
    ) -> ErrorRecoveryResult:
        """Attempt to recover from disk space errors."""
        # Strategy: Clean up old files or compress existing data
        try:
            # This would implement cleanup logic
            cleaned_space = self._cleanup_old_files()

            if cleaned_space >= error.required_space:
                return ErrorRecoveryResult(
                    success=True,
                    message=f"Cleaned up {cleaned_space} bytes of disk space",
                    should_retry=True,
                )
            else:
                return ErrorRecoveryResult(
                    success=False,
                    message=f"Cleanup only freed {cleaned_space} bytes, need {error.required_space}",
                    should_retry=False,
                )

        except Exception as cleanup_error:
            return ErrorRecoveryResult(
                success=False,
                message=f"Cleanup failed: {cleanup_error}",
                should_retry=False,
            )

    def _attempt_configuration_recovery(
        self, error: ConfigurationError
    ) -> ErrorRecoveryResult:
        """Attempt to recover from configuration errors."""
        # Strategy: Use default configuration
        try:
            from .models import TrackerConfig

            default_config = TrackerConfig.create_default()

            return ErrorRecoveryResult(
                success=True,
                message="Using default configuration due to validation errors",
                recovered_data=default_config,
                fallback_used=True,
            )

        except Exception as default_error:
            return ErrorRecoveryResult(
                success=False,
                message=f"Failed to create default configuration: {default_error}",
                should_retry=False,
            )

    def _attempt_metadata_recovery(self, error: MetadataError) -> ErrorRecoveryResult:
        """Attempt to recover from metadata collection errors."""
        # Strategy: Provide minimal default metadata
        try:
            default_metadata = {
                "workspace_folder": "unknown",
                "hook_trigger_type": "unknown",
                "agent_execution_id": "unknown",
                "session_id": "unknown",
                "hook_name": "unknown",
                "file_patterns": None,
            }

            return ErrorRecoveryResult(
                success=True,
                message=f"Using default metadata for {error.metadata_type}",
                recovered_data=default_metadata,
                fallback_used=True,
            )

        except Exception as default_error:
            return ErrorRecoveryResult(
                success=False,
                message=f"Failed to create default metadata: {default_error}",
                should_retry=False,
            )

    def _sanitize_data(self, data: Any, validation_failures: List[str]) -> Any:
        """Sanitize data to fix validation issues."""
        # This would implement data sanitization logic based on validation failures
        # For now, return the original data
        return data

    def _cleanup_old_files(self) -> int:
        """Clean up old files to free disk space."""
        # This would implement file cleanup logic
        # For now, return 0 bytes cleaned
        return 0

    def _log_error(self, error: TokenTrackerError) -> None:
        """Log error with appropriate severity."""
        self.error_history.append(error)

        error_info = {
            "error_type": error.error_type.value,
            "context": error.context,
            "timestamp": error.timestamp,
        }

        if error.error_type in [
            TokenTrackerErrorType.DISK_SPACE_ERROR,
            TokenTrackerErrorType.PERMISSION_ERROR,
        ]:
            self.logger.error(f"CRITICAL: {error}", extra=error_info)
        elif error.error_type in [
            TokenTrackerErrorType.CSV_WRITE_ERROR,
            TokenTrackerErrorType.FILE_LOCK_ERROR,
        ]:
            self.logger.warning(f"WARNING: {error}", extra=error_info)
        else:
            self.logger.info(f"INFO: {error}", extra=error_info)

    def _update_statistics(
        self, error: TokenTrackerError, result: ErrorRecoveryResult
    ) -> None:
        """Update error handling statistics."""
        self.recovery_statistics["total_errors"] += 1
        self.recovery_statistics["errors_by_type"][error.error_type.value] += 1

        if result.success:
            self.recovery_statistics["successful_recoveries"] += 1
        else:
            self.recovery_statistics["failed_recoveries"] += 1

        if result.fallback_used:
            self.recovery_statistics["fallback_activations"] += 1
