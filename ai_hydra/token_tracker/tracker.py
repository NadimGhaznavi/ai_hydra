"""
Core token tracker service.

This module provides the main TokenTracker class that orchestrates
token usage tracking, CSV operations, and metadata collection.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import threading
import time

from .models import TokenTransaction, TrackerConfig
from .csv_writer import CSVWriter
from .metadata_collector import MetadataCollector
from .error_handler import (
    TokenTrackerErrorHandler,
    TokenTrackerError,
    ValidationError,
    ConfigurationError,
)


class TokenTracker:
    """
    Core service for tracking token transactions.

    This class provides the main interface for recording token usage,
    managing CSV storage, and collecting metadata from the Kiro IDE environment.
    """

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the token tracker.

        Args:
            config: Tracker configuration (uses default if not provided)
            logger: Logger instance (creates default if not provided)
        """
        # Initialize configuration
        self.config = config or TrackerConfig.create_default()

        # Validate configuration
        config_issues = self.config.validate()
        if config_issues:
            raise ConfigurationError("Configuration validation failed", config_issues)

        # Initialize logging
        self.logger = logger or self._setup_logging()

        # Initialize components
        self.error_handler = TokenTrackerErrorHandler(self.logger)
        self.metadata_collector = MetadataCollector(self.error_handler, self.logger)
        self.csv_writer = CSVWriter(self.config, self.error_handler, self.logger)

        # Thread safety
        self._operation_lock = threading.RLock()

        # Statistics tracking
        self.statistics = {
            "transactions_recorded": 0,
            "transactions_failed": 0,
            "total_tokens_tracked": 0,
            "total_execution_time": 0.0,
            "start_time": datetime.now(),
            "last_transaction_time": None,
            "errors_encountered": 0,
        }

        # Performance monitoring
        self._performance_metrics = {
            "avg_record_time": 0.0,
            "max_record_time": 0.0,
            "min_record_time": float("inf"),
            "recent_record_times": [],
        }

        self.logger.info(
            f"TokenTracker initialized with config: {self.config.to_dict()}"
        )

    def record_transaction(
        self,
        prompt_text: str,
        tokens_used: int,
        elapsed_time: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Record a token transaction with full metadata.

        Args:
            prompt_text: The prompt text sent to the AI model
            tokens_used: Number of tokens consumed
            elapsed_time: Time taken for the operation in seconds
            context: Additional context information from the hook

        Returns:
            bool: True if transaction was recorded successfully, False otherwise
        """
        if not self.config.enabled:
            self.logger.debug("Token tracking is disabled")
            return True

        record_start_time = time.time()

        try:
            with self._operation_lock:
                # Collect metadata
                metadata = self.metadata_collector.collect_execution_metadata(context)

                # Create transaction
                transaction = TokenTransaction.create_new(
                    prompt_text=prompt_text,
                    tokens_used=tokens_used,
                    elapsed_time=elapsed_time,
                    workspace_folder=metadata.get("workspace_folder", "unknown"),
                    hook_trigger_type=metadata.get("hook_trigger_type", "unknown"),
                    session_id=metadata.get("session_id"),
                    agent_execution_id=metadata.get("agent_execution_id"),
                    file_patterns=metadata.get("file_patterns"),
                    hook_name=metadata.get("hook_name", "unknown"),
                )

                # Validate transaction
                if self.config.enable_validation:
                    validation_result = self._validate_transaction(transaction)
                    if not validation_result["valid"]:
                        recovery_result = self.error_handler.handle_validation_error(
                            validation_result["issues"], transaction
                        )
                        if not recovery_result.success:
                            self._update_failure_statistics()
                            return False

                        # Use sanitized transaction if available
                        if recovery_result.recovered_data:
                            transaction = recovery_result.recovered_data

                # Write to CSV
                write_success = self.csv_writer.write_transaction(transaction)

                if write_success:
                    self._update_success_statistics(transaction, record_start_time)
                    self.logger.debug(
                        f"Transaction recorded: {transaction.get_summary()}"
                    )
                    return True
                else:
                    self._update_failure_statistics()
                    return False

        except Exception as e:
            self._update_failure_statistics()
            recovery_result = self.error_handler.handle_error(
                e,
                "token_tracker",
                "record_transaction",
                {"prompt_length": len(prompt_text), "tokens_used": tokens_used},
            )

            if recovery_result.success and recovery_result.should_retry:
                # Retry once with recovered data
                try:
                    return self._retry_record_transaction(
                        prompt_text, tokens_used, elapsed_time, context, recovery_result
                    )
                except Exception as retry_error:
                    self.logger.error(f"Retry failed: {retry_error}")

            return False

    def get_transaction_history(
        self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None
    ) -> List[TokenTransaction]:
        """
        Retrieve transaction history with optional filtering.

        Args:
            filters: Optional filters to apply (e.g., date range, workspace)
            limit: Maximum number of transactions to return

        Returns:
            List[TokenTransaction]: List of transactions matching filters
        """
        try:
            with self._operation_lock:
                # Read all transactions from CSV
                all_transactions = self.csv_writer.read_transactions(limit)

                # Apply filters if provided
                if filters:
                    filtered_transactions = self._apply_filters(
                        all_transactions, filters
                    )
                    return filtered_transactions

                return all_transactions

        except Exception as e:
            self.logger.error(f"Failed to retrieve transaction history: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive tracking statistics.

        Returns:
            Dict[str, Any]: Statistics about token tracking
        """
        with self._operation_lock:
            current_time = datetime.now()
            uptime = current_time - self.statistics["start_time"]

            stats = self.statistics.copy()
            stats.update(
                {
                    "uptime_seconds": uptime.total_seconds(),
                    "uptime_formatted": str(uptime),
                    "transactions_per_hour": self._calculate_transactions_per_hour(),
                    "average_tokens_per_transaction": self._calculate_average_tokens(),
                    "average_execution_time": self._calculate_average_execution_time(),
                    "performance_metrics": self._performance_metrics.copy(),
                    "error_statistics": self.error_handler.get_error_statistics(),
                    "csv_integrity": self.csv_writer.validate_csv_integrity(),
                }
            )

            return stats

    def validate_csv_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of the CSV file.

        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            return self.csv_writer.validate_csv_integrity()
        except Exception as e:
            self.logger.error(f"CSV integrity validation failed: {e}")
            return {
                "file_exists": False,
                "validation_issues": [f"Validation failed: {e}"],
            }

    def create_backup(self) -> Optional[Path]:
        """
        Create a backup of the CSV file.

        Returns:
            Optional[Path]: Path to backup file if successful, None otherwise
        """
        try:
            return self.csv_writer.create_backup()
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return None

    def cleanup_old_data(self, max_age_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Clean up old transaction data based on retention policy.

        Args:
            max_age_days: Maximum age of data to keep (uses config default if not provided)

        Returns:
            Dict[str, Any]: Cleanup results
        """
        max_age = max_age_days or self.config.retention_days
        cleanup_results = {
            "transactions_removed": 0,
            "backups_cleaned": 0,
            "space_freed_bytes": 0,
            "cleanup_successful": False,
        }

        try:
            with self._operation_lock:
                # Clean up old backups
                backups_cleaned = self.csv_writer.cleanup_old_backups(max_age)
                cleanup_results["backups_cleaned"] = backups_cleaned

                # TODO: Implement transaction data cleanup
                # This would involve reading the CSV, filtering out old transactions,
                # and rewriting the file

                cleanup_results["cleanup_successful"] = True
                self.logger.info(f"Cleanup completed: {cleanup_results}")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            cleanup_results["error"] = str(e)

        return cleanup_results

    def export_data(
        self,
        output_path: Union[str, Path],
        format: str = "csv",
        filters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Export transaction data to a file.

        Args:
            output_path: Path to export file
            format: Export format ('csv', 'json')
            filters: Optional filters to apply

        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            output_path = Path(output_path)

            # Get transactions to export
            transactions = self.get_transaction_history(filters)

            if format.lower() == "csv":
                return self._export_csv(transactions, output_path)
            elif format.lower() == "json":
                return self._export_json(transactions, output_path)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False

    def _setup_logging(self) -> logging.Logger:
        """Setup logging based on configuration."""
        logger = logging.getLogger(f"{__name__}.TokenTracker")

        # Set log level
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

    def _validate_transaction(self, transaction: TokenTransaction) -> Dict[str, Any]:
        """Validate a transaction before recording."""
        issues = []

        # Check prompt length
        if len(transaction.prompt_text) > self.config.max_prompt_length:
            issues.append(
                f"Prompt text exceeds maximum length: {len(transaction.prompt_text)} > {self.config.max_prompt_length}"
            )

        # Check for required fields
        if not transaction.prompt_text.strip():
            issues.append("Prompt text is empty")

        if transaction.tokens_used < 0:
            issues.append("Tokens used cannot be negative")

        if transaction.elapsed_time < 0:
            issues.append("Elapsed time cannot be negative")

        # Check for reasonable values
        if transaction.tokens_used > 1000000:  # 1M tokens seems excessive
            issues.append(f"Tokens used seems excessive: {transaction.tokens_used}")

        if transaction.elapsed_time > 3600:  # 1 hour seems excessive
            issues.append(
                f"Elapsed time seems excessive: {transaction.elapsed_time} seconds"
            )

        return {"valid": len(issues) == 0, "issues": issues}

    def _retry_record_transaction(
        self,
        prompt_text: str,
        tokens_used: int,
        elapsed_time: float,
        context: Optional[Dict[str, Any]],
        recovery_result,
    ) -> bool:
        """Retry recording transaction with recovery data."""
        # This would implement retry logic using recovery_result data
        # For now, just try the original operation again
        return self.record_transaction(prompt_text, tokens_used, elapsed_time, context)

    def _apply_filters(
        self, transactions: List[TokenTransaction], filters: Dict[str, Any]
    ) -> List[TokenTransaction]:
        """Apply filters to transaction list."""
        filtered = transactions

        # Date range filter
        if "start_date" in filters or "end_date" in filters:
            start_date = filters.get("start_date")
            end_date = filters.get("end_date")

            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)

            filtered = [
                t
                for t in filtered
                if (not start_date or t.timestamp >= start_date)
                and (not end_date or t.timestamp <= end_date)
            ]

        # Workspace filter
        if "workspace_folder" in filters:
            workspace = filters["workspace_folder"]
            filtered = [t for t in filtered if t.workspace_folder == workspace]

        # Hook type filter
        if "hook_trigger_type" in filters:
            hook_type = filters["hook_trigger_type"]
            filtered = [t for t in filtered if t.hook_trigger_type == hook_type]

        # Token range filter
        if "min_tokens" in filters or "max_tokens" in filters:
            min_tokens = filters.get("min_tokens", 0)
            max_tokens = filters.get("max_tokens", float("inf"))
            filtered = [
                t for t in filtered if min_tokens <= t.tokens_used <= max_tokens
            ]

        return filtered

    def _update_success_statistics(
        self, transaction: TokenTransaction, record_start_time: float
    ) -> None:
        """Update statistics after successful transaction recording."""
        record_time = time.time() - record_start_time

        self.statistics["transactions_recorded"] += 1
        self.statistics["total_tokens_tracked"] += transaction.tokens_used
        self.statistics["total_execution_time"] += transaction.elapsed_time
        self.statistics["last_transaction_time"] = datetime.now()

        # Update performance metrics
        self._performance_metrics["recent_record_times"].append(record_time)
        if len(self._performance_metrics["recent_record_times"]) > 100:
            self._performance_metrics["recent_record_times"].pop(0)

        self._performance_metrics["max_record_time"] = max(
            self._performance_metrics["max_record_time"], record_time
        )
        self._performance_metrics["min_record_time"] = min(
            self._performance_metrics["min_record_time"], record_time
        )

        if self._performance_metrics["recent_record_times"]:
            self._performance_metrics["avg_record_time"] = sum(
                self._performance_metrics["recent_record_times"]
            ) / len(self._performance_metrics["recent_record_times"])

    def _update_failure_statistics(self) -> None:
        """Update statistics after failed transaction recording."""
        self.statistics["transactions_failed"] += 1
        self.statistics["errors_encountered"] += 1

    def _calculate_transactions_per_hour(self) -> float:
        """Calculate transactions per hour rate."""
        uptime_hours = (
            datetime.now() - self.statistics["start_time"]
        ).total_seconds() / 3600
        if uptime_hours > 0:
            return self.statistics["transactions_recorded"] / uptime_hours
        return 0.0

    def _calculate_average_tokens(self) -> float:
        """Calculate average tokens per transaction."""
        if self.statistics["transactions_recorded"] > 0:
            return (
                self.statistics["total_tokens_tracked"]
                / self.statistics["transactions_recorded"]
            )
        return 0.0

    def _calculate_average_execution_time(self) -> float:
        """Calculate average execution time per transaction."""
        if self.statistics["transactions_recorded"] > 0:
            return (
                self.statistics["total_execution_time"]
                / self.statistics["transactions_recorded"]
            )
        return 0.0

    def _export_csv(
        self, transactions: List[TokenTransaction], output_path: Path
    ) -> bool:
        """Export transactions to CSV format."""
        try:
            import csv

            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                from .models import CSVSchema

                schema = CSVSchema()
                writer.writerow(schema.headers)

                # Write transactions
                for transaction in transactions:
                    writer.writerow(transaction.to_csv_row())

            self.logger.info(
                f"Exported {len(transactions)} transactions to {output_path}"
            )
            return True

        except Exception as e:
            self.logger.error(f"CSV export failed: {e}")
            return False

    def _export_json(
        self, transactions: List[TokenTransaction], output_path: Path
    ) -> bool:
        """Export transactions to JSON format."""
        try:
            import json

            # Convert transactions to dictionaries
            transaction_dicts = []
            for transaction in transactions:
                transaction_dict = {
                    "timestamp": transaction.timestamp.isoformat(),
                    "prompt_text": transaction.prompt_text,
                    "tokens_used": transaction.tokens_used,
                    "elapsed_time": transaction.elapsed_time,
                    "session_id": transaction.session_id,
                    "workspace_folder": transaction.workspace_folder,
                    "hook_trigger_type": transaction.hook_trigger_type,
                    "agent_execution_id": transaction.agent_execution_id,
                    "file_patterns": transaction.file_patterns,
                    "hook_name": transaction.hook_name,
                    "error_occurred": transaction.error_occurred,
                    "error_message": transaction.error_message,
                }
                transaction_dicts.append(transaction_dict)

            # Write JSON file
            with open(output_path, "w", encoding="utf-8") as jsonfile:
                json.dump(transaction_dicts, jsonfile, indent=2, ensure_ascii=False)

            self.logger.info(
                f"Exported {len(transactions)} transactions to {output_path}"
            )
            return True

        except Exception as e:
            self.logger.error(f"JSON export failed: {e}")
            return False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        try:
            # Log final statistics
            final_stats = self.get_statistics()
            self.logger.info(f"TokenTracker session completed: {final_stats}")

            # Cleanup CSV writer
            if hasattr(self.csv_writer, "__exit__"):
                self.csv_writer.__exit__(exc_type, exc_val, exc_tb)

        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
