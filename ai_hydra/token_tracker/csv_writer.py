"""
Thread-safe CSV writer for token transactions.

This module provides a robust CSV writer with file locking, validation,
and error handling for storing token transaction data safely.
"""

import csv
import fcntl
import os
import threading
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from contextlib import contextmanager
import logging
import datetime as dt
from datetime import timedelta

from .models import TokenTransaction, TrackerConfig, CSVSchema
from .error_handler import (
    TokenTrackerErrorHandler,
    CSVWriteError,
    CSVReadError,
    FileLockError,
    PermissionError,
    DiskSpaceError,
)


class CSVWriter:
    """
    Thread-safe CSV writer for token transactions.

    This class provides safe, concurrent access to CSV files with proper
    locking, validation, and error handling mechanisms.
    """

    def __init__(
        self,
        config: TrackerConfig,
        error_handler: Optional[TokenTrackerErrorHandler] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the CSV writer.

        Args:
            config: Tracker configuration
            error_handler: Error handler instance
            logger: Logger instance
        """
        self.config = config
        self.error_handler = error_handler or TokenTrackerErrorHandler()
        self.logger = logger or logging.getLogger(__name__)
        self.schema = CSVSchema()

        # Thread safety - use RLock for reentrant locking
        self._write_lock = threading.RLock()
        self._active_locks: Dict[str, threading.Lock] = {}
        self._lock_manager_lock = threading.Lock()  # Protects _active_locks dict

        # File management
        self._last_backup_time: Optional[dt.datetime] = None
        self._file_handles: Dict[str, Any] = {}

        # Transaction serialization/deserialization support
        self._serialization_lock = threading.Lock()

        # Initialize CSV file if needed
        self._initialize_csv_file()

        self.logger.info(f"CSVWriter initialized for {self.config.get_csv_file_path()}")

    def write_transaction(self, transaction: TokenTransaction) -> bool:
        """
        Write a single transaction to the CSV file.

        Args:
            transaction: Transaction to write

        Returns:
            bool: True if write was successful, False otherwise
        """
        try:
            with self._write_lock:
                return self._write_transaction_locked(transaction)

        except Exception as e:
            recovery_result = self.error_handler.handle_csv_write_error(
                e, self.config.get_csv_file_path(), transaction
            )

            if recovery_result.success and recovery_result.should_retry:
                # Try again with recovered data or fallback path
                try:
                    return self._write_transaction_with_fallback(
                        transaction, recovery_result
                    )
                except Exception as retry_error:
                    self.logger.error(f"Retry write failed: {retry_error}")
                    return False

            return False

    def write_transactions_batch(self, transactions: List[TokenTransaction]) -> int:
        """
        Write multiple transactions in a batch operation.

        Args:
            transactions: List of transactions to write

        Returns:
            int: Number of transactions successfully written
        """
        if not transactions:
            return 0

        successful_writes = 0

        try:
            with self._write_lock:
                for transaction in transactions:
                    if self._write_transaction_locked(transaction):
                        successful_writes += 1
                    else:
                        self.logger.warning(
                            f"Failed to write transaction: {transaction.get_summary()}"
                        )

        except Exception as e:
            self.logger.error(
                f"Batch write failed after {successful_writes} transactions: {e}"
            )

        self.logger.info(
            f"Batch write completed: {successful_writes}/{len(transactions)} successful"
        )
        return successful_writes

    def read_transactions(self, limit: Optional[int] = None) -> List[TokenTransaction]:
        """
        Read transactions from the CSV file.

        Args:
            limit: Maximum number of transactions to read (None for all)

        Returns:
            List[TokenTransaction]: List of transactions
        """
        transactions = []
        csv_path = self.config.get_csv_file_path()

        try:
            if not csv_path.exists():
                self.logger.info(f"CSV file does not exist: {csv_path}")
                return transactions

            with self._acquire_file_lock(csv_path, "r"):
                with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
                    reader = csv.reader(csvfile)

                    # Skip header row
                    try:
                        headers = next(reader)
                        validation_issues = self.schema.validate_headers(headers)
                        if validation_issues:
                            self.logger.warning(
                                f"CSV header validation issues: {validation_issues}"
                            )
                    except StopIteration:
                        self.logger.info("CSV file is empty")
                        return transactions

                    # Read data rows
                    for row_num, row in enumerate(reader, start=2):
                        if limit and len(transactions) >= limit:
                            break

                        try:
                            # Validate row
                            if self.config.enable_validation:
                                validation_issues = self.schema.validate_row(row)
                                if validation_issues:
                                    self.logger.warning(
                                        f"Row {row_num} validation issues: {validation_issues}"
                                    )
                                    continue

                            # Parse transaction
                            transaction = TokenTransaction.from_csv_row(row)
                            transactions.append(transaction)

                        except Exception as row_error:
                            self.logger.error(
                                f"Failed to parse row {row_num}: {row_error}"
                            )
                            continue

            self.logger.info(f"Read {len(transactions)} transactions from CSV")
            return transactions

        except Exception as e:
            recovery_result = self.error_handler.handle_csv_read_error(e, csv_path)
            if recovery_result.success and recovery_result.recovered_data:
                return recovery_result.recovered_data

            self.logger.error(f"Failed to read transactions: {e}")
            return transactions

    def serialize_transaction(self, transaction: TokenTransaction) -> Dict[str, Any]:
        """
        Serialize a transaction to a dictionary format.

        Args:
            transaction: Transaction to serialize

        Returns:
            Dict[str, Any]: Serialized transaction data
        """
        with self._serialization_lock:
            try:
                serialized = {
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

                self.logger.debug(
                    f"Transaction serialized: {transaction.get_summary()}"
                )
                return serialized

            except Exception as e:
                self.logger.error(f"Failed to serialize transaction: {e}")
                raise

    def deserialize_transaction(self, data: Dict[str, Any]) -> TokenTransaction:
        """
        Deserialize a transaction from dictionary format.

        Args:
            data: Serialized transaction data

        Returns:
            TokenTransaction: Deserialized transaction

        Raises:
            ValueError: If data is invalid or incomplete
        """
        with self._serialization_lock:
            try:
                # Parse timestamp
                timestamp = dt.datetime.fromisoformat(data["timestamp"])

                # Create transaction
                transaction = TokenTransaction(
                    timestamp=timestamp,
                    prompt_text=data["prompt_text"],
                    tokens_used=data["tokens_used"],
                    elapsed_time=data["elapsed_time"],
                    session_id=data["session_id"],
                    workspace_folder=data["workspace_folder"],
                    hook_trigger_type=data["hook_trigger_type"],
                    agent_execution_id=data["agent_execution_id"],
                    file_patterns=data.get("file_patterns"),
                    hook_name=data.get("hook_name", "unknown"),
                    error_occurred=data.get("error_occurred", False),
                    error_message=data.get("error_message"),
                )

                self.logger.debug(
                    f"Transaction deserialized: {transaction.get_summary()}"
                )
                return transaction

            except Exception as e:
                self.logger.error(f"Failed to deserialize transaction: {e}")
                raise ValueError(f"Invalid transaction data: {e}")

    def validate_csv_headers(self) -> bool:
        """
        Validate that CSV file has correct headers.

        Returns:
            bool: True if headers are valid, False otherwise
        """
        csv_path = self.config.get_csv_file_path()

        if not csv_path.exists():
            self.logger.warning(f"CSV file does not exist: {csv_path}")
            return False

        try:
            with self._acquire_file_lock(csv_path, "r"):
                with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
                    reader = csv.reader(csvfile)
                    try:
                        headers = next(reader)
                        validation_issues = self.schema.validate_headers(headers)

                        if validation_issues:
                            self.logger.error(
                                f"CSV header validation failed: {validation_issues}"
                            )
                            return False

                        self.logger.debug("CSV headers validated successfully")
                        return True

                    except StopIteration:
                        self.logger.error("CSV file is empty - no headers found")
                        return False

        except Exception as e:
            self.logger.error(f"Failed to validate CSV headers: {e}")
            return False
        """
        Validate the integrity of the CSV file.

        Returns:
            Dict[str, Any]: Validation results
        """
        csv_path = self.config.get_csv_file_path()
        results = {
            "file_exists": csv_path.exists(),
            "file_readable": False,
            "file_writable": False,
            "header_valid": False,
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "validation_issues": [],
            "file_size_bytes": 0,
        }

        try:
            if not results["file_exists"]:
                results["validation_issues"].append("CSV file does not exist")
                return results

            # Check file permissions
            results["file_readable"] = os.access(csv_path, os.R_OK)
            results["file_writable"] = os.access(csv_path, os.W_OK)
            results["file_size_bytes"] = csv_path.stat().st_size

            if not results["file_readable"]:
                results["validation_issues"].append("CSV file is not readable")
                return results

            # Validate file contents
            with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)

                # Validate headers
                try:
                    headers = next(reader)
                    header_issues = self.schema.validate_headers(headers)
                    results["header_valid"] = len(header_issues) == 0
                    if header_issues:
                        results["validation_issues"].extend(header_issues)
                except StopIteration:
                    results["validation_issues"].append("CSV file is empty")
                    return results

                # Validate data rows
                for row_num, row in enumerate(reader, start=2):
                    results["total_rows"] += 1

                    row_issues = self.schema.validate_row(row)
                    if row_issues:
                        results["invalid_rows"] += 1
                        results["validation_issues"].extend(
                            [f"Row {row_num}: {issue}" for issue in row_issues]
                        )
                    else:
                        results["valid_rows"] += 1

            self.logger.info(
                f"CSV validation completed: {results['valid_rows']}/{results['total_rows']} valid rows"
            )

        except Exception as e:
            results["validation_issues"].append(f"Validation failed: {e}")
            self.logger.error(f"CSV validation error: {e}")

        return results

    def create_backup(self) -> Optional[Path]:
        """
        Create a backup of the CSV file.

        Returns:
            Optional[Path]: Path to backup file if successful, None otherwise
        """
        if not self.config.backup_enabled:
            return None

        csv_path = self.config.get_csv_file_path()
        if not csv_path.exists():
            self.logger.info("No CSV file to backup")
            return None

        try:
            backup_path = self.config.get_backup_file_path()
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file with locking
            with self._acquire_file_lock(csv_path, "r"):
                with open(csv_path, "rb") as src:
                    with open(backup_path, "wb") as dst:
                        dst.write(src.read())

            self._last_backup_time = dt.datetime.now()
            self.logger.info(f"Backup created: {backup_path}")
            return backup_path

        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return None

    def cleanup_old_backups(self, max_age_days: int = 30) -> int:
        """
        Clean up old backup files.

        Args:
            max_age_days: Maximum age of backups to keep

        Returns:
            int: Number of backups cleaned up
        """
        if not self.config.backup_enabled:
            return 0

        csv_path = self.config.get_csv_file_path()
        backup_dir = csv_path.parent
        backup_pattern = f"{csv_path.stem}_backup_*{csv_path.suffix}"

        cleaned_count = 0
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        try:
            for backup_file in backup_dir.glob(backup_pattern):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    cleaned_count += 1
                    self.logger.debug(f"Cleaned up old backup: {backup_file}")

            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old backup files")

        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")

        return cleaned_count

    def _initialize_csv_file(self) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        csv_path = self.config.get_csv_file_path()

        try:
            # Create directory if needed
            if self.config.auto_create_directories:
                csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Create file with headers if it doesn't exist
            if not csv_path.exists():
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.schema.headers)

                self.logger.info(f"Created new CSV file with headers: {csv_path}")

        except Exception as e:
            recovery_result = self.error_handler.handle_csv_write_error(
                e, csv_path, "headers"
            )
            if not recovery_result.success:
                raise CSVWriteError(f"Failed to initialize CSV file: {e}", csv_path)

    def _write_transaction_locked(self, transaction: TokenTransaction) -> bool:
        """Write transaction with existing lock held."""
        csv_path = self.config.get_csv_file_path()

        try:
            # Check disk space
            self._check_disk_space(transaction)

            # Create backup if needed
            if self.config.should_create_backup(self._last_backup_time):
                self.create_backup()

            # Write transaction
            with self._acquire_file_lock(csv_path, "a"):
                with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(transaction.to_csv_row())

            self.logger.debug(f"Transaction written: {transaction.get_summary()}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to write transaction: {e}")
            raise

    def _write_transaction_with_fallback(
        self, transaction: TokenTransaction, recovery_result
    ) -> bool:
        """Write transaction using fallback mechanism."""
        if "fallback_path" in recovery_result.recovered_data:
            fallback_path = Path(recovery_result.recovered_data["fallback_path"])

            try:
                # Initialize fallback file if needed
                if not fallback_path.exists():
                    fallback_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(
                        fallback_path, "w", newline="", encoding="utf-8"
                    ) as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(self.schema.headers)

                # Write to fallback file
                with open(fallback_path, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(transaction.to_csv_row())

                self.logger.info(
                    f"Transaction written to fallback file: {fallback_path}"
                )
                return True

            except Exception as fallback_error:
                self.logger.error(f"Fallback write failed: {fallback_error}")
                return False

        return False

    @contextmanager
    def _acquire_file_lock(self, file_path: Path, mode: str):
        """
        Acquire file lock with timeout and enhanced error handling.

        This method provides thread-safe and process-safe file locking
        with proper timeout handling and resource cleanup.
        """
        lock_key = str(file_path)

        # Thread-safe lock management
        with self._lock_manager_lock:
            if lock_key not in self._active_locks:
                self._active_locks[lock_key] = threading.Lock()
            file_lock = self._active_locks[lock_key]

        # Acquire thread lock with timeout
        acquired = file_lock.acquire(timeout=self.config.file_lock_timeout_seconds)
        if not acquired:
            raise FileLockError(
                f"Failed to acquire thread lock for {file_path}",
                file_path,
                self.config.file_lock_timeout_seconds,
            )

        file_handle = None
        try:
            # Open file and acquire file system lock
            file_handle = open(file_path, mode, encoding="utf-8")

            # Try to acquire file lock with timeout and retry logic
            start_time = time.time()
            lock_acquired = False

            while time.time() - start_time < self.config.file_lock_timeout_seconds:
                try:
                    fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_acquired = True
                    break
                except BlockingIOError:
                    # File is locked by another process, wait and retry
                    time.sleep(0.1)
                except OSError as e:
                    # Handle other OS-level errors
                    self.logger.warning(f"OS error during file locking: {e}")
                    time.sleep(0.1)

            if not lock_acquired:
                raise FileLockError(
                    f"Failed to acquire file system lock for {file_path}",
                    file_path,
                    self.config.file_lock_timeout_seconds,
                )

            self.logger.debug(f"File lock acquired for {file_path} in mode {mode}")
            yield file_handle

        except Exception as e:
            # Close file handle if we opened it but failed to lock
            if file_handle:
                try:
                    file_handle.close()
                except:
                    pass
            raise

        finally:
            # Release file system lock and close file
            if file_handle:
                try:
                    fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
                except:
                    # Ignore errors during unlock - file will be unlocked when closed
                    pass
                try:
                    file_handle.close()
                except:
                    pass

            # Release thread lock
            file_lock.release()
            self.logger.debug(f"File lock released for {file_path}")

    def validate_csv_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of the CSV file.

        Returns:
            Dict[str, Any]: Validation results
        """
        csv_path = self.config.get_csv_file_path()
        results = {
            "file_exists": csv_path.exists(),
            "file_readable": False,
            "file_writable": False,
            "header_valid": False,
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "validation_issues": [],
            "file_size_bytes": 0,
        }

        try:
            if not results["file_exists"]:
                results["validation_issues"].append("CSV file does not exist")
                return results

            # Check file permissions
            results["file_readable"] = os.access(csv_path, os.R_OK)
            results["file_writable"] = os.access(csv_path, os.W_OK)
            results["file_size_bytes"] = csv_path.stat().st_size

            if not results["file_readable"]:
                results["validation_issues"].append("CSV file is not readable")
                return results

            # Validate file contents
            with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)

                # Validate headers
                try:
                    headers = next(reader)
                    header_issues = self.schema.validate_headers(headers)
                    results["header_valid"] = len(header_issues) == 0
                    if header_issues:
                        results["validation_issues"].extend(header_issues)
                except StopIteration:
                    results["validation_issues"].append("CSV file is empty")
                    return results

                # Validate data rows
                for row_num, row in enumerate(reader, start=2):
                    results["total_rows"] += 1

                    row_issues = self.schema.validate_row(row)
                    if row_issues:
                        results["invalid_rows"] += 1
                        results["validation_issues"].extend(
                            [f"Row {row_num}: {issue}" for issue in row_issues]
                        )
                    else:
                        results["valid_rows"] += 1

            self.logger.info(
                f"CSV validation completed: {results['valid_rows']}/{results['total_rows']} valid rows"
            )

        except Exception as e:
            results["validation_issues"].append(f"Validation failed: {e}")
            self.logger.error(f"CSV validation error: {e}")

        return results

    def append_transactions_safe(
        self, transactions: List[TokenTransaction]
    ) -> Dict[str, Any]:
        """
        Safely append multiple transactions with data preservation and ordering.

        This method ensures that existing data is preserved, new transactions are
        added in chronological order, and data integrity is maintained throughout
        the operation.

        Args:
            transactions: List of transactions to append

        Returns:
            Dict[str, Any]: Results of the append operation including success count
        """
        if not transactions:
            return {
                "success": True,
                "transactions_added": 0,
                "transactions_failed": 0,
                "existing_data_preserved": True,
                "chronological_order_maintained": True,
                "validation_passed": True,
                "issues": [],
            }

        results = {
            "success": False,
            "transactions_added": 0,
            "transactions_failed": 0,
            "existing_data_preserved": False,
            "chronological_order_maintained": False,
            "validation_passed": False,
            "issues": [],
        }

        try:
            with self._write_lock:
                # Read existing transactions to preserve data and check ordering
                existing_transactions = []
                csv_path = self.config.get_csv_file_path()

                if csv_path.exists():
                    try:
                        existing_transactions = self.read_transactions()
                        self.logger.debug(
                            f"Found {len(existing_transactions)} existing transactions"
                        )
                    except Exception as e:
                        results["issues"].append(f"Failed to read existing data: {e}")
                        return results

                # Sort new transactions chronologically
                sorted_transactions = sorted(transactions, key=lambda t: t.timestamp)

                # Validate chronological ordering with existing data
                if existing_transactions and sorted_transactions:
                    last_existing_time = existing_transactions[-1].timestamp
                    first_new_time = sorted_transactions[0].timestamp

                    # Allow some tolerance for concurrent operations
                    time_tolerance = timedelta(seconds=5)
                    if first_new_time < (last_existing_time - time_tolerance):
                        results["issues"].append(
                            f"New transactions would break chronological order: "
                            f"last existing {last_existing_time}, first new {first_new_time}"
                        )
                        # Still proceed but note the issue
                        results["chronological_order_maintained"] = False
                    else:
                        results["chronological_order_maintained"] = True
                else:
                    results["chronological_order_maintained"] = True

                # Validate data integrity before writing
                validation_issues = []
                for i, transaction in enumerate(sorted_transactions):
                    try:
                        # Validate transaction can be serialized/deserialized
                        csv_row = transaction.to_csv_row()
                        TokenTransaction.from_csv_row(csv_row)

                        # Validate against schema
                        if self.config.enable_validation:
                            row_issues = self.schema.validate_row(csv_row)
                            if row_issues:
                                validation_issues.extend(
                                    [
                                        f"Transaction {i}: {issue}"
                                        for issue in row_issues
                                    ]
                                )
                    except Exception as e:
                        validation_issues.append(
                            f"Transaction {i} validation failed: {e}"
                        )

                if validation_issues:
                    results["issues"].extend(validation_issues)
                    results["validation_passed"] = False
                    return results
                else:
                    results["validation_passed"] = True

                # Create backup before modifying if enabled
                backup_created = False
                if self.config.backup_enabled and csv_path.exists():
                    backup_path = self.create_backup()
                    if backup_path:
                        backup_created = True
                        self.logger.debug(
                            f"Backup created before append: {backup_path}"
                        )

                # Append transactions one by one with error handling
                successful_writes = 0
                failed_writes = 0

                for transaction in sorted_transactions:
                    try:
                        if self._write_transaction_locked(transaction):
                            successful_writes += 1
                        else:
                            failed_writes += 1
                            results["issues"].append(
                                f"Failed to write transaction: {transaction.get_summary()}"
                            )
                    except Exception as e:
                        failed_writes += 1
                        results["issues"].append(f"Write error: {e}")

                results["transactions_added"] = successful_writes
                results["transactions_failed"] = failed_writes

                # Verify data preservation by checking total count
                if csv_path.exists():
                    try:
                        final_transactions = self.read_transactions()
                        expected_count = len(existing_transactions) + successful_writes
                        actual_count = len(final_transactions)

                        if actual_count >= expected_count:
                            results["existing_data_preserved"] = True
                        else:
                            results["existing_data_preserved"] = False
                            results["issues"].append(
                                f"Data loss detected: expected {expected_count}, got {actual_count}"
                            )
                    except Exception as e:
                        results["issues"].append(
                            f"Failed to verify data preservation: {e}"
                        )

                # Overall success if all transactions written and data preserved
                results["success"] = (
                    failed_writes == 0
                    and results["existing_data_preserved"]
                    and results["validation_passed"]
                )

                self.logger.info(
                    f"Append operation completed: {successful_writes} added, "
                    f"{failed_writes} failed, data preserved: {results['existing_data_preserved']}"
                )

        except Exception as e:
            results["issues"].append(f"Append operation failed: {e}")
            self.logger.error(f"Safe append failed: {e}")

        return results

    def verify_data_integrity_after_append(
        self, original_count: int, added_count: int
    ) -> Dict[str, Any]:
        """
        Verify data integrity after append operation.

        Args:
            original_count: Number of transactions before append
            added_count: Number of transactions that should have been added

        Returns:
            Dict[str, Any]: Integrity verification results
        """
        results = {
            "integrity_verified": False,
            "expected_count": original_count + added_count,
            "actual_count": 0,
            "chronological_order_valid": False,
            "no_duplicates": False,
            "all_data_valid": False,
            "issues": [],
        }

        try:
            # Read all transactions
            all_transactions = self.read_transactions()
            results["actual_count"] = len(all_transactions)

            # Check count
            if results["actual_count"] != results["expected_count"]:
                results["issues"].append(
                    f"Count mismatch: expected {results['expected_count']}, "
                    f"got {results['actual_count']}"
                )

            # Check chronological order
            if len(all_transactions) > 1:
                timestamps = [t.timestamp for t in all_transactions]
                sorted_timestamps = sorted(timestamps)
                results["chronological_order_valid"] = timestamps == sorted_timestamps

                if not results["chronological_order_valid"]:
                    results["issues"].append(
                        "Transactions are not in chronological order"
                    )
            else:
                # Single transaction or no transactions are always in chronological order
                results["chronological_order_valid"] = True

            # Check for duplicates (by session_id + agent_execution_id + timestamp)
            seen_keys = set()
            duplicate_count = 0
            for transaction in all_transactions:
                key = (
                    transaction.session_id,
                    transaction.agent_execution_id,
                    transaction.timestamp.isoformat(),
                )
                if key in seen_keys:
                    duplicate_count += 1
                else:
                    seen_keys.add(key)

            results["no_duplicates"] = duplicate_count == 0
            if duplicate_count > 0:
                results["issues"].append(
                    f"Found {duplicate_count} duplicate transactions"
                )

            # Validate all data
            validation_errors = 0
            for i, transaction in enumerate(all_transactions):
                try:
                    # Test round-trip
                    csv_row = transaction.to_csv_row()
                    TokenTransaction.from_csv_row(csv_row)
                except Exception as e:
                    validation_errors += 1
                    if validation_errors <= 5:  # Limit error reporting
                        results["issues"].append(
                            f"Transaction {i} validation error: {e}"
                        )

            results["all_data_valid"] = validation_errors == 0
            if validation_errors > 0:
                results["issues"].append(
                    f"Found {validation_errors} invalid transactions"
                )

            # Overall integrity check
            results["integrity_verified"] = (
                results["actual_count"] == results["expected_count"]
                and results["chronological_order_valid"]
                and results["no_duplicates"]
                and results["all_data_valid"]
            )

        except Exception as e:
            results["issues"].append(f"Integrity verification failed: {e}")
            self.logger.error(f"Data integrity verification error: {e}")

        return results

    def _check_disk_space(self, transaction: TokenTransaction) -> None:
        """Check if there's enough disk space for the transaction."""
        csv_path = self.config.get_csv_file_path()

        try:
            # Estimate space needed (rough calculation)
            transaction_size = (
                len(str(transaction.to_csv_row())) + 100
            )  # Buffer for CSV formatting

            # Get available disk space
            stat = os.statvfs(csv_path.parent)
            available_space = stat.f_bavail * stat.f_frsize

            if available_space < transaction_size:
                raise DiskSpaceError(
                    f"Insufficient disk space for transaction",
                    transaction_size,
                    available_space,
                )

        except DiskSpaceError:
            raise
        except Exception as e:
            # If we can't check disk space, log warning but continue
            self.logger.warning(f"Could not check disk space: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        # Close any open file handles
        for handle in self._file_handles.values():
            try:
                handle.close()
            except:
                pass
        self._file_handles.clear()

        # Clear locks
        self._active_locks.clear()
