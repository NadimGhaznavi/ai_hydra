"""
Property-based tests for Token Tracker concurrent access safety.
"""

import pytest
import threading
import time
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from hypothesis import given, strategies as st, settings, assume
from typing import List, Dict, Any

from ai_hydra.token_tracker.models import TokenTransaction, TrackerConfig
from ai_hydra.token_tracker.csv_writer import CSVWriter
from ai_hydra.token_tracker.error_handler import TokenTrackerErrorHandler


class TestTokenTrackerConcurrentAccessSafety:
    """Property-based tests for concurrent access safety."""

    @given(
        num_threads=st.integers(min_value=2, max_value=8),
        transactions_per_thread=st.integers(min_value=1, max_value=10),
        use_queuing=st.booleans(),
        enable_process_locks=st.booleans(),
    )
    @settings(max_examples=20, deadline=10000)
    def test_concurrent_access_safety_property(
        self, num_threads, transactions_per_thread, use_queuing, enable_process_locks
    ):
        """
        **Feature: kiro-token-tracker, Property 3: Concurrent Access Safety**
        **Validates: Requirements 1.5, 6.5**

        For any set of concurrent token tracking operations, the CSV file should
        maintain data integrity with no lost transactions or corrupted data.
        """
        # Create temporary CSV file for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_concurrent_access.csv"

            # Create test configuration with concurrent access features
            config = TrackerConfig(
                csv_file_path=csv_path,
                backup_enabled=False,  # Disable backup for simpler testing
                enable_validation=True,
                file_lock_timeout_seconds=5.0,
                max_concurrent_writes=num_threads,
                enable_transaction_queuing=use_queuing,
                queue_max_size=num_threads * transactions_per_thread * 2,
                deadlock_detection_enabled=True,
                max_lock_wait_time_seconds=10.0,
                error_backoff_enabled=True,
                process_lock_enabled=enable_process_locks,
            )

            # Create CSV writer
            error_handler = TokenTrackerErrorHandler()
            csv_writer = CSVWriter(config, error_handler)

            try:
                # Generate transactions for each thread
                all_transactions = []
                thread_transactions = {}

                for thread_id in range(num_threads):
                    thread_transactions[thread_id] = []
                    for i in range(transactions_per_thread):
                        transaction = TokenTransaction.create_new(
                            prompt_text=f"Thread {thread_id} transaction {i}",
                            tokens_used=100 + thread_id * 10 + i,
                            elapsed_time=1.0 + (thread_id * 0.1) + (i * 0.01),
                            workspace_folder=f"workspace_{thread_id}",
                            hook_trigger_type="agentExecutionCompleted",
                            hook_name=f"test_hook_{thread_id}",
                            session_id=f"session_{thread_id}",
                            agent_execution_id=f"exec_{thread_id}_{i}",
                        )
                        thread_transactions[thread_id].append(transaction)
                        all_transactions.append(transaction)

                # Track results from each thread
                write_results = {}
                thread_errors = {}

                def write_transactions_for_thread(thread_id: int) -> Dict[str, Any]:
                    """Write transactions for a specific thread."""
                    results = {
                        "thread_id": thread_id,
                        "successful_writes": 0,
                        "failed_writes": 0,
                        "errors": [],
                        "write_times": [],
                    }

                    for i, transaction in enumerate(thread_transactions[thread_id]):
                        try:
                            start_time = time.time()

                            # Use queued write if enabled, otherwise direct write
                            if use_queuing and hasattr(
                                csv_writer, "write_transaction_queued"
                            ):
                                success = csv_writer.write_transaction_queued(
                                    transaction
                                )
                            else:
                                success = csv_writer.write_transaction(transaction)

                            write_time = time.time() - start_time
                            results["write_times"].append(write_time)

                            if success:
                                results["successful_writes"] += 1
                            else:
                                results["failed_writes"] += 1
                                results["errors"].append(
                                    f"Write failed for transaction {i}"
                                )

                        except Exception as e:
                            results["failed_writes"] += 1
                            results["errors"].append(
                                f"Exception in transaction {i}: {e}"
                            )

                    return results

                # Execute concurrent writes using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    # Submit all thread tasks
                    future_to_thread = {
                        executor.submit(
                            write_transactions_for_thread, thread_id
                        ): thread_id
                        for thread_id in range(num_threads)
                    }

                    # Collect results as they complete
                    for future in as_completed(future_to_thread):
                        thread_id = future_to_thread[future]
                        try:
                            result = future.result(
                                timeout=30
                            )  # 30 second timeout per thread
                            write_results[thread_id] = result
                        except Exception as e:
                            thread_errors[thread_id] = str(e)

                # If using queuing, wait for queue to be processed
                if use_queuing and hasattr(csv_writer, "_transaction_queue"):
                    # Wait for queue to be empty (with timeout)
                    wait_start = time.time()
                    while (
                        csv_writer._transaction_queue.qsize() > 0
                        and time.time() - wait_start < 10.0
                    ):
                        time.sleep(0.1)

                # Verify no thread had critical errors
                assert (
                    len(thread_errors) == 0
                ), f"Thread errors occurred: {thread_errors}"

                # Calculate total expected successful writes
                total_expected_writes = sum(
                    result["successful_writes"] for result in write_results.values()
                )
                total_failed_writes = sum(
                    result["failed_writes"] for result in write_results.values()
                )

                # At least 80% of writes should succeed (allowing for some contention)
                min_expected_success = int(len(all_transactions) * 0.8)
                assert total_expected_writes >= min_expected_success, (
                    f"Too many failed writes: {total_failed_writes} failed, "
                    f"{total_expected_writes} succeeded out of {len(all_transactions)} total"
                )

                # Verify CSV file integrity
                assert (
                    csv_path.exists()
                ), "CSV file should exist after concurrent writes"

                # Read back all transactions
                retrieved_transactions = csv_writer.read_transactions()

                # Should have retrieved at least the successful writes
                assert len(retrieved_transactions) >= total_expected_writes, (
                    f"Retrieved {len(retrieved_transactions)} transactions, "
                    f"expected at least {total_expected_writes}"
                )

                # Verify no data corruption - all retrieved transactions should be valid
                for i, transaction in enumerate(retrieved_transactions):
                    try:
                        # Test round-trip to ensure data integrity
                        csv_row = transaction.to_csv_row()
                        reconstructed = TokenTransaction.from_csv_row(csv_row)

                        # Verify essential fields are preserved
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
                        pytest.fail(f"Data corruption detected in transaction {i}: {e}")

                # Verify CSV file structure integrity
                csv_integrity = csv_writer.validate_csv_integrity()
                assert csv_integrity["file_exists"], "CSV file should exist"
                assert csv_integrity["header_valid"], "CSV headers should be valid"
                assert csv_integrity["invalid_rows"] == 0, (
                    f"Found {csv_integrity['invalid_rows']} invalid rows, "
                    f"issues: {csv_integrity['validation_issues']}"
                )

                # Verify no duplicate transactions (by unique execution ID)
                execution_ids = [t.agent_execution_id for t in retrieved_transactions]
                unique_execution_ids = set(execution_ids)
                assert len(execution_ids) == len(unique_execution_ids), (
                    f"Duplicate transactions detected: {len(execution_ids)} total, "
                    f"{len(unique_execution_ids)} unique"
                )

                # Verify concurrent access statistics if available
                if hasattr(csv_writer, "get_concurrent_access_stats"):
                    stats = csv_writer.get_concurrent_access_stats()

                    # Should not have excessive errors
                    assert (
                        stats["concurrent_error_count"] < num_threads
                    ), f"Too many concurrent errors: {stats['concurrent_error_count']}"

                    # Active writers should be reasonable
                    assert stats["active_writers"] <= stats["max_concurrent_writers"]

                # Performance validation - writes shouldn't take too long
                all_write_times = []
                for result in write_results.values():
                    all_write_times.extend(result["write_times"])

                if all_write_times:
                    avg_write_time = sum(all_write_times) / len(all_write_times)
                    max_write_time = max(all_write_times)

                    # Average write time should be reasonable (under 2 seconds)
                    assert (
                        avg_write_time < 2.0
                    ), f"Average write time too high: {avg_write_time:.2f}s"

                    # No single write should take more than 10 seconds
                    assert (
                        max_write_time < 10.0
                    ), f"Maximum write time too high: {max_write_time:.2f}s"

            finally:
                # Cleanup
                if hasattr(csv_writer, "_shutdown_queue_processor"):
                    csv_writer._shutdown_queue_processor()
                if csv_path.exists():
                    csv_path.unlink()

    @given(
        num_readers=st.integers(min_value=2, max_value=4),
        num_writers=st.integers(min_value=1, max_value=3),
        transactions_per_writer=st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=10, deadline=15000)
    def test_concurrent_read_write_safety_property(
        self, num_readers, num_writers, transactions_per_writer
    ):
        """
        **Feature: kiro-token-tracker, Property 3: Concurrent Access Safety**
        **Validates: Requirements 1.5, 6.5**

        For any combination of concurrent read and write operations, the system
        should maintain data consistency without corruption or deadlocks.
        """
        # Create temporary CSV file for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_read_write_concurrent.csv"

            # Create test configuration
            config = TrackerConfig(
                csv_file_path=csv_path,
                backup_enabled=False,
                enable_validation=True,
                file_lock_timeout_seconds=8.0,
                max_concurrent_writes=num_writers + 1,
                enable_transaction_queuing=True,
                queue_max_size=num_writers * transactions_per_writer * 2,
                deadlock_detection_enabled=True,
                max_lock_wait_time_seconds=15.0,
                error_backoff_enabled=True,
                process_lock_enabled=True,
            )

            # Create CSV writer
            error_handler = TokenTrackerErrorHandler()
            csv_writer = CSVWriter(config, error_handler)

            try:
                # Pre-populate with some initial data
                initial_transactions = []
                for i in range(5):
                    transaction = TokenTransaction.create_new(
                        prompt_text=f"Initial transaction {i}",
                        tokens_used=50 + i,
                        elapsed_time=0.5 + i * 0.1,
                        workspace_folder="initial_workspace",
                        hook_trigger_type="agentExecutionCompleted",
                        hook_name="initial_hook",
                    )
                    initial_transactions.append(transaction)
                    csv_writer.write_transaction(transaction)

                # Verify initial state
                initial_count = len(csv_writer.read_transactions())
                assert initial_count == 5

                # Track results
                reader_results = {}
                writer_results = {}
                operation_errors = {}

                def reader_task(reader_id: int) -> Dict[str, Any]:
                    """Continuously read transactions."""
                    results = {
                        "reader_id": reader_id,
                        "read_operations": 0,
                        "successful_reads": 0,
                        "read_counts": [],
                        "errors": [],
                    }

                    # Read for about 3 seconds
                    start_time = time.time()
                    while time.time() - start_time < 3.0:
                        try:
                            transactions = csv_writer.read_transactions()
                            results["read_operations"] += 1
                            results["successful_reads"] += 1
                            results["read_counts"].append(len(transactions))

                            # Verify data integrity of read transactions
                            for transaction in transactions:
                                assert transaction.tokens_used >= 0
                                assert transaction.elapsed_time >= 0.0
                                assert len(transaction.workspace_folder) > 0

                        except Exception as e:
                            results["errors"].append(str(e))

                        # Small delay between reads
                        time.sleep(0.1)

                    return results

                def writer_task(writer_id: int) -> Dict[str, Any]:
                    """Write transactions concurrently."""
                    results = {
                        "writer_id": writer_id,
                        "successful_writes": 0,
                        "failed_writes": 0,
                        "errors": [],
                    }

                    for i in range(transactions_per_writer):
                        try:
                            transaction = TokenTransaction.create_new(
                                prompt_text=f"Writer {writer_id} transaction {i}",
                                tokens_used=100 + writer_id * 10 + i,
                                elapsed_time=1.0 + writer_id * 0.1 + i * 0.01,
                                workspace_folder=f"writer_workspace_{writer_id}",
                                hook_trigger_type="agentExecutionCompleted",
                                hook_name=f"writer_hook_{writer_id}",
                                session_id=f"writer_session_{writer_id}",
                                agent_execution_id=f"writer_exec_{writer_id}_{i}",
                            )

                            if csv_writer.write_transaction(transaction):
                                results["successful_writes"] += 1
                            else:
                                results["failed_writes"] += 1

                        except Exception as e:
                            results["failed_writes"] += 1
                            results["errors"].append(str(e))

                        # Small delay between writes
                        time.sleep(0.05)

                    return results

                # Execute concurrent read/write operations
                with ThreadPoolExecutor(
                    max_workers=num_readers + num_writers
                ) as executor:
                    # Submit reader tasks
                    reader_futures = {
                        executor.submit(reader_task, reader_id): reader_id
                        for reader_id in range(num_readers)
                    }

                    # Submit writer tasks
                    writer_futures = {
                        executor.submit(writer_task, writer_id): writer_id
                        for writer_id in range(num_writers)
                    }

                    # Collect reader results
                    for future in as_completed(reader_futures):
                        reader_id = reader_futures[future]
                        try:
                            result = future.result(timeout=10)
                            reader_results[reader_id] = result
                        except Exception as e:
                            operation_errors[f"reader_{reader_id}"] = str(e)

                    # Collect writer results
                    for future in as_completed(writer_futures):
                        writer_id = writer_futures[future]
                        try:
                            result = future.result(timeout=10)
                            writer_results[writer_id] = result
                        except Exception as e:
                            operation_errors[f"writer_{writer_id}"] = str(e)

                # Verify no critical errors occurred
                assert (
                    len(operation_errors) == 0
                ), f"Operation errors: {operation_errors}"

                # Verify readers were able to read successfully
                total_read_operations = sum(
                    result["read_operations"] for result in reader_results.values()
                )
                total_successful_reads = sum(
                    result["successful_reads"] for result in reader_results.values()
                )

                assert total_read_operations > 0, "No read operations were performed"
                read_success_rate = total_successful_reads / total_read_operations
                assert (
                    read_success_rate >= 0.8
                ), f"Read success rate too low: {read_success_rate:.2f}"

                # Verify writers were able to write successfully
                total_successful_writes = sum(
                    result["successful_writes"] for result in writer_results.values()
                )
                total_expected_writes = num_writers * transactions_per_writer

                write_success_rate = total_successful_writes / total_expected_writes
                assert write_success_rate >= 0.7, (
                    f"Write success rate too low: {write_success_rate:.2f} "
                    f"({total_successful_writes}/{total_expected_writes})"
                )

                # Verify final data integrity
                final_transactions = csv_writer.read_transactions()
                expected_final_count = initial_count + total_successful_writes

                assert len(final_transactions) == expected_final_count, (
                    f"Final count mismatch: got {len(final_transactions)}, "
                    f"expected {expected_final_count}"
                )

                # Verify read consistency - readers should have seen increasing counts
                for reader_id, result in reader_results.items():
                    read_counts = result["read_counts"]
                    if len(read_counts) > 1:
                        # Counts should generally increase or stay the same (monotonic)
                        non_decreasing_count = 0
                        for i in range(1, len(read_counts)):
                            if read_counts[i] >= read_counts[i - 1]:
                                non_decreasing_count += 1

                        # At least 80% of reads should show non-decreasing counts
                        consistency_rate = non_decreasing_count / (len(read_counts) - 1)
                        assert (
                            consistency_rate >= 0.8
                        ), f"Reader {reader_id} saw inconsistent counts: {read_counts}"

                # Verify no data corruption in final state
                csv_integrity = csv_writer.validate_csv_integrity()
                assert csv_integrity["file_exists"]
                assert csv_integrity["header_valid"]
                assert csv_integrity["invalid_rows"] == 0

            finally:
                # Cleanup
                if hasattr(csv_writer, "_shutdown_queue_processor"):
                    csv_writer._shutdown_queue_processor()
                if csv_path.exists():
                    csv_path.unlink()

    @given(
        lock_timeout=st.floats(
            min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False
        ),
        max_wait_time=st.floats(
            min_value=2.0, max_value=8.0, allow_nan=False, allow_infinity=False
        ),
        num_contending_threads=st.integers(min_value=3, max_value=6),
    )
    @settings(max_examples=10, deadline=12000)
    def test_deadlock_prevention_property(
        self, lock_timeout, max_wait_time, num_contending_threads
    ):
        """
        **Feature: kiro-token-tracker, Property 3: Concurrent Access Safety**
        **Validates: Requirements 1.5, 6.5**

        For any configuration with potential for lock contention, the system
        should prevent deadlocks and handle timeout scenarios gracefully.
        """
        assume(max_wait_time > lock_timeout)  # Ensure max wait is longer than timeout

        # Create temporary CSV file for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_deadlock_prevention.csv"

            # Create configuration that could lead to contention
            config = TrackerConfig(
                csv_file_path=csv_path,
                backup_enabled=False,
                enable_validation=True,
                file_lock_timeout_seconds=lock_timeout,
                max_concurrent_writes=2,  # Limit to force contention
                enable_transaction_queuing=False,  # Disable queuing to test direct locking
                deadlock_detection_enabled=True,
                max_lock_wait_time_seconds=max_wait_time,
                error_backoff_enabled=True,
                process_lock_enabled=True,
            )

            # Create CSV writer
            error_handler = TokenTrackerErrorHandler()
            csv_writer = CSVWriter(config, error_handler)

            try:
                # Track results and timing
                thread_results = {}
                start_time = time.time()

                def contending_writer_task(thread_id: int) -> Dict[str, Any]:
                    """Writer task that may encounter lock contention."""
                    results = {
                        "thread_id": thread_id,
                        "operations_attempted": 0,
                        "operations_successful": 0,
                        "timeout_errors": 0,
                        "deadlock_errors": 0,
                        "other_errors": 0,
                        "total_time": 0.0,
                    }

                    thread_start = time.time()

                    # Attempt multiple operations that could cause contention
                    for i in range(5):
                        results["operations_attempted"] += 1

                        try:
                            transaction = TokenTransaction.create_new(
                                prompt_text=f"Contending thread {thread_id} op {i}",
                                tokens_used=200 + thread_id * 10 + i,
                                elapsed_time=2.0 + thread_id * 0.1,
                                workspace_folder=f"contend_workspace_{thread_id}",
                                hook_trigger_type="agentExecutionCompleted",
                                hook_name=f"contend_hook_{thread_id}",
                                session_id=f"contend_session_{thread_id}",
                                agent_execution_id=f"contend_exec_{thread_id}_{i}",
                            )

                            if csv_writer.write_transaction(transaction):
                                results["operations_successful"] += 1
                            else:
                                results["other_errors"] += 1

                        except Exception as e:
                            error_msg = str(e).lower()
                            if (
                                "timeout" in error_msg
                                or "failed to acquire" in error_msg
                            ):
                                results["timeout_errors"] += 1
                            elif "deadlock" in error_msg:
                                results["deadlock_errors"] += 1
                            else:
                                results["other_errors"] += 1

                        # Small delay to create more contention opportunities
                        time.sleep(0.1)

                    results["total_time"] = time.time() - thread_start
                    return results

                # Execute contending threads
                with ThreadPoolExecutor(max_workers=num_contending_threads) as executor:
                    futures = {
                        executor.submit(contending_writer_task, thread_id): thread_id
                        for thread_id in range(num_contending_threads)
                    }

                    for future in as_completed(futures):
                        thread_id = futures[future]
                        try:
                            result = future.result(timeout=max_wait_time + 5.0)
                            thread_results[thread_id] = result
                        except Exception as e:
                            # If a thread completely fails, that's also a valid outcome
                            # as long as it doesn't hang indefinitely
                            thread_results[thread_id] = {
                                "thread_id": thread_id,
                                "operations_attempted": 0,
                                "operations_successful": 0,
                                "timeout_errors": 0,
                                "deadlock_errors": 0,
                                "other_errors": 1,
                                "total_time": max_wait_time + 5.0,
                                "executor_error": str(e),
                            }

                total_execution_time = time.time() - start_time

                # Verify no deadlocks occurred (system should complete within reasonable time)
                max_reasonable_time = max_wait_time * 2 + 10.0  # Allow some overhead
                assert total_execution_time < max_reasonable_time, (
                    f"Execution took too long ({total_execution_time:.2f}s), "
                    f"possible deadlock. Max reasonable: {max_reasonable_time:.2f}s"
                )

                # Verify all threads completed (no infinite hangs)
                assert (
                    len(thread_results) == num_contending_threads
                ), f"Not all threads completed: {len(thread_results)}/{num_contending_threads}"

                # Analyze results
                total_operations = sum(
                    r["operations_attempted"] for r in thread_results.values()
                )
                total_successful = sum(
                    r["operations_successful"] for r in thread_results.values()
                )
                total_timeouts = sum(
                    r["timeout_errors"] for r in thread_results.values()
                )
                total_deadlocks = sum(
                    r["deadlock_errors"] for r in thread_results.values()
                )
                total_other_errors = sum(
                    r["other_errors"] for r in thread_results.values()
                )

                # At least some operations should have been attempted
                assert total_operations > 0, "No operations were attempted"

                # Deadlock detection should prevent actual deadlocks
                # (We allow timeout errors as they're a valid way to handle contention)
                assert total_deadlocks == 0, f"Deadlocks detected: {total_deadlocks}"

                # At least some operations should succeed (system shouldn't be completely blocked)
                min_success_rate = 0.3  # Allow for high contention
                success_rate = (
                    total_successful / total_operations if total_operations > 0 else 0
                )
                assert success_rate >= min_success_rate, (
                    f"Success rate too low: {success_rate:.2f} "
                    f"({total_successful}/{total_operations})"
                )

                # Verify timeout handling is working (should see some timeouts under contention)
                if num_contending_threads > config.max_concurrent_writes:
                    # With more threads than allowed concurrent writes, we expect some timeouts
                    assert (
                        total_timeouts > 0 or total_successful == total_operations
                    ), "Expected either timeouts or all operations to succeed under contention"

                # Verify data integrity despite contention
                if total_successful > 0:
                    final_transactions = csv_writer.read_transactions()
                    assert (
                        len(final_transactions) == total_successful
                    ), f"Transaction count mismatch: {len(final_transactions)} != {total_successful}"

                    # Verify no corruption
                    csv_integrity = csv_writer.validate_csv_integrity()
                    assert csv_integrity["file_exists"]
                    assert csv_integrity["header_valid"]
                    assert csv_integrity["invalid_rows"] == 0

            finally:
                # Cleanup
                if hasattr(csv_writer, "_shutdown_queue_processor"):
                    csv_writer._shutdown_queue_processor()
                if csv_path.exists():
                    csv_path.unlink()
