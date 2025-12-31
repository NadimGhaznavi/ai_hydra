#!/usr/bin/env python3
"""
Debug script to test file patterns preservation in metadata collection.
"""

import sys
import tempfile
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, ".")

from ai_hydra.token_tracker.metadata_collector import MetadataCollector
from ai_hydra.token_tracker.error_handler import TokenTrackerErrorHandler
from ai_hydra.token_tracker.tracker import TokenTracker
from ai_hydra.token_tracker.models import TrackerConfig


def test_file_patterns_preservation():
    """Test that file patterns are preserved through the metadata collection chain."""

    print("Testing file patterns preservation...")

    # Create test context with file patterns
    test_context = {
        "trigger_type": "agentExecutionCompleted",
        "hook_name": "test-hook",
        "file_patterns": ["*.py", "*.md", "*.txt"],
    }

    print(f"Input context: {test_context}")

    # Test 1: MetadataCollector directly
    print("\n1. Testing MetadataCollector.get_hook_context()...")
    error_handler = TokenTrackerErrorHandler()
    collector = MetadataCollector(error_handler)

    hook_context = collector.get_hook_context(test_context)
    print(f"Hook context result: {hook_context}")

    if "file_patterns" in hook_context:
        print(f"✓ file_patterns found: {hook_context['file_patterns']}")
        if hook_context["file_patterns"] == test_context["file_patterns"]:
            print("✓ file_patterns match input")
        else:
            print(
                f"✗ file_patterns don't match: expected {test_context['file_patterns']}, got {hook_context['file_patterns']}"
            )
    else:
        print("✗ file_patterns not found in hook context")

    # Test 2: Full metadata collection
    print("\n2. Testing MetadataCollector.collect_execution_metadata()...")
    metadata = collector.collect_execution_metadata(test_context)
    print(f"Full metadata keys: {list(metadata.keys())}")

    if "file_patterns" in metadata:
        print(f"✓ file_patterns found in metadata: {metadata['file_patterns']}")
        if metadata["file_patterns"] == test_context["file_patterns"]:
            print("✓ file_patterns match input in full metadata")
        else:
            print(
                f"✗ file_patterns don't match in metadata: expected {test_context['file_patterns']}, got {metadata['file_patterns']}"
            )
    else:
        print("✗ file_patterns not found in full metadata")

    # Test 3: TokenTracker integration
    print("\n3. Testing TokenTracker integration...")
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "test_integration.csv"

        config = TrackerConfig(
            csv_file_path=csv_path,
            backup_enabled=False,
            enable_validation=True,
            file_lock_timeout_seconds=2.0,
        )

        tracker = TokenTracker(config)

        # Record transaction
        result = tracker.record_transaction(
            prompt_text="test prompt",
            tokens_used=100,
            elapsed_time=1.5,
            context=test_context,
        )

        print(f"Transaction recording result: {result}")

        if result:
            # Get transaction history
            transactions = tracker.get_transaction_history()
            if transactions:
                transaction = transactions[0]
                print(f"Transaction file_patterns: {transaction.file_patterns}")

                if transaction.file_patterns == test_context["file_patterns"]:
                    print("✓ file_patterns preserved in transaction")
                    return True
                else:
                    print(
                        f"✗ file_patterns not preserved: expected {test_context['file_patterns']}, got {transaction.file_patterns}"
                    )
                    return False
            else:
                print("✗ No transactions found")
                return False
        else:
            print("✗ Transaction recording failed")
            return False


if __name__ == "__main__":
    success = test_file_patterns_preservation()
    if success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
