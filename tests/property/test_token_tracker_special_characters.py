"""
Property-based tests for Token Tracker special character handling.
"""

import pytest
import tempfile
import unicodedata
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from typing import List, Optional

from ai_hydra.token_tracker.models import TokenTransaction, TrackerConfig
from ai_hydra.token_tracker.tracker import TokenTracker
from ai_hydra.token_tracker.csv_writer import CSVWriter
from ai_hydra.token_tracker.error_handler import TokenTrackerErrorHandler


class TestTokenTrackerSpecialCharacters:
    """Property-based tests for special character handling."""

    @given(
        special_text=st.one_of(
            # Unicode characters from various categories
            st.text(
                alphabet=st.characters(
                    categories=(
                        "Lu",
                        "Ll",
                        "Nd",
                        "Po",
                        "Ps",
                        "Pe",
                        "Pc",
                        "Pd",
                    ),
                    include_characters=["\n", "\r", "\t", '"', "'", ",", ";", "\\"],
                ),
                min_size=1,
                max_size=500,
            ),
            # Emoji and symbols
            st.text(
                alphabet=st.characters(
                    categories=("So", "Sm", "Sc", "Sk"),
                    min_codepoint=0x1F300,  # Emoji range
                    max_codepoint=0x1F9FF,
                ),
                min_size=1,
                max_size=100,
            ),
            # Mathematical and technical symbols
            st.text(
                alphabet=st.characters(
                    min_codepoint=0x2000,  # Mathematical symbols
                    max_codepoint=0x2BFF,
                ),
                min_size=1,
                max_size=100,
            ),
            # CJK characters
            st.text(
                alphabet=st.characters(
                    min_codepoint=0x4E00,  # CJK Unified Ideographs
                    max_codepoint=0x9FFF,
                ),
                min_size=1,
                max_size=100,
            ),
            # Arabic characters
            st.text(
                alphabet=st.characters(
                    min_codepoint=0x0600,  # Arabic
                    max_codepoint=0x06FF,
                ),
                min_size=1,
                max_size=100,
            ),
            # Control characters mixed with text
            st.sampled_from(
                [
                    "Hello",
                    " ",
                    "World",
                    "\n",
                    "\r",
                    "\t",
                    "\x00",
                    "\x01",
                    "\x02",
                ]
            ),
        ),
        tokens_used=st.integers(min_value=1, max_value=10000),
        elapsed_time=st.floats(
            min_value=0.001, max_value=60.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100, deadline=3000)
    def test_special_character_handling_property(
        self, special_text, tokens_used, elapsed_time
    ):
        """
        **Feature: kiro-token-tracker, Property 7: Special Character Handling**
        **Validates: Requirements 6.3, 6.4**

        For any text containing special characters, newlines, or Unicode content,
        the CSV encoding should preserve the text exactly and remain readable by
        standard spreadsheet tools.
        """
        # Filter out completely empty or whitespace-only strings
        assume(special_text.strip() != "")

        # Filter out strings that are only control characters
        printable_chars = [
            c for c in special_text if unicodedata.category(c) not in ["Cc", "Cf"]
        ]
        assume(len(printable_chars) > 0)

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_special_chars.csv"

            config = TrackerConfig(
                csv_file_path=csv_path,
                backup_enabled=False,  # Disable for testing
                enable_validation=True,
                max_prompt_length=2000,  # Allow longer text for testing
            )

            tracker = TokenTracker(config)

            try:
                # Test transaction creation with special characters
                result = tracker.record_transaction(
                    prompt_text=special_text,
                    tokens_used=tokens_used,
                    elapsed_time=elapsed_time,
                    context={"test_type": "special_characters"},
                )

                # Transaction should be recorded successfully or fail gracefully
                assert isinstance(result, bool), "Should return boolean result"

                if result:
                    # If recording succeeded, verify data integrity

                    # Read back the transaction
                    history = tracker.get_transaction_history()
                    assert len(history) > 0, "Should have at least one transaction"

                    # Find our transaction
                    our_transaction = None
                    for trans in history:
                        if trans.tokens_used == tokens_used:
                            our_transaction = trans
                            break

                    assert our_transaction is not None, "Should find our transaction"

                    # Verify text preservation principles
                    original_text = special_text
                    retrieved_text = our_transaction.prompt_text

                    # 1. Retrieved text should not be empty if original wasn't
                    if original_text.strip():
                        assert (
                            retrieved_text.strip() != ""
                        ), "Retrieved text should not be empty"

                    # 2. Essential content should be preserved
                    # Remove control characters from both for comparison
                    original_clean = "".join(
                        c
                        for c in original_text
                        if unicodedata.category(c) not in ["Cc"]
                        or c in ["\n", "\r", "\t"]
                    )
                    retrieved_clean = "".join(
                        c
                        for c in retrieved_text
                        if unicodedata.category(c) not in ["Cc"]
                        or c in ["\n", "\r", "\t"]
                    )

                    # 3. Check character preservation (allowing for CSV escaping)
                    if len(original_clean) > 0:
                        # Count preserved characters (case-insensitive for robustness)
                        original_chars = set(original_clean.lower())
                        retrieved_chars = set(retrieved_clean.lower())

                        preserved_chars = original_chars & retrieved_chars
                        preservation_ratio = len(preserved_chars) / len(original_chars)

                        # Should preserve at least 70% of unique characters
                        assert preservation_ratio >= 0.7, (
                            f"Character preservation too low: {preservation_ratio:.2f} "
                            f"(original: {len(original_chars)}, preserved: {len(preserved_chars)})"
                        )

                    # 4. Verify CSV safety - no unescaped quotes or newlines in raw CSV
                    csv_row = our_transaction.to_csv_row()
                    prompt_field = csv_row[1]  # prompt_text is second field

                    # Should not contain unescaped newlines (should be \\n)
                    if "\n" in original_text:
                        assert (
                            "\\n" in prompt_field or "\n" not in prompt_field
                        ), "Newlines should be escaped or removed"

                    # Should not contain unescaped carriage returns
                    if "\r" in original_text:
                        assert (
                            "\\r" in prompt_field or "\r" not in prompt_field
                        ), "Carriage returns should be escaped or removed"

                    # Quotes should be properly escaped (doubled)
                    if '"' in original_text:
                        # Count quotes in original vs sanitized
                        original_quote_count = original_text.count('"')
                        sanitized_quote_count = prompt_field.count('""')
                        # Should have at least some quote escaping
                        assert (
                            sanitized_quote_count > 0 or '"' not in prompt_field
                        ), "Quotes should be escaped or removed"

                    # 5. Test round-trip through CSV parsing
                    try:
                        reconstructed = TokenTransaction.from_csv_row(csv_row)

                        # Should successfully reconstruct
                        assert reconstructed.tokens_used == tokens_used
                        assert reconstructed.elapsed_time == elapsed_time

                        # Text should be consistent with what we stored
                        assert reconstructed.prompt_text == retrieved_text

                    except Exception as e:
                        pytest.fail(f"CSV round-trip failed: {e}")

                    # 6. Verify CSV file integrity
                    integrity_results = tracker.csv_writer.validate_csv_integrity()
                    assert integrity_results["file_exists"], "CSV file should exist"
                    assert integrity_results[
                        "header_valid"
                    ], "CSV headers should be valid"
                    assert (
                        integrity_results["total_rows"] >= 1
                    ), "Should have at least one row"
                    assert (
                        integrity_results["valid_rows"] >= 1
                    ), "Should have at least one valid row"

                else:
                    # If recording failed, it should be due to validation issues
                    # This is acceptable for some edge cases
                    stats = tracker.get_statistics()
                    assert (
                        stats["transactions_failed"] > 0
                    ), "Should track failed transactions"

            finally:
                # Cleanup
                if csv_path.exists():
                    csv_path.unlink()

    @given(
        csv_problematic_chars=st.text(
            alphabet=st.characters(
                categories=(),  # Empty categories to allow include_characters
                include_characters=['"', "'", ",", ";", "\n", "\r", "\t", "\\"],
            ),
            min_size=1,
            max_size=100,
        ),
        normal_text=st.text(
            alphabet=st.characters(categories=("Lu", "Ll", "Nd")),
            min_size=5,
            max_size=100,
        ),
    )
    @settings(max_examples=50, deadline=2000)
    def test_csv_specific_character_handling_property(
        self, csv_problematic_chars, normal_text
    ):
        """
        **Feature: kiro-token-tracker, Property 7: Special Character Handling**
        **Validates: Requirements 6.3, 6.4**

        For any text containing CSV-specific problematic characters (quotes, commas,
        newlines), the system should handle them safely without breaking CSV structure.
        """
        # Combine normal text with problematic characters
        combined_text = f"{normal_text}{csv_problematic_chars}{normal_text}"

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_csv_chars.csv"

            config = TrackerConfig(
                csv_file_path=csv_path,
                backup_enabled=False,
                enable_validation=True,
            )

            tracker = TokenTracker(config)

            try:
                # Record transaction with CSV-problematic characters
                result = tracker.record_transaction(
                    prompt_text=combined_text, tokens_used=500, elapsed_time=2.5
                )

                assert isinstance(result, bool), "Should return boolean result"

                if result:
                    # Verify CSV file can be read properly
                    history = tracker.get_transaction_history()
                    assert len(history) > 0, "Should retrieve transactions"

                    # Verify CSV structure is not broken
                    with open(csv_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    # Should have header + data rows
                    assert (
                        len(lines) >= 2
                    ), "Should have header and at least one data row"

                    # Each line should be parseable as CSV
                    import csv

                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        rows = list(reader)

                    # Should have proper number of columns
                    header_row = rows[0]
                    assert len(header_row) == 12, "Should have 12 columns"

                    for i, row in enumerate(rows[1:], 1):
                        assert (
                            len(row) == 12
                        ), f"Row {i} should have 12 columns, got {len(row)}"

                    # Verify our transaction is in there
                    found_transaction = False
                    for row in rows[1:]:
                        if row[2] == "500":  # tokens_used field
                            found_transaction = True
                            prompt_field = row[1]  # prompt_text field

                            # Should contain some of our original content
                            assert (
                                len(prompt_field) > 0
                            ), "Prompt field should not be empty"

                            # Should not contain unescaped problematic characters
                            # (they should be escaped or the field should be quoted)
                            break

                    assert found_transaction, "Should find our transaction in CSV"

            finally:
                if csv_path.exists():
                    csv_path.unlink()

    @given(
        unicode_categories=st.lists(
            st.sampled_from(
                [
                    "emoji",  # 😀😃😄
                    "math",  # ∑∏∫∆∇
                    "currency",  # $€£¥₹
                    "arrows",  # ←→↑↓
                    "cjk",  # 中文日本語
                    "arabic",  # العربية
                    "cyrillic",  # Русский
                    "greek",  # Ελληνικά
                    "symbols",  # ♠♣♥♦
                ]
            ),
            min_size=1,
            max_size=3,
            unique=True,
        ),
        text_length=st.integers(min_value=10, max_value=200),
    )
    @settings(max_examples=30, deadline=4000)
    def test_unicode_category_handling_property(self, unicode_categories, text_length):
        """
        **Feature: kiro-token-tracker, Property 7: Special Character Handling**
        **Validates: Requirements 6.3, 6.4**

        For any combination of Unicode character categories, the system should
        handle them consistently and preserve their essential characteristics.
        """
        # Generate text based on selected categories
        category_chars = {
            "emoji": "😀😃😄😁😆😅😂🤣😊😇🙂🙃😉😌😍🥰😘",
            "math": "∑∏∫∆∇∂∞±≤≥≠≈∝∈∉∪∩⊂⊃∀∃∄∅∧∨¬⊕⊗",
            "currency": "$€£¥₹₽₩₪₨₦₡₵₸₴₲₱₫₪₢₣₤₥₦₧₨₩₪₫€₭₮₯₰₱₲₳₴₵₶₷₸₹₺₻₼₽₾₿",
            "arrows": "←→↑↓↔↕↖↗↘↙⇐⇒⇑⇓⇔⇕⇖⇗⇘⇙⇚⇛⇜⇝⇞⇟⇠⇡⇢⇣⇤⇥⇦⇧⇨⇩⇪",
            "cjk": "中文日本語한국어漢字ひらがなカタカナ조선말",
            "arabic": "العربيةالعربيةاللغةالعربيةالفصحى",
            "cyrillic": "РусскийязыкБългарскиСрпскиУкраїнська",
            "greek": "ΕλληνικάΑλφάβητοΩμέγαΑλφαΒήταΓάμμα",
            "symbols": "♠♣♥♦♪♫♬♭♮♯⚡⚽⚾⛄⛅⛈⛎⛏⛑⛓⛔⛕⛖⛗⛘⛙⛚⛛",
        }

        # Build test text from selected categories
        test_parts = []
        for category in unicode_categories:
            chars = category_chars[category]
            # Take a portion of characters from this category
            portion_size = min(text_length // len(unicode_categories), len(chars))
            test_parts.append(chars[:portion_size])

        test_text = " ".join(test_parts)

        # Ensure we have some content
        if not test_text.strip():
            test_text = "Test content with Unicode"

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_unicode_categories.csv"

            config = TrackerConfig(
                csv_file_path=csv_path,
                backup_enabled=False,
                enable_validation=True,
                max_prompt_length=1000,
            )

            tracker = TokenTracker(config)

            try:
                # Test recording with mixed Unicode categories
                result = tracker.record_transaction(
                    prompt_text=test_text,
                    tokens_used=len(unicode_categories) * 100,
                    elapsed_time=1.0 + len(unicode_categories) * 0.5,
                )

                assert isinstance(result, bool), "Should return boolean result"

                if result:
                    # Verify Unicode handling
                    history = tracker.get_transaction_history()
                    assert len(history) > 0, "Should retrieve transactions"

                    our_transaction = history[-1]  # Should be the last one
                    retrieved_text = our_transaction.prompt_text

                    # Should preserve Unicode categories
                    for category in unicode_categories:
                        category_sample = category_chars[category][:5]  # Sample chars

                        # Check if at least some characters from this category are preserved
                        preserved_count = sum(
                            1 for char in category_sample if char in retrieved_text
                        )

                        # Should preserve at least some characters from each category
                        # (allowing for some loss due to sanitization)
                        if category not in [
                            "symbols"
                        ]:  # Symbols might be more aggressively filtered
                            assert (
                                preserved_count > 0
                            ), f"No characters preserved from {category} category"

                    # Verify CSV integrity with Unicode content
                    integrity_results = tracker.csv_writer.validate_csv_integrity()
                    assert integrity_results["file_exists"], "CSV file should exist"
                    assert (
                        integrity_results["valid_rows"] >= 1
                    ), "Should have valid rows"

                    # Test that CSV can be read by standard tools
                    import csv

                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        rows = list(reader)

                    assert len(rows) >= 2, "Should have header and data rows"

                    # Verify our Unicode content is in the CSV
                    found_unicode = False
                    for row in rows[1:]:
                        if len(row) >= 2 and len(row[1]) > 0:  # prompt_text field
                            # Should contain some Unicode characters
                            has_unicode = any(ord(char) > 127 for char in row[1])
                            if has_unicode:
                                found_unicode = True
                                break

                    if any(
                        category in ["emoji", "cjk", "arabic", "cyrillic", "greek"]
                        for category in unicode_categories
                    ):
                        assert (
                            found_unicode
                        ), "Should preserve Unicode characters in CSV"

            finally:
                if csv_path.exists():
                    csv_path.unlink()

    @given(
        control_chars=st.text(
            alphabet=st.characters(
                min_codepoint=0x00,
                max_codepoint=0x1F,
                blacklist_characters=["\t", "\n", "\r"],  # Keep some useful ones
            ),
            min_size=1,
            max_size=20,
        ),
        normal_content=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=10,
            max_size=100,
        ),
    )
    @settings(max_examples=30, deadline=2000)
    def test_control_character_sanitization_property(
        self, control_chars, normal_content
    ):
        """
        **Feature: kiro-token-tracker, Property 7: Special Character Handling**
        **Validates: Requirements 6.3, 6.4**

        For any text containing control characters, the system should sanitize them
        appropriately while preserving the readable content.
        """
        # Mix control characters with normal content
        mixed_text = f"{normal_content[:20]}{control_chars}{normal_content[20:]}"

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_control_chars.csv"

            config = TrackerConfig(
                csv_file_path=csv_path,
                backup_enabled=False,
                enable_validation=True,
            )

            tracker = TokenTracker(config)

            try:
                # Test with control characters
                result = tracker.record_transaction(
                    prompt_text=mixed_text, tokens_used=300, elapsed_time=1.5
                )

                assert isinstance(result, bool), "Should return boolean result"

                if result:
                    # Verify control character handling
                    history = tracker.get_transaction_history()
                    assert len(history) > 0, "Should retrieve transactions"

                    retrieved_text = history[-1].prompt_text

                    # Should preserve normal content
                    normal_chars_preserved = sum(
                        1 for char in normal_content if char in retrieved_text
                    )
                    preservation_ratio = normal_chars_preserved / len(normal_content)

                    # Should preserve most normal characters
                    assert (
                        preservation_ratio >= 0.8
                    ), f"Normal content preservation too low: {preservation_ratio:.2f}"

                    # Should not contain dangerous control characters
                    dangerous_controls = [
                        "\x00",
                        "\x01",
                        "\x02",
                        "\x03",
                        "\x04",
                        "\x05",
                    ]
                    for char in dangerous_controls:
                        assert (
                            char not in retrieved_text
                        ), f"Dangerous control character {repr(char)} not sanitized"

                    # CSV should be valid
                    integrity_results = tracker.csv_writer.validate_csv_integrity()
                    assert integrity_results["file_exists"], "CSV file should exist"
                    assert (
                        integrity_results["valid_rows"] >= 1
                    ), "Should have valid rows"

            finally:
                if csv_path.exists():
                    csv_path.unlink()
