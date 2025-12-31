#!/usr/bin/env python3
"""
Simple documentation validation script.
Checks RST syntax and cross-references without requiring Sphinx.
"""

import re
from pathlib import Path
from typing import List, Set, Tuple


def validate_rst_syntax(file_path: Path) -> List[str]:
    """Basic RST syntax validation."""
    errors = []
    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    for i, line in enumerate(lines, 1):
        # Check for common RST syntax issues
        if line.strip().startswith(".. ") and not line.strip().endswith("::"):
            if "::" not in line and not any(
                directive in line
                for directive in [
                    "code-block",
                    "note",
                    "warning",
                    "toctree",
                    "automodule",
                    "literalinclude",
                ]
            ):
                # This might be a malformed directive
                pass

        # Check for unbalanced backticks
        backtick_count = line.count("`")
        if backtick_count % 2 != 0:
            errors.append(f"Line {i}: Unbalanced backticks")

    return errors


def extract_labels_and_references(
    docs_dir: Path,
) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """Extract all labels and references from RST files."""
    labels = set()
    references = []

    for rst_file in docs_dir.rglob("*.rst"):
        content = rst_file.read_text(encoding="utf-8")

        # Extract labels (.. _label:)
        file_labels = re.findall(r"^\.\. _([^:]+):", content, re.MULTILINE)
        labels.update(file_labels)

        # Extract references (:ref:`label`)
        file_refs = re.findall(r":ref:`([^`]+)`", content)
        references.extend([(ref, rst_file.name) for ref in file_refs])

    return labels, references


def validate_cross_references(docs_dir: Path) -> List[str]:
    """Validate cross-references in documentation."""
    errors = []
    labels, references = extract_labels_and_references(docs_dir)

    # Standard Sphinx references that are automatically generated
    standard_refs = {"genindex", "modindex", "search"}

    for ref, filename in references:
        if ref not in labels and ref not in standard_refs:
            errors.append(f"Broken reference in {filename}: :ref:`{ref}`")

    return errors


def main():
    """Main validation function."""
    docs_source = Path("docs/_source")

    if not docs_source.exists():
        print("❌ Documentation source directory not found")
        return 1

    print("🔍 Validating documentation...")

    # Validate RST syntax
    syntax_errors = []
    rst_files = list(docs_source.rglob("*.rst"))

    for rst_file in rst_files:
        file_errors = validate_rst_syntax(rst_file)
        if file_errors:
            syntax_errors.extend([f"{rst_file.name}: {error}" for error in file_errors])

    # Validate cross-references
    ref_errors = validate_cross_references(docs_source)

    # Report results
    total_errors = len(syntax_errors) + len(ref_errors)

    if syntax_errors:
        print(f"❌ RST Syntax Errors ({len(syntax_errors)}):")
        for error in syntax_errors:
            print(f"  - {error}")

    if ref_errors:
        print(f"❌ Cross-reference Errors ({len(ref_errors)}):")
        for error in ref_errors:
            print(f"  - {error}")

    if total_errors == 0:
        print("✅ Documentation validation passed!")
        print(f"📄 Validated {len(rst_files)} RST files")
        return 0
    else:
        print(f"❌ Documentation validation failed with {total_errors} errors")
        return 1


if __name__ == "__main__":
    exit(main())
