# Directory Layout Standards

## Overview

This document defines the standard directory structure and file organization for the AI Hydra project. Following these standards ensures consistency, maintainability, and ease of navigation across the codebase.

## Project Root Structure

```
ai_hydra/                           # Main project root
├── .git/                          # Git version control
├── .github/                       # GitHub workflows and templates
├── .kiro/                         # Kiro IDE configuration and specs
│   ├── hooks/                     # Agent hooks configuration
│   ├── specs/                     # Feature specifications
│   └── steering/                  # Development guidelines and standards
├── .pytest_cache/                 # Pytest cache (auto-generated)
├── .hypothesis/                   # Hypothesis test data (auto-generated)
├── .vscode/                       # VS Code configuration (optional)
├── ai_hydra/                      # Main Python package
├── ai_snake_lab/                  # Legacy snake game implementation
├── docs/                          # Documentation source and build
├── examples/                      # Usage examples and demos
├── htmlcov/                       # Coverage reports (auto-generated)
├── images/                        # Project images and assets
├── scripts/                       # Build and maintenance scripts
├── tests/                         # Test suite
├── tools/                         # Development tools and utilities
└── [config files]                 # Root-level configuration files
```

## Core Directories

### Source Code Organization

#### `ai_hydra/` - Main Package
```
ai_hydra/
├── __init__.py                    # Package initialization and version
├── cli.py                         # Command-line interface
├── config.py                      # Configuration management
├── models.py                      # Core data models
├── [component_modules].py         # Individual component modules
├── token_tracker/                 # Token tracking subsystem
│   ├── __init__.py
│   ├── tracker.py                # Main tracker implementation
│   ├── models.py                 # Token tracker data models
│   ├── csv_writer.py             # CSV file operations
│   ├── metadata_collector.py     # Metadata collection
│   ├── error_handler.py          # Error handling
│   ├── backup_manager.py         # Backup operations
│   ├── monitoring.py             # System monitoring
│   ├── maintenance.py            # Maintenance tasks
│   └── hook.py                   # Kiro IDE integration
└── tui/                          # Terminal user interface
    ├── __init__.py
    ├── client.py                 # TUI client implementation
    ├── game_board.py             # Game board display
    └── hydra_client.tcss         # TUI styling
```

**Package Organization Rules:**
- Each major subsystem gets its own subdirectory
- Related functionality is grouped together
- Public APIs are exposed through `__init__.py` files
- Private modules use leading underscore naming

#### `ai_snake_lab/` - Legacy Implementation
```
ai_snake_lab/
├── ai/                           # AI components
├── constants/                    # Game constants
├── game/                         # Game mechanics
├── hydra/                        # Hydra-specific logic
├── server/                       # Server components
├── ui/                           # User interface
└── utils/                        # Utility functions
```

**Legacy Code Rules:**
- Maintain existing structure for backward compatibility
- New features should be added to `ai_hydra/` package
- Gradual migration path from legacy to new implementation

### Testing Organization

#### `tests/` - Test Suite
```
tests/
├── __init__.py                   # Test package initialization
├── conftest.py                   # Shared test configuration
├── unit/                         # Unit tests
│   ├── __pycache__/             # Python cache (auto-generated)
│   ├── test_[component].py      # Component-specific unit tests
│   └── test_[module].py         # Module-specific unit tests
├── property/                     # Property-based tests
│   ├── test_[property_name].py  # Property test implementations
│   └── test_[component]_properties.py
├── integration/                  # Integration tests
│   ├── test_[integration_scenario].py
│   └── test_[component]_integration.py
├── e2e/                         # End-to-end tests
│   └── test_[workflow].py       # Complete workflow tests
└── [individual_test_files].py   # Standalone test files
```

**Testing Organization Rules:**
- Group tests by type (unit, property, integration, e2e)
- Name test files with `test_` prefix
- Mirror source code structure in test organization
- Use descriptive names for test scenarios

### Documentation Structure

#### `docs/` - Documentation
```
docs/
├── _source/                      # Sphinx source files
│   ├── architecture/            # Architecture documentation
│   ├── end_user/                # End-user guides
│   ├── runbook/                 # Operational procedures
│   ├── conf.py                  # Sphinx configuration
│   └── index.rst               # Main documentation index
├── _build/                      # Generated documentation (auto-generated)
├── _static/                     # Static assets (images, CSS)
├── Makefile                     # Documentation build commands
├── make.bat                     # Windows build commands
└── requirements.txt             # Documentation dependencies
```

**Documentation Rules:**
- Use RST format for Sphinx compatibility
- Organize by audience (architecture, end-user, runbook)
- Include comprehensive API documentation
- Maintain cross-references between documents

### Development Tools

#### `tools/` - Development Utilities
```
tools/
├── debug/                       # Debugging utilities
│   ├── debug_file_patterns.py  # File pattern debugging
│   └── [debug_scripts].py      # Other debugging tools
├── testing/                     # Testing utilities
│   ├── run_tests.py            # Test runner script
│   └── [test_tools].py        # Testing helper tools
├── documentation/               # Documentation tools
│   ├── test_documentation.py   # Documentation validation
│   ├── validate_docs.py        # Simple doc validation
│   └── [doc_tools].py         # Documentation utilities
└── [other_categories]/         # Additional tool categories
```

**Tools Organization Rules:**
- Group tools by purpose (debug, testing, documentation)
- Make scripts executable where appropriate
- Include clear documentation for each tool
- Provide usage examples in docstrings

#### `scripts/` - Build and Maintenance Scripts
```
scripts/
├── update_version.sh           # Version update automation
├── build.sh                   # Build automation
├── deploy.sh                  # Deployment scripts
└── [maintenance_scripts].sh   # Other maintenance tasks
```

**Scripts Rules:**
- Use shell scripts for system-level operations
- Make all scripts executable (`chmod +x`)
- Include usage documentation in script headers
- Follow consistent naming conventions

### Configuration and Metadata

#### `.kiro/` - Kiro IDE Configuration
```
.kiro/
├── hooks/                      # Agent hooks configuration
│   └── [hook_name].kiro.hook  # Individual hook configurations
├── specs/                      # Feature specifications
│   └── [feature_name]/        # Feature-specific directories
│       ├── requirements.md    # Feature requirements
│       ├── design.md         # Feature design
│       └── tasks.md          # Implementation tasks
├── steering/                   # Development guidelines
│   ├── [standard_name].md    # Individual standards documents
│   └── directory-layout-standards.md  # This document
└── [config_files]            # Other Kiro configuration
```

**Kiro Configuration Rules:**
- Organize specs by feature name (kebab-case)
- Use consistent file naming for spec documents
- Group related steering documents together
- Maintain clear separation between specs and configuration

#### Root-Level Configuration Files
```
# Build and Package Management
pyproject.toml                  # Python project configuration
poetry.lock                     # Poetry dependency lock file
requirements.txt                # Pip requirements
requirements-docs.txt           # Documentation requirements
MANIFEST.in                     # Package manifest

# Version Control
.gitignore                      # Git ignore patterns

# Documentation and Metadata
README.md                       # Project overview
CHANGELOG.md                    # Version history
LICENSE                         # Project license
CNAME                          # GitHub Pages domain
_config.yml                    # Jekyll configuration

# CI/CD and Quality
.readthedocs.yaml              # Read the Docs configuration
.coverage                      # Coverage configuration
```

## File Naming Conventions

### Python Files
- **Modules**: `snake_case.py` (e.g., `game_logic.py`)
- **Classes**: `PascalCase` (e.g., `GameBoard`, `TokenTracker`)
- **Functions**: `snake_case` (e.g., `execute_move`, `collect_metadata`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_BUDGET`, `DEFAULT_GRID_SIZE`)
- **Private**: Leading underscore (e.g., `_internal_method`)

### Test Files
- **Unit tests**: `test_[component].py` (e.g., `test_game_logic.py`)
- **Property tests**: `test_[property_name].py` (e.g., `test_deterministic_reproducibility.py`)
- **Integration tests**: `test_[scenario]_integration.py`
- **End-to-end tests**: `test_[workflow]_e2e.py`

### Documentation Files
- **RST files**: `snake_case.rst` (e.g., `getting_started.rst`)
- **Markdown files**: `kebab-case.md` (e.g., `directory-layout-standards.md`)
- **Images**: Descriptive names with project prefix (e.g., `ai_hydra_architecture.png`)

### Configuration Files
- **Kiro specs**: Feature directories use `kebab-case` (e.g., `token-tracker`)
- **Steering files**: `kebab-case.md` (e.g., `testing-standards.md`)
- **Hook files**: `kebab-case.kiro.hook` (e.g., `token-tracking.kiro.hook`)

### Scripts and Tools
- **Shell scripts**: `snake_case.sh` (e.g., `update_version.sh`)
- **Python tools**: `snake_case.py` (e.g., `run_tests.py`)
- **Executable scripts**: Include shebang and make executable

## Directory Creation Guidelines

### When to Create New Directories

**Create a new directory when:**
- You have 3+ related files that form a logical group
- The functionality represents a distinct subsystem
- The files have different access patterns or audiences
- The grouping improves navigation and understanding

**Don't create directories for:**
- Single files or very small groups (< 3 files)
- Temporary or experimental code
- Files that are closely coupled to existing modules

### Directory Naming Rules

1. **Use descriptive names**: Directory names should clearly indicate their purpose
2. **Follow language conventions**: Use `snake_case` for Python packages, `kebab-case` for specs
3. **Avoid abbreviations**: Use full words unless the abbreviation is widely understood
4. **Be consistent**: Follow established patterns within the project
5. **Consider hierarchy**: Deeper nesting should represent more specific functionality

### Examples of Good Directory Structure

```python
# Good: Clear hierarchy and purpose
ai_hydra/
├── token_tracker/              # Subsystem with multiple components
│   ├── models.py              # Data models
│   ├── tracker.py             # Main implementation
│   ├── csv_writer.py          # File operations
│   └── error_handler.py       # Error handling
└── neural_network/            # Another subsystem
    ├── models.py              # NN models
    ├── training.py            # Training logic
    └── inference.py           # Inference engine

# Bad: Unnecessary nesting
ai_hydra/
├── token/                     # Too generic
│   └── tracker/               # Unnecessary nesting
│       └── impl/              # Over-engineered
│           └── tracker.py     # Finally the actual code
```

## File Organization Best Practices

### Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import torch
import pandas as pd
from hypothesis import given

# Local imports
from .models import TokenTransaction
from ..config import TrackerConfig
```

### Module Structure
```python
"""
Module docstring explaining purpose and usage.
"""

# Module-level constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

# Type definitions
TransactionId = str
Timestamp = float

# Main implementation classes
class ComponentName:
    """Class implementing main functionality."""
    pass

# Utility functions
def helper_function() -> None:
    """Helper function for internal use."""
    pass

# Public API functions
def public_api_function() -> None:
    """Public function exposed to users."""
    pass
```

### Configuration File Organization
```python
# Group related configuration together
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "ai_hydra"

class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
class AppConfig:
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()
```

## Migration Guidelines

### Moving Existing Files

When reorganizing existing files:

1. **Plan the move**: Identify all affected imports and references
2. **Update imports**: Modify all import statements to reflect new locations
3. **Update documentation**: Change any documentation references
4. **Test thoroughly**: Ensure all tests pass after the move
5. **Commit atomically**: Make the move in a single, well-documented commit

### Backward Compatibility

When moving public APIs:

1. **Provide import aliases**: Keep old import paths working temporarily
2. **Add deprecation warnings**: Warn users about the upcoming change
3. **Update examples**: Modify all example code to use new paths
4. **Document the change**: Add migration notes to CHANGELOG.md

### Example Migration
```python
# Old location: ai_hydra/utils.py
# New location: ai_hydra/token_tracker/utils.py

# In ai_hydra/__init__.py - provide backward compatibility
from .token_tracker.utils import TokenUtils
import warnings

# Deprecated import path
def get_token_utils():
    warnings.warn(
        "ai_hydra.utils.TokenUtils is deprecated. "
        "Use ai_hydra.token_tracker.utils.TokenUtils instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return TokenUtils
```

## Maintenance and Evolution

### Regular Maintenance Tasks

1. **Review directory structure** quarterly for optimization opportunities
2. **Clean up empty directories** and unused files
3. **Update documentation** to reflect structural changes
4. **Validate import paths** and fix any broken references
5. **Monitor directory growth** and split large directories when needed

### Evolution Guidelines

As the project grows:

1. **Maintain consistency** with established patterns
2. **Document changes** in this standards document
3. **Consider impact** on existing code and users
4. **Plan migrations** carefully to minimize disruption
5. **Seek feedback** from team members on structural changes

### Quality Metrics

Monitor these metrics to ensure good directory organization:

- **Directory depth**: Avoid nesting deeper than 4 levels
- **Files per directory**: Keep directories under 20 files when possible
- **Import complexity**: Minimize complex relative imports
- **Test coverage**: Ensure test structure mirrors source structure
- **Documentation coverage**: All directories should have clear purpose

## Tools and Automation

### Directory Validation Script

```bash
#!/bin/bash
# validate_structure.sh - Validate project directory structure

# Check for required directories
required_dirs=("ai_hydra" "tests" "docs" "scripts" "tools")
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "❌ Missing required directory: $dir"
        exit 1
    fi
done

# Check for proper test organization
if [ ! -d "tests/unit" ] || [ ! -d "tests/property" ]; then
    echo "❌ Test directories not properly organized"
    exit 1
fi

echo "✅ Directory structure validation passed"
```

### Import Path Checker

```python
#!/usr/bin/env python3
"""Check for broken import paths after directory reorganization."""

import ast
import sys
from pathlib import Path

def check_imports(file_path: Path) -> list:
    """Check imports in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return imports
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

# Usage: python check_imports.py
```

This directory layout standard ensures that the AI Hydra project maintains a clean, organized, and scalable structure that supports both current development needs and future growth.