Development Standards and Guidelines
====================================

This document defines the comprehensive development standards for the AI Hydra project, including directory layout, code organization, file naming conventions, and project structure guidelines. These standards ensure consistency, maintainability, and scalability across the entire codebase.

This document works in conjunction with the :doc:`../architecture/ai_documentation_manager` system to provide automated documentation organization and quality assurance.

Overview
--------

The AI Hydra project follows structured development standards that promote:

- **Consistency**: Uniform code organization and naming conventions
- **Maintainability**: Clear structure that supports long-term maintenance
- **Scalability**: Architecture that grows with project complexity
- **Discoverability**: Intuitive organization for new developers
- **Automation**: Standards that support tooling and automation
- **Documentation Integration**: Seamless integration with the AI Documentation Manager system

These standards work in conjunction with the :doc:`sdlc_procedures` and the :doc:`../architecture/ai_documentation_manager` to provide a complete development framework with automated documentation management.

Directory Layout Standards
--------------------------

Project Root Structure
~~~~~~~~~~~~~~~~~~~~~

The AI Hydra project follows a standardized directory structure:

.. code-block:: text

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

Core Directory Organization
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Source Code Structure (`ai_hydra/`)**

.. code-block:: text

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

**Package Organization Rules:**

- Each major subsystem gets its own subdirectory
- Related functionality is grouped together
- Public APIs are exposed through ``__init__.py`` files
- Private modules use leading underscore naming

**Testing Structure (`tests/`)**

.. code-block:: text

   tests/
   ├── __init__.py                   # Test package initialization
   ├── conftest.py                   # Shared test configuration
   ├── unit/                         # Unit tests
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

**Testing Organization Rules:**

- Group tests by type (unit, property, integration, e2e)
- Name test files with ``test_`` prefix
- Mirror source code structure in test organization
- Use descriptive names for test scenarios

**Development Tools Structure (`tools/`)**

.. code-block:: text

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

**Tools Organization Rules:**

- Group tools by purpose (debug, testing, documentation)
- Make scripts executable where appropriate
- Include clear documentation for each tool
- Provide usage examples in docstrings

File Naming Conventions
----------------------

Python Files
~~~~~~~~~~~

**Modules and Packages:**

.. code-block:: python

   # Good: snake_case for modules
   game_logic.py
   neural_network.py
   token_tracker.py
   
   # Good: PascalCase for classes
   class GameBoard:
       pass
   
   class TokenTracker:
       pass
   
   # Good: snake_case for functions
   def execute_move(board, move):
       pass
   
   def collect_metadata():
       pass
   
   # Good: UPPER_SNAKE_CASE for constants
   MAX_BUDGET = 100
   DEFAULT_GRID_SIZE = (20, 20)
   
   # Good: Leading underscore for private
   def _internal_method(self):
       pass

Test Files
~~~~~~~~~

**Test File Naming:**

.. code-block:: text

   # Unit tests
   test_game_logic.py
   test_neural_network.py
   test_budget_controller.py
   
   # Property-based tests
   test_deterministic_reproducibility.py
   test_budget_management.py
   test_clone_independence.py
   
   # Integration tests
   test_simulation_pipeline_integration.py
   test_zmq_communication_integration.py
   
   # End-to-end tests
   test_complete_simulation_e2e.py
   test_user_workflow_e2e.py

Documentation Files
~~~~~~~~~~~~~~~~~~

**Documentation Naming:**

.. code-block:: text

   # RST files: snake_case
   getting_started.rst
   api_reference.rst
   development_standards.rst
   
   # Markdown files: kebab-case
   directory-layout-standards.md
   testing-standards.md
   git-conventions.md
   
   # Images: descriptive with project prefix
   ai_hydra_architecture.png
   token_tracker_flow_diagram.svg

Configuration Files
~~~~~~~~~~~~~~~~~~

**Configuration Naming:**

.. code-block:: text

   # Kiro specs: kebab-case directories
   .kiro/specs/token-tracker/
   .kiro/specs/neural-network/
   
   # Steering files: kebab-case
   .kiro/steering/testing-standards.md
   .kiro/steering/code-style.md
   
   # Hook files: kebab-case
   .kiro/hooks/token-tracking.kiro.hook
   .kiro/hooks/auto-commit.kiro.hook

Scripts and Tools
~~~~~~~~~~~~~~~~

**Script Naming:**

.. code-block:: text

   # Shell scripts: snake_case
   update_version.sh
   organize_files.sh
   run_tests.sh
   
   # Python tools: snake_case
   run_tests.py
   debug_file_patterns.py
   validate_docs.py
   
   # All scripts should be executable
   chmod +x scripts/*.sh
   chmod +x tools/*/*.py

Directory Creation Guidelines
----------------------------

When to Create New Directories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Create a new directory when:**

- You have 3+ related files that form a logical group
- The functionality represents a distinct subsystem
- The files have different access patterns or audiences
- The grouping improves navigation and understanding

**Don't create directories for:**

- Single files or very small groups (< 3 files)
- Temporary or experimental code
- Files that are closely coupled to existing modules

Directory Naming Rules
~~~~~~~~~~~~~~~~~~~~~

1. **Use descriptive names**: Directory names should clearly indicate their purpose
2. **Follow language conventions**: Use ``snake_case`` for Python packages, ``kebab-case`` for specs
3. **Avoid abbreviations**: Use full words unless the abbreviation is widely understood
4. **Be consistent**: Follow established patterns within the project
5. **Consider hierarchy**: Deeper nesting should represent more specific functionality

**Examples of Good Directory Structure:**

.. code-block:: text

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

**Examples of Poor Directory Structure:**

.. code-block:: text

   # Bad: Unnecessary nesting
   ai_hydra/
   ├── token/                     # Too generic
   │   └── tracker/               # Unnecessary nesting
   │       └── impl/              # Over-engineered
   │           └── tracker.py     # Finally the actual code

Code Organization Standards
--------------------------

Import Organization
~~~~~~~~~~~~~~~~~~

**Standard Import Order:**

.. code-block:: python

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

Module Structure
~~~~~~~~~~~~~~~

**Standard Module Template:**

.. code-block:: python

   """
   Module docstring explaining purpose and usage.
   
   This module provides functionality for [specific purpose].
   Key components include [list main classes/functions].
   
   Example:
       >>> from ai_hydra.module import MainClass
       >>> instance = MainClass()
       >>> result = instance.main_method()
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

Class Structure
~~~~~~~~~~~~~~

**Standard Class Organization:**

.. code-block:: python

   class HydraMgr:
       """Main orchestrator for the hybrid neural network + tree search system."""
       
       # Class constants
       DEFAULT_BUDGET = 100
       MAX_DECISION_CYCLES = 1000
       
       def __init__(self, config: SimulationConfig):
           """Initialize with configuration validation."""
           # Public attributes
           self.config = config
           self.master_game = MasterGame(config)
           
           # Private attributes
           self._budget_controller = BudgetController(config.move_budget)
           self._state_manager = StateManager()
           self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
           
           # Initialize components
           self._setup_neural_network()
           self._setup_logging()
       
       # Public interface methods
       def run_simulation(self) -> SimulationResult:
           """Run complete simulation."""
           pass
       
       def execute_decision_cycle(self) -> DecisionResult:
           """Execute single decision cycle."""
           pass
       
       # Private implementation methods
       def _setup_neural_network(self) -> None:
           """Initialize neural network components."""
           pass
       
       def _execute_tree_search(self, initial_move: Move) -> TreeSearchResult:
           """Execute tree search starting from initial move."""
           pass
       
       # Properties for computed values
       @property
       def current_score(self) -> int:
           """Get current game score."""
           return self.master_game.get_score()

Configuration Management
-----------------------

Configuration File Organization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Root-Level Configuration Files:**

.. code-block:: text

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

**Kiro IDE Configuration (`.kiro/`)**

.. code-block:: text

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

**Configuration Organization Rules:**

- Organize specs by feature name (kebab-case)
- Use consistent file naming for spec documents
- Group related steering documents together
- Maintain clear separation between specs and configuration

Migration and Maintenance
------------------------

File Migration Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

**When reorganizing existing files:**

1. **Plan the move**: Identify all affected imports and references
2. **Update imports**: Modify all import statements to reflect new locations
3. **Update documentation**: Change any documentation references
4. **Test thoroughly**: Ensure all tests pass after the move
5. **Commit atomically**: Make the move in a single, well-documented commit

**Backward Compatibility:**

When moving public APIs:

.. code-block:: python

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

Regular Maintenance Tasks
~~~~~~~~~~~~~~~~~~~~~~~~

**Quarterly Maintenance:**

.. code-block:: bash

   # Review directory structure
   python tools/debug/analyze_structure.py
   
   # Clean up empty directories
   find . -type d -empty -delete
   
   # Update documentation
   python tools/documentation/update_structure_docs.py
   
   # Validate import paths
   python tools/debug/check_imports.py
   
   # Monitor directory growth
   python tools/debug/directory_metrics.py

Quality Metrics
~~~~~~~~~~~~~~

**Monitor these metrics to ensure good directory organization:**

- **Directory depth**: Avoid nesting deeper than 4 levels
- **Files per directory**: Keep directories under 20 files when possible
- **Import complexity**: Minimize complex relative imports
- **Test coverage**: Ensure test structure mirrors source structure
- **Documentation coverage**: All directories should have clear purpose

Automation and Tooling
----------------------

Directory Validation
~~~~~~~~~~~~~~~~~~~

**Automated Structure Validation:**

.. code-block:: bash

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

**Import Path Validation:**

.. code-block:: python

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

File Organization Tools
~~~~~~~~~~~~~~~~~~~~~~

**Automated File Organization:**

The project includes ``scripts/organize_files.sh`` for automated file organization:

.. code-block:: bash

   # Preview file organization changes
   ./scripts/organize_files.sh --dry-run
   
   # Apply file organization changes
   ./scripts/organize_files.sh --fix
   
   # Generate organization report
   ./scripts/organize_files.sh --report

**Tool Features:**

- Detects misplaced files in root directory
- Suggests proper locations based on file type and content
- Validates directory structure compliance
- Checks executable permissions on scripts
- Generates comprehensive organization reports

Integration with Development Workflow
------------------------------------

Git Integration
~~~~~~~~~~~~~~

**Automated Git Commits:**

When using Kiro IDE, file modifications trigger automated commits following these standards:

.. code-block:: bash

   # Standard commit categories
   feat(scope): New feature implementation
   fix(scope): Bug fix or error correction
   docs(scope): Documentation updates
   test(scope): Test additions or modifications
   refactor(scope): Code restructuring
   chore(scope): Build or maintenance tasks

**Pre-commit Hooks:**

.. code-block:: yaml

   # .pre-commit-config.yaml
   repos:
     - repo: local
       hooks:
         - id: validate-structure
           name: Validate directory structure
           entry: ./scripts/validate_structure.sh
           language: system
           pass_filenames: false
         
         - id: organize-files
           name: Check file organization
           entry: ./scripts/organize_files.sh --dry-run
           language: system
           pass_filenames: false

IDE Integration
~~~~~~~~~~~~~~

**Kiro IDE Configuration:**

.. code-block:: json

   // .kiro/settings.json
   {
     "file_organization": {
       "auto_organize": true,
       "validate_on_save": true,
       "enforce_naming_conventions": true
     },
     "directory_standards": {
       "max_depth": 4,
       "max_files_per_directory": 20,
       "enforce_test_structure": true
     }
   }

**VS Code Configuration (Optional):**

.. code-block:: json

   // .vscode/settings.json
   {
     "files.exclude": {
       "**/__pycache__": true,
       "**/.pytest_cache": true,
       "**/.hypothesis": true,
       "**/htmlcov": true
     },
     "python.defaultInterpreterPath": "./venv/bin/python",
     "python.testing.pytestEnabled": true,
     "python.testing.pytestArgs": ["tests/"]
   }

Best Practices Summary
---------------------

Directory Organization
~~~~~~~~~~~~~~~~~~~~~

1. **Follow established patterns**: Use the standardized directory structure
2. **Group related functionality**: Keep related files together
3. **Use descriptive names**: Directory and file names should be self-explanatory
4. **Maintain shallow hierarchies**: Avoid deep nesting (max 4 levels)
5. **Separate concerns**: Keep different types of files in appropriate directories

File Naming
~~~~~~~~~~

1. **Be consistent**: Follow naming conventions throughout the project
2. **Use appropriate case**: snake_case for Python, kebab-case for configs
3. **Be descriptive**: Names should clearly indicate purpose
4. **Follow language conventions**: Respect Python and other language standards
5. **Avoid abbreviations**: Use full words unless widely understood

Code Organization
~~~~~~~~~~~~~~~~

1. **Structure modules consistently**: Follow the standard module template
2. **Organize imports properly**: Standard library, third-party, local
3. **Group related code**: Keep related functions and classes together
4. **Use appropriate visibility**: Public vs private interfaces
5. **Document everything**: Comprehensive docstrings and comments

Maintenance
~~~~~~~~~~

1. **Regular reviews**: Quarterly structure reviews and cleanup
2. **Automated validation**: Use tools to enforce standards
3. **Migration planning**: Careful planning for structural changes
4. **Backward compatibility**: Maintain compatibility during transitions
5. **Documentation updates**: Keep documentation current with structure

These development standards ensure that the AI Hydra project maintains a clean, organized, and scalable structure that supports both current development needs and future growth while integrating seamlessly with the established SDLC procedures.