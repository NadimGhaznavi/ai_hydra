---
inclusion: always
---

# Documentation Standards

This document defines the documentation standards and practices for the AI Hydra project to ensure consistency, accuracy, and maintainability across all documentation.

## Documentation Philosophy

### Core Principles

- **Accuracy First**: Documentation must accurately reflect the current codebase and architecture
- **User-Centric**: Write for the intended audience (developers, researchers, operators)
- **Maintainable**: Documentation should be easy to update and keep in sync with code changes
- **Comprehensive**: Cover all public APIs, architectural decisions, and operational procedures
- **Testable**: Documentation should include examples that can be validated

### Documentation Types

1. **API Documentation**: Auto-generated from docstrings using Sphinx
2. **Architecture Documentation**: High-level system design and component interactions
3. **User Guides**: Step-by-step instructions for common tasks
4. **Developer Guides**: Setup, contribution guidelines, and development workflows
5. **Operational Documentation**: Deployment, monitoring, and troubleshooting guides

## File Organization

### Directory Structure
```
docs/
├── _source/                    # Sphinx source files
│   ├── api_reference.rst      # Auto-generated API docs
│   ├── architecture.rst       # System architecture
│   ├── decision_flow.rst      # Decision flow architecture
│   ├── getting_started.rst    # Quick start guide
│   ├── deployment.rst         # Deployment instructions
│   ├── testing.rst           # Testing documentation
│   ├── troubleshooting.rst   # Common issues and solutions
│   └── index.rst             # Main table of contents
├── _build/                    # Generated documentation
└── _static/                   # Static assets (images, CSS)
```

### Naming Conventions

- **RST Files**: Use snake_case (e.g., `getting_started.rst`)
- **Images**: Use descriptive names with project prefix (e.g., `ai_hydra_architecture.png`)
- **Code Examples**: Include in dedicated `examples/` directory when substantial

## RST Style Guidelines

### Document Structure

```rst
Document Title
==============

Brief introduction paragraph explaining the document's purpose and scope.

Section Title
-------------

Content with proper formatting...

Subsection Title
~~~~~~~~~~~~~~~~

More detailed content...

Sub-subsection Title
^^^^^^^^^^^^^^^^^^^^

Specific implementation details...
```

### Code Blocks

**Always use proper syntax:**

```rst
.. code-block:: python

    class ExampleClass:
        def method_example(self) -> str:
            """Method with proper docstring."""
            return "example"
```

**Include language specification:**
- Use `python` for Python code
- Use `bash` for shell commands
- Use `yaml` for configuration files
- Use `json` for JSON examples

### Cross-References

**Internal References:**
```rst
See :ref:`architecture-overview` for system design details.
```

**External References:**
```rst
For more information, see the `PyTorch Documentation <https://pytorch.org/docs/>`_.
```

### Lists and Tables

**Use consistent formatting:**

```rst
**Requirements:**

* Python 3.11 or higher
* PyTorch 2.0+
* ZeroMQ 4.3+

**Configuration Options:**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Type
     - Description
   * - grid_size
     - Tuple[int, int]
     - Game board dimensions (width, height)
```

## Content Standards

### API Documentation

**Docstring Format (Google Style):**

```python
def execute_decision_cycle(self, budget: int = 100) -> DecisionResult:
    """
    Execute a complete decision cycle with budget constraints.
    
    This method orchestrates the entire decision-making process including
    neural network prediction, tree search exploration, path evaluation,
    and master game state updates.
    
    Args:
        budget: Maximum number of moves to explore (default: 100)
        
    Returns:
        DecisionResult: Contains the selected move, paths evaluated,
            budget consumed, and performance metrics
            
    Raises:
        ValueError: If budget is less than 1
        GameStateError: If the game is in an invalid state
        
    Example:
        >>> hydra_mgr = HydraMgr(config)
        >>> result = hydra_mgr.execute_decision_cycle(budget=50)
        >>> print(f"Selected move: {result.move}")
        
    Note:
        This method is the core of the AI Hydra system and should be
        called once per game turn. The budget parameter directly affects
        computational cost and decision quality.
    """
```

**Required Docstring Elements:**
- Brief description (one line)
- Detailed explanation (if complex)
- Args section with types and descriptions
- Returns section with type and description
- Raises section for exceptions
- Example usage (when helpful)
- Notes for important implementation details

### Architecture Documentation

**Include These Sections:**
1. **Overview**: High-level system purpose and key features
2. **Components**: Major system components and their responsibilities
3. **Data Flow**: How data moves through the system
4. **Interfaces**: APIs and communication protocols
5. **Configuration**: How to configure the system
6. **Performance**: Expected performance characteristics
7. **Error Handling**: How errors are managed and recovered

**Use Mermaid Diagrams:**

```rst
.. mermaid::

   graph TD
       A[Neural Network] --> B[Tree Search]
       B --> C[Path Evaluation]
       C --> D[Master Game Update]
       D --> A
```

### User Guides

**Follow This Structure:**
1. **Prerequisites**: What users need before starting
2. **Installation**: Step-by-step setup instructions
3. **Quick Start**: Minimal example to get running
4. **Common Tasks**: Frequently performed operations
5. **Configuration**: How to customize behavior
6. **Troubleshooting**: Common issues and solutions

**Use Numbered Steps:**

```rst
Getting Started
===============

1. **Install Dependencies**

   .. code-block:: bash

      pip install -r requirements.txt

2. **Configure the System**

   Create a configuration file:

   .. code-block:: yaml

      simulation:
        grid_size: [20, 20]
        move_budget: 100

3. **Run Your First Simulation**

   .. code-block:: bash

      python -m ai_hydra.main --config config.yaml
```

## Quality Assurance

### Documentation Testing

**Automated Tests:**
- RST syntax validation
- Code block compilation
- Link checking (internal and external)
- Cross-reference validation
- Content accuracy against source code

**Manual Review Checklist:**
- [ ] Accurate technical content
- [ ] Clear, concise writing
- [ ] Proper formatting and structure
- [ ] Working code examples
- [ ] Up-to-date screenshots/diagrams
- [ ] Consistent terminology

### Maintenance Procedures

**Regular Updates:**
- Review documentation quarterly
- Update after major code changes
- Validate examples against current codebase
- Check external links for validity
- Update version numbers and compatibility info

**Change Management:**
- Document changes in commit messages
- Update related documentation when changing code
- Review documentation in pull requests
- Maintain changelog for documentation updates

## Tools and Automation

### Sphinx Configuration

**Required Extensions:**
```python
extensions = [
    'sphinx.ext.autodoc',      # Auto-generate API docs
    'sphinx.ext.viewcode',     # Link to source code
    'sphinx.ext.napoleon',     # Google/NumPy docstring support
    'sphinx.ext.intersphinx',  # Cross-project references
    'sphinxext.opengraph',     # Social media previews
    'myst_parser',             # Markdown support
]
```

### Build Automation

**Documentation Build Script:**
```bash
#!/bin/bash
# Build and validate documentation

cd docs
make clean
make html

# Run validation tests
python ../test_documentation.py

# Check for broken links
sphinx-build -b linkcheck _source _build/linkcheck
```

### Integration with Development

**Pre-commit Hooks:**
- Validate RST syntax
- Check code block compilation
- Verify cross-references
- Format code examples

**CI/CD Integration:**
- Build documentation on every commit
- Deploy to documentation site on main branch
- Run comprehensive tests in CI pipeline
- Generate coverage reports for API documentation

## Common Patterns

### Error Documentation

```rst
Common Errors
~~~~~~~~~~~~~

**ConfigurationError: Invalid grid size**

This error occurs when the grid size is too small for the initial snake length.

*Solution:* Increase grid size or decrease initial snake length:

.. code-block:: yaml

   simulation:
     grid_size: [10, 10]  # Minimum recommended
     initial_snake_length: 3
```

### Performance Documentation

```rst
Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision Cycle Performance:**

.. list-table::
   :header-rows: 1

   * - Budget Size
     - Average Time
     - Memory Usage
   * - 50 moves
     - 0.5 seconds
     - 50 MB
   * - 100 moves
     - 1.2 seconds
     - 75 MB
   * - 200 moves
     - 3.1 seconds
     - 120 MB
```

### Configuration Documentation

```rst
Configuration Reference
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ai_hydra.config
   :members:
   :show-inheritance:

**Example Configuration:**

.. literalinclude:: ../../examples/basic_config.yaml
   :language: yaml
   :caption: Basic simulation configuration
```

This documentation standard ensures that all AI Hydra documentation is consistent, accurate, maintainable, and serves the needs of developers, researchers, and operators effectively.