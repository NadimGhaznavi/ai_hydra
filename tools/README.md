# Development Tools

This directory contains development utilities and tools for the AI Hydra project.

## Directory Structure

```
tools/
├── debug/                      # Debugging utilities
├── testing/                    # Testing utilities  
├── documentation/              # Documentation tools
└── README.md                   # This file
```

## Debug Tools (`debug/`)

### `debug_file_patterns.py`
Tests file patterns preservation through the metadata collection chain.

**Usage:**
```bash
python tools/debug/debug_file_patterns.py
```

**Purpose:**
- Validates that file patterns are correctly preserved in token tracking
- Tests MetadataCollector functionality
- Verifies TokenTracker integration

## Testing Tools (`testing/`)

### `run_tests.py`
Comprehensive test runner with multiple execution modes.

**Usage:**
```bash
# Run all tests with coverage
python tools/testing/run_tests.py all

# Run fast tests only (exclude slow tests)
python tools/testing/run_tests.py fast

# Run specific test categories
python tools/testing/run_tests.py unit
python tools/testing/run_tests.py property
python tools/testing/run_tests.py integration

# Run specific test file
python tools/testing/run_tests.py specific tests/test_game_logic.py

# Generate coverage report
python tools/testing/run_tests.py coverage --html
```

**Features:**
- Multiple test execution modes
- Coverage reporting
- Timeout management
- Parallel execution support
- Verbose output options

## Documentation Tools (`documentation/`)

### `test_documentation.py`
Comprehensive documentation validation suite.

**Usage:**
```bash
python tools/documentation/test_documentation.py
```

**Features:**
- RST syntax validation
- Cross-reference checking
- Content accuracy verification
- Sphinx build integrity testing
- Link validation
- Code block syntax checking

### `validate_docs.py`
Simple documentation validation without Sphinx dependencies.

**Usage:**
```bash
python tools/documentation/validate_docs.py
```

**Features:**
- Basic RST syntax checking
- Cross-reference validation
- Lightweight validation for CI/CD

## Usage Guidelines

### Making Tools Executable

All Python tools in this directory should be executable:

```bash
chmod +x tools/*/tool_name.py
```

### Adding New Tools

When adding new tools:

1. **Choose appropriate category**: Place in `debug/`, `testing/`, `documentation/`, or create new category
2. **Include shebang**: Start Python files with `#!/usr/bin/env python3`
3. **Add docstring**: Include comprehensive module docstring with usage examples
4. **Make executable**: Set appropriate file permissions
5. **Update README**: Add documentation to this file

### Tool Development Standards

All tools should follow these standards:

```python
#!/usr/bin/env python3
"""
Tool Name - Brief description

Detailed description of what the tool does and when to use it.

Usage:
    python tools/category/tool_name.py [options]

Examples:
    python tools/category/tool_name.py --verbose
    python tools/category/tool_name.py --config config.yaml

Options:
    --verbose    Enable verbose output
    --config     Specify configuration file
    --help       Show this help message
"""

import sys
import argparse
from pathlib import Path

def main():
    """Main function with proper argument parsing."""
    parser = argparse.ArgumentParser(
        description="Tool description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Tool implementation
    try:
        # Main logic here
        result = perform_tool_operation(args)
        
        if result:
            print("✅ Tool completed successfully")
            return 0
        else:
            print("❌ Tool failed")
            return 1
            
    except Exception as e:
        print(f"💥 Tool error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### Integration with Development Workflow

These tools integrate with the development workflow:

1. **Pre-commit hooks**: Use validation tools in git pre-commit hooks
2. **CI/CD pipeline**: Run tools in continuous integration
3. **Development scripts**: Include tools in build and deployment scripts
4. **IDE integration**: Configure IDE to use tools for validation

### Tool Categories

#### Debug Tools
- File pattern debugging
- Metadata collection testing
- Component integration debugging
- Performance profiling utilities

#### Testing Tools
- Test runners and orchestration
- Coverage analysis
- Performance benchmarking
- Test data generation

#### Documentation Tools
- Documentation validation
- Cross-reference checking
- Build integrity testing
- Content accuracy verification

#### Future Categories
As the project grows, consider adding:
- `deployment/` - Deployment utilities
- `monitoring/` - System monitoring tools
- `analysis/` - Code analysis and metrics
- `migration/` - Data migration utilities

## Maintenance

### Regular Tasks
- Review tool effectiveness quarterly
- Update tool documentation
- Ensure tools work with latest dependencies
- Add new tools as development needs evolve

### Quality Standards
- All tools must have comprehensive help text
- Tools should handle errors gracefully
- Include usage examples in docstrings
- Follow project coding standards
- Provide clear success/failure indicators