Version Update Procedure
========================

This document provides a step-by-step procedure for updating the AI Hydra project version across all relevant files. This ensures consistency and proper versioning for releases.

Overview
--------

The AI Hydra project uses semantic versioning (MAJOR.MINOR.PATCH) and maintains version information in multiple files that must be updated synchronously during releases.

**Current Version Locations:**
- Primary version source: ``pyproject.toml``
- Python package versions: ``ai_hydra/__init__.py``, ``ai_hydra/tui/__init__.py``
- Documentation version: ``docs/_source/conf.py``
- Legacy setup file: ``setup.py`` (if present)

Version Update Checklist
-------------------------

Follow this checklist to ensure all version references are updated correctly:

Prerequisites
~~~~~~~~~~~~~

1. **Determine New Version Number**
   
   Follow semantic versioning guidelines:
   
   - **MAJOR** version: Incompatible API changes
   - **MINOR** version: New functionality (backward compatible)
   - **PATCH** version: Bug fixes (backward compatible)
   
   Example: ``0.5.3`` → ``0.6.0`` (minor release) or ``0.5.4`` (patch release)

2. **Verify Current Version**
   
   Check the current version in the primary source:
   
   .. code-block:: bash
   
       grep "version" pyproject.toml

Step-by-Step Update Procedure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Update Primary Version Source**

Update the version in ``pyproject.toml``:

.. code-block:: toml

    [project]
    name = "ai-hydra"
    version = "X.Y.Z"  # ← Update this line

**Step 2: Update Main Package Version**

Update ``ai_hydra/__init__.py``:

.. code-block:: python

    __version__ = "X.Y.Z"  # ← Update this line

**Step 3: Update TUI Package Version**

Update ``ai_hydra/tui/__init__.py``:

.. code-block:: python

    __version__ = "X.Y.Z"  # ← Update this line

**Step 4: Update Documentation Version**

Update ``docs/_source/conf.py``:

.. code-block:: python

    # The full version, including alpha/beta/rc tags
    release = 'X.Y.Z'  # ← Update this line
    version = 'X.Y.Z'  # ← Update this line

**Step 5: Update Legacy Setup File (if present)**

If ``setup.py`` exists, update it:

.. code-block:: python

    setup(
        name="ai-hydra",
        version="X.Y.Z",  # ← Update this line
        # ... rest of setup configuration
    )

**Step 6: Verification**

Verify all versions are consistent:

.. code-block:: bash

    # Check all version references
    grep -r "version.*=.*[0-9]\+\.[0-9]\+\.[0-9]\+" . --include="*.py" --include="*.toml"
    
    # Should show consistent version across all files

File-by-File Update Guide
-------------------------

Primary Version Files
~~~~~~~~~~~~~~~~~~~~~

1. **pyproject.toml** (Primary Source)
   
   **Location:** Line 3
   
   **Format:** ``version = "X.Y.Z"``
   
   **Example:**
   
   .. code-block:: toml
   
       [project]
       name = "ai-hydra"
       version = "0.6.0"

2. **ai_hydra/__init__.py** (Main Package)
   
   **Location:** Line 9
   
   **Format:** ``__version__ = "X.Y.Z"``
   
   **Example:**
   
   .. code-block:: python
   
       __version__ = "0.6.0"
       __author__ = "AI Hydra Team"

3. **ai_hydra/tui/__init__.py** (TUI Package)
   
   **Location:** Line 7
   
   **Format:** ``__version__ = "X.Y.Z"``
   
   **Example:**
   
   .. code-block:: python
   
       """
       AI Hydra TUI Client Package
       """
       
       __version__ = "0.6.0"

4. **docs/_source/conf.py** (Documentation)
   
   **Location:** Lines 37-38
   
   **Format:** ``release = 'X.Y.Z'`` and ``version = 'X.Y.Z'``
   
   **Example:**
   
   .. code-block:: python
   
       # The full version, including alpha/beta/rc tags
       release = '0.6.0'
       version = '0.6.0'

Legacy Files (Update if Present)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

5. **setup.py** (Legacy Setup - Optional)
   
   **Location:** Line 15
   
   **Format:** ``version="X.Y.Z",``
   
   **Example:**
   
   .. code-block:: python
   
       setup(
           name="ai-hydra",
           version="0.6.0",
           author="AI Hydra Team",
           # ... rest of configuration
       )

Automated Update Script
-----------------------

You can use this script to update all versions automatically:

.. code-block:: bash

    #!/bin/bash
    # update_version.sh - Automated version update script
    
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <new_version>"
        echo "Example: $0 0.6.0"
        exit 1
    fi
    
    NEW_VERSION=$1
    
    echo "Updating AI Hydra version to $NEW_VERSION..."
    
    # Update pyproject.toml
    sed -i "s/version = \"[0-9]\+\.[0-9]\+\.[0-9]\+\"/version = \"$NEW_VERSION\"/" pyproject.toml
    
    # Update main package __init__.py
    sed -i "s/__version__ = \"[0-9]\+\.[0-9]\+\.[0-9]\+\"/__version__ = \"$NEW_VERSION\"/" ai_hydra/__init__.py
    
    # Update TUI package __init__.py
    sed -i "s/__version__ = \"[0-9]\+\.[0-9]\+\.[0-9]\+\"/__version__ = \"$NEW_VERSION\"/" ai_hydra/tui/__init__.py
    
    # Update documentation conf.py
    sed -i "s/release = '[0-9]\+\.[0-9]\+\.[0-9]\+'/release = '$NEW_VERSION'/" docs/_source/conf.py
    sed -i "s/version = '[0-9]\+\.[0-9]\+\.[0-9]\+'/version = '$NEW_VERSION'/" docs/_source/conf.py
    
    # Update setup.py if it exists
    if [ -f "setup.py" ]; then
        sed -i "s/version=\"[0-9]\+\.[0-9]\+\.[0-9]\+\"/version=\"$NEW_VERSION\"/" setup.py
    fi
    
    echo "Version update complete. Verifying..."
    
    # Verify updates
    echo "=== Version Check ==="
    echo "pyproject.toml:"
    grep "version.*=" pyproject.toml
    echo "ai_hydra/__init__.py:"
    grep "__version__" ai_hydra/__init__.py
    echo "ai_hydra/tui/__init__.py:"
    grep "__version__" ai_hydra/tui/__init__.py
    echo "docs/_source/conf.py:"
    grep -E "(release|version) = " docs/_source/conf.py
    if [ -f "setup.py" ]; then
        echo "setup.py:"
        grep "version=" setup.py
    fi

**Usage:**

.. code-block:: bash

    # Make script executable
    chmod +x update_version.sh
    
    # Update to version 0.6.0
    ./update_version.sh 0.6.0

Manual Update Template
----------------------

Use this template for manual updates:

.. code-block:: text

    Version Update: X.Y.Z → A.B.C
    
    Files to Update:
    ☐ pyproject.toml (line 3)
    ☐ ai_hydra/__init__.py (line 9)
    ☐ ai_hydra/tui/__init__.py (line 7)
    ☐ docs/_source/conf.py (lines 37-38)
    ☐ setup.py (line 15) - if present
    
    Verification:
    ☐ All files show consistent version
    ☐ Package builds successfully
    ☐ Documentation builds successfully
    ☐ Tests pass with new version

Post-Update Verification
------------------------

After updating all version files, perform these verification steps:

**1. Version Consistency Check**

.. code-block:: bash

    # Check all version references are consistent
    python -c "
    import ai_hydra
    import ai_hydra.tui
    print(f'Main package: {ai_hydra.__version__}')
    print(f'TUI package: {ai_hydra.tui.__version__}')
    "

**2. Package Build Test**

.. code-block:: bash

    # Test package builds correctly
    python -m build
    
    # Check generated package
    ls dist/

**3. Documentation Build Test**

.. code-block:: bash

    # Test documentation builds
    cd docs
    make clean
    make html
    
    # Check for version in generated docs
    grep -r "0\.6\.0" _build/html/

**4. Installation Test**

.. code-block:: bash

    # Test installation from source
    pip install -e .
    
    # Verify installed version
    python -c "import ai_hydra; print(ai_hydra.__version__)"

**5. Test Suite Verification**

.. code-block:: bash

    # Run tests to ensure version update didn't break anything
    pytest tests/ -v

Release Workflow Integration
----------------------------

This version update procedure should be integrated into your release workflow:

**Pre-Release Checklist:**

1. ☐ Update version numbers using this procedure
2. ☐ Verify all tests pass
3. ☐ Update CHANGELOG.md with new version
4. ☐ Build and test documentation
5. ☐ Create git tag with version number
6. ☐ Build distribution packages
7. ☐ Test installation from built packages

**Git Tagging:**

.. code-block:: bash

    # Create and push version tag
    git add .
    git commit -m "Bump version to X.Y.Z"
    git tag -a vX.Y.Z -m "Release version X.Y.Z"
    git push origin main --tags

Common Issues and Solutions
---------------------------

**Issue: Inconsistent Versions**

*Problem:* Different files show different version numbers.

*Solution:* Use the automated script or carefully check each file manually.

**Issue: Documentation Not Updating**

*Problem:* Documentation still shows old version after update.

*Solution:* Clear documentation build cache:

.. code-block:: bash

    cd docs
    make clean
    make html

**Issue: Package Import Fails**

*Problem:* Python can't import the package after version update.

*Solution:* Reinstall the package in development mode:

.. code-block:: bash

    pip install -e .

**Issue: Tests Reference Old Version**

*Problem:* Some tests hardcode version numbers and fail after update.

*Solution:* Update test files that reference specific versions:

.. code-block:: bash

    # Find tests with hardcoded versions
    grep -r "0\.[0-9]\+\.[0-9]\+" tests/

Version History Tracking
-------------------------

Maintain a record of version updates:

.. code-block:: text

    Version History:
    - 0.5.3: Current stable release
    - 0.5.2: Documentation improvements
    - 0.5.1: Bug fixes
    - 0.5.0: Major feature release
    
    Next Planned:
    - 0.5.4: Patch release (bug fixes)
    - 0.6.0: Minor release (new features)

This procedure ensures consistent versioning across the entire AI Hydra project and reduces the risk of version-related issues during releases.