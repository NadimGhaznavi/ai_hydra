Version Update Procedure
========================

This document provides a step-by-step procedure for updating the AI Hydra project version across all relevant files. This ensures consistency and proper versioning for releases.

Overview
--------

The AI Hydra project uses semantic versioning (MAJOR.MINOR.PATCH) and maintains version information in multiple files that must be updated synchronously during releases. The project follows modern Python packaging standards using ``pyproject.toml`` as the primary configuration file.

**Current Version Locations:**
- Primary version source: ``pyproject.toml``
- Python package versions: ``ai_hydra/__init__.py``, ``ai_hydra/tui/__init__.py``
- Documentation version: ``docs/_source/conf.py``

**Note:** Legacy ``setup.py`` files are no longer used and should be removed if found. The project has been modernized to use ``pyproject.toml`` exclusively for packaging configuration.

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

**Step 5: Verification**

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

Automated Update Script
-----------------------

The project includes an enhanced automated version update script (``update_version.sh``) that handles all version updates and includes additional features:

**Key Features:**
- Comprehensive version validation and error checking
- Interactive confirmation before making changes
- Automatic CHANGELOG.md update with release timestamps
- Deprecation warnings for legacy setup.py files
- Colored output for better user experience
- Verification of all updated files

**Script Location:** ``update_version.sh`` (project root)

**Enhanced Script Capabilities:**

.. code-block:: bash

    #!/bin/bash
    # Enhanced update_version.sh - Automated version update script for AI Hydra
    
    # Features:
    # - Version format validation (semantic versioning)
    # - Interactive confirmation
    # - Automatic CHANGELOG.md updates
    # - Legacy setup.py deprecation warnings
    # - Comprehensive verification
    # - Colored output and error handling

**Usage:**

.. code-block:: bash

    # Make script executable
    chmod +x update_version.sh
    
    # Update to version 0.6.0 (interactive mode)
    ./update_version.sh 0.6.0
    
    # The script will:
    # 1. Validate the version format
    # 2. Show current vs new version
    # 3. Ask for confirmation
    # 4. Update all version files
    # 5. Update CHANGELOG.md with release timestamp
    # 6. Verify all changes
    # 7. Provide next steps guidance

**Script Output Example:**

.. code-block:: text

    [INFO] Updating AI Hydra version to 0.6.0...
    [INFO] Current version: 0.5.3
    [INFO] New version: 0.6.0
    
    Do you want to proceed with the version update? (y/N): y
    
    [INFO] Starting version update process...
    [INFO] Updating primary version (pyproject.toml): pyproject.toml
    [SUCCESS] ✓ Updated pyproject.toml
    [INFO] Updating main package version: ai_hydra/__init__.py
    [SUCCESS] ✓ Updated ai_hydra/__init__.py
    [INFO] Updating TUI package version: ai_hydra/tui/__init__.py
    [SUCCESS] ✓ Updated ai_hydra/tui/__init__.py
    [INFO] Updating documentation version: docs/_source/conf.py
    [SUCCESS] ✓ Updated docs/_source/conf.py
    [INFO] Updating CHANGELOG.md with release 0.6.0
    [SUCCESS] ✓ Updated CHANGELOG.md with release 0.6.0
    
    [SUCCESS] Version update complete!

**Legacy File Handling:**

The enhanced script includes deprecation warnings for legacy files:

.. code-block:: text

    [WARNING] Found setup.py - this file should be removed as it's no longer used
    [WARNING] The project now uses pyproject.toml exclusively for packaging

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

**Issue: Legacy setup.py Found**

*Problem:* The script warns about setup.py files that should be removed.

*Solution:* Remove the legacy setup.py file as the project now uses pyproject.toml exclusively:

.. code-block:: bash

    # Remove legacy setup.py (if it exists)
    rm setup.py
    
    # Verify pyproject.toml is the primary packaging file
    cat pyproject.toml | grep -A 5 "\[project\]"

**Issue: Project Structure Changes**

*Problem:* Legacy files or incorrect project structure after cleanup.

*Solution:* Ensure clean project structure following modern Python packaging:

.. code-block:: bash

    # Verify modern project structure
    ls -la | grep -E "(pyproject\.toml|setup\.py)"
    
    # Should show pyproject.toml only, no setup.py
    # Tests should be in tests/ directory, not in project root
    # Legacy files like qasync.sh, manual_test_server.py should be removed

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