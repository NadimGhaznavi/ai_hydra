#!/bin/bash
# update_version.sh - Automated version update script for AI Hydra
#
# This script updates the version number across all relevant files in the AI Hydra project.
# It ensures consistency across pyproject.toml, Python packages, and documentation.
#
# Usage: ./update_version.sh <new_version>
# Example: ./update_version.sh 0.6.0

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if version argument is provided
if [ $# -eq 0 ]; then
    print_error "No version number provided"
    echo "Usage: $0 <new_version>"
    echo "Example: $0 0.6.0"
    echo ""
    echo "The version should follow semantic versioning (MAJOR.MINOR.PATCH)"
    exit 1
fi

NEW_VERSION=$1

# Validate version format (basic check for X.Y.Z)
if ! [[ $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    print_error "Invalid version format: $NEW_VERSION"
    echo "Version must follow semantic versioning format: MAJOR.MINOR.PATCH (e.g., 1.2.3)"
    exit 1
fi

print_status "Updating AI Hydra version to $NEW_VERSION..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Please run this script from the project root directory."
    exit 1
fi

# Get current version for comparison
CURRENT_VERSION=$(grep "version = " pyproject.toml | head -1 | sed 's/.*version = "\([^"]*\)".*/\1/')
print_status "Current version: $CURRENT_VERSION"
print_status "New version: $NEW_VERSION"

# Confirm update
echo ""
read -p "Do you want to proceed with the version update? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Version update cancelled"
    exit 0
fi

echo ""
print_status "Starting version update process..."

# Function to update version in a file
update_file_version() {
    local file=$1
    local pattern=$2
    local replacement=$3
    local description=$4
    
    if [ -f "$file" ]; then
        print_status "Updating $description: $file"
        if sed -i.bak "$pattern" "$file"; then
            rm -f "$file.bak"  # Remove backup file created by sed
            print_success "✓ Updated $file"
        else
            print_error "✗ Failed to update $file"
            return 1
        fi
    else
        print_warning "File not found: $file (skipping)"
    fi
}

# Function to update CHANGELOG.md with new release heading
update_changelog() {
    local version=$1
    local changelog_file="CHANGELOG.md"
    
    if [ ! -f "$changelog_file" ]; then
        print_warning "CHANGELOG.md not found (skipping changelog update)"
        return 0
    fi
    
    print_status "Updating CHANGELOG.md with release $version"
    
    # Get current date and time in the desired format
    local release_date=$(date '+%Y-%m-%d %H:%M')
    local release_heading="## [Release $version] - $release_date"
    
    # Create a temporary file for the updated changelog
    local temp_file=$(mktemp)
    
    # Process the changelog: add new release heading after [Unreleased]
    awk -v release_heading="$release_heading" '
    /^## \[Unreleased\]/ {
        print $0
        print ""
        print release_heading
        print ""
        next
    }
    { print }
    ' "$changelog_file" > "$temp_file"
    
    # Replace the original file
    if mv "$temp_file" "$changelog_file"; then
        print_success "✓ Updated CHANGELOG.md with release $version"
    else
        print_error "✗ Failed to update CHANGELOG.md"
        rm -f "$temp_file"
        return 1
    fi
}

# Update pyproject.toml
update_file_version "pyproject.toml" \
    "s/version = \"[0-9]\+\.[0-9]\+\.[0-9]\+\"/version = \"$NEW_VERSION\"/" \
    "s/version = \"[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\"/version = \"$NEW_VERSION\"/" \
    "primary version (pyproject.toml)"

# Update main package __init__.py
update_file_version "ai_hydra/__init__.py" \
    "s/__version__ = \"[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\"/__version__ = \"$NEW_VERSION\"/" \
    "main package version"

# Update TUI package __init__.py
update_file_version "ai_hydra/tui/__init__.py" \
    "s/__version__ = \"[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\"/__version__ = \"$NEW_VERSION\"/" \
    "TUI package version"

# Update documentation conf.py
if [ -f "docs/_source/conf.py" ]; then
    print_status "Updating documentation version: docs/_source/conf.py"
    sed -i.bak "s/release = '[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*'/release = '$NEW_VERSION'/" docs/_source/conf.py
    sed -i.bak "s/version = '[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*'/version = '$NEW_VERSION'/" docs/_source/conf.py
    rm -f docs/_source/conf.py.bak
    print_success "✓ Updated docs/_source/conf.py"
fi

# Update setup.py if it exists (DEPRECATED - setup.py has been removed)
# This section is kept for backward compatibility but setup.py is no longer used
if [ -f "setup.py" ]; then
    print_warning "Found setup.py - this file should be removed as it's no longer used"
    print_warning "The project now uses pyproject.toml exclusively for packaging"
fi

# Update CHANGELOG.md with new release heading
update_changelog "$NEW_VERSION"

echo ""
print_success "Version update complete!"

# Verification
print_status "Verifying version consistency..."
echo ""
echo "=== Version Verification ==="

# Check pyproject.toml
if [ -f "pyproject.toml" ]; then
    echo "📄 pyproject.toml:"
    grep "version.*=" pyproject.toml | head -1
fi

# Check main package
if [ -f "ai_hydra/__init__.py" ]; then
    echo "🐍 ai_hydra/__init__.py:"
    grep "__version__" ai_hydra/__init__.py
fi

# Check TUI package
if [ -f "ai_hydra/tui/__init__.py" ]; then
    echo "🖥️  ai_hydra/tui/__init__.py:"
    grep "__version__" ai_hydra/tui/__init__.py
fi

# Check documentation
if [ -f "docs/_source/conf.py" ]; then
    echo "📚 docs/_source/conf.py:"
    grep -E "(release|version) = " docs/_source/conf.py
fi

# Check setup.py (DEPRECATED)
if [ -f "setup.py" ]; then
    echo "⚠️  setup.py (DEPRECATED - should be removed):"
    grep "version=" setup.py | head -1
fi

# Check CHANGELOG.md
if [ -f "CHANGELOG.md" ]; then
    echo "📝 CHANGELOG.md:"
    grep -A 2 "## \[Unreleased\]" CHANGELOG.md | head -5
fi

echo ""

# Test Python import
print_status "Testing Python package import..."
if python -c "import ai_hydra; print(f'✓ Main package version: {ai_hydra.__version__}')" 2>/dev/null; then
    print_success "Python import test passed"
else
    print_warning "Python import test failed - you may need to reinstall the package"
    echo "  Run: pip install -e ."
fi

echo ""
print_success "Version update completed successfully!"

echo ""
echo "📋 Next Steps:"
echo "1. Review all changes: git diff"
echo "2. Review CHANGELOG.md to ensure release notes are complete"
echo "3. Run tests: pytest tests/"
echo "4. Build documentation: cd docs && make html"
echo "5. Commit changes: git add . && git commit -m 'Release version $NEW_VERSION'"
echo "6. Create tag: git tag -a v$NEW_VERSION -m 'Release version $NEW_VERSION'"
echo "7. Push changes: git push origin main --tags"
echo ""
echo "💡 Tip: Use 'git checkout .' to revert all changes if needed"

echo ""
print_status "Version update process complete! 🎉"