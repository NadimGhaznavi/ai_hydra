#!/bin/bash
# organize_files.sh - File organization utility for AI Hydra project
#
# This script helps organize files according to the directory layout standards.
# It can detect misplaced files and suggest proper locations.
#
# Usage: ./scripts/organize_files.sh [--dry-run] [--fix]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
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

# Parse command line arguments
DRY_RUN=false
FIX_FILES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --fix)
            FIX_FILES=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--fix]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be moved without actually moving files"
            echo "  --fix        Actually move files to their proper locations"
            echo "  --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dry-run    # Preview file organization changes"
            echo "  $0 --fix        # Apply file organization changes"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Please run this script from the project root directory."
    exit 1
fi

print_info "Analyzing AI Hydra project file organization..."

# Create required directories if they don't exist
create_directories() {
    local dirs=(
        "scripts"
        "tools/debug"
        "tools/testing" 
        "tools/documentation"
        "tools/analysis"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            if [ "$FIX_FILES" = true ]; then
                mkdir -p "$dir"
                print_success "Created directory: $dir"
            else
                print_info "Would create directory: $dir"
            fi
        fi
    done
}

# Function to suggest file moves
suggest_move() {
    local file=$1
    local target_dir=$2
    local reason=$3
    
    if [ -f "$file" ]; then
        local target_path="$target_dir/$(basename "$file")"
        
        if [ "$FIX_FILES" = true ]; then
            if [ ! -d "$target_dir" ]; then
                mkdir -p "$target_dir"
            fi
            mv "$file" "$target_path"
            print_success "Moved: $file → $target_path ($reason)"
        else
            print_info "Would move: $file → $target_path ($reason)"
        fi
    fi
}

# Function to check for misplaced files in root directory
check_root_files() {
    print_info "Checking root directory for misplaced files..."
    
    # Debug scripts
    for file in debug_*.py; do
        if [ -f "$file" ]; then
            suggest_move "$file" "tools/debug" "debug utility"
        fi
    done
    
    # Test scripts
    for file in test_*.py run_test*.py; do
        if [ -f "$file" ] && [ "$file" != "test.log" ]; then
            # Skip if it's already a proper test file that should be in tests/
            if [[ "$file" == test_*.py ]] && grep -q "def test_" "$file" 2>/dev/null; then
                suggest_move "$file" "tests" "test file"
            else
                suggest_move "$file" "tools/testing" "testing utility"
            fi
        fi
    done
    
    # Documentation scripts
    for file in *doc*.py validate_*.py; do
        if [ -f "$file" ]; then
            suggest_move "$file" "tools/documentation" "documentation utility"
        fi
    done
    
    # Build/maintenance scripts
    for file in *.sh build*.py setup*.py update*.py; do
        if [ -f "$file" ] && [ "$file" != "setup.py" ]; then  # setup.py is deprecated but might exist
            suggest_move "$file" "scripts" "build/maintenance script"
        fi
    done
    
    # Analysis/profiling scripts
    for file in profile_*.py analyze_*.py benchmark_*.py; do
        if [ -f "$file" ]; then
            suggest_move "$file" "tools/analysis" "analysis utility"
        fi
    done
}

# Function to check for files that should be executable
check_executable_permissions() {
    print_info "Checking executable permissions..."
    
    local script_files=(
        "scripts/*.sh"
        "tools/*/*.py"
    )
    
    for pattern in "${script_files[@]}"; do
        for file in $pattern; do
            if [ -f "$file" ] && [ ! -x "$file" ]; then
                if [ "$FIX_FILES" = true ]; then
                    chmod +x "$file"
                    print_success "Made executable: $file"
                else
                    print_info "Would make executable: $file"
                fi
            fi
        done
    done
}

# Function to validate directory structure
validate_structure() {
    print_info "Validating directory structure..."
    
    local required_dirs=(
        "ai_hydra"
        "tests"
        "docs"
        ".kiro/specs"
        ".kiro/steering"
    )
    
    local missing_dirs=()
    
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            missing_dirs+=("$dir")
        fi
    done
    
    if [ ${#missing_dirs[@]} -gt 0 ]; then
        print_warning "Missing required directories:"
        for dir in "${missing_dirs[@]}"; do
            echo "  - $dir"
        done
    else
        print_success "All required directories present"
    fi
}

# Function to check test organization
check_test_organization() {
    print_info "Checking test organization..."
    
    if [ -d "tests" ]; then
        local test_categories=("unit" "property" "integration" "e2e")
        
        for category in "${test_categories[@]}"; do
            if [ ! -d "tests/$category" ]; then
                if [ "$FIX_FILES" = true ]; then
                    mkdir -p "tests/$category"
                    print_success "Created test directory: tests/$category"
                else
                    print_info "Would create test directory: tests/$category"
                fi
            fi
        done
        
        # Check for test files in wrong locations
        for test_file in tests/test_*.py; do
            if [ -f "$test_file" ]; then
                # Analyze test file to determine proper category
                if grep -q "@given\|@settings\|hypothesis" "$test_file" 2>/dev/null; then
                    if [[ "$test_file" != tests/property/* ]]; then
                        suggest_move "$test_file" "tests/property" "property-based test"
                    fi
                elif grep -q "integration\|Integration" "$test_file" 2>/dev/null; then
                    if [[ "$test_file" != tests/integration/* ]]; then
                        suggest_move "$test_file" "tests/integration" "integration test"
                    fi
                elif grep -q "e2e\|end.to.end\|EndToEnd" "$test_file" 2>/dev/null; then
                    if [[ "$test_file" != tests/e2e/* ]]; then
                        suggest_move "$test_file" "tests/e2e" "end-to-end test"
                    fi
                else
                    if [[ "$test_file" != tests/unit/* ]] && [[ "$test_file" == tests/test_*.py ]]; then
                        suggest_move "$test_file" "tests/unit" "unit test"
                    fi
                fi
            fi
        done
    fi
}

# Function to generate organization report
generate_report() {
    print_info "Generating file organization report..."
    
    echo ""
    echo "=== AI Hydra File Organization Report ==="
    echo ""
    
    # Count files by category
    local total_files=$(find . -name "*.py" -not -path "./.git/*" -not -path "./.pytest_cache/*" -not -path "./.hypothesis/*" -not -path "./htmlcov/*" | wc -l)
    local test_files=$(find tests -name "*.py" 2>/dev/null | wc -l || echo 0)
    local tool_files=$(find tools -name "*.py" 2>/dev/null | wc -l || echo 0)
    local script_files=$(find scripts -name "*" -type f 2>/dev/null | wc -l || echo 0)
    
    echo "📊 File Statistics:"
    echo "  Total Python files: $total_files"
    echo "  Test files: $test_files"
    echo "  Tool files: $tool_files"
    echo "  Script files: $script_files"
    echo ""
    
    # Directory structure overview
    echo "📁 Directory Structure:"
    tree -d -L 3 . 2>/dev/null || find . -type d -not -path "./.git/*" -not -path "./.pytest_cache/*" | head -20
    echo ""
    
    # Organization compliance
    local compliance_score=0
    local total_checks=5
    
    [ -d "tools" ] && ((compliance_score++))
    [ -d "scripts" ] && ((compliance_score++))
    [ -d "tests/unit" ] && ((compliance_score++))
    [ -d "tests/property" ] && ((compliance_score++))
    [ -f ".kiro/steering/directory-layout-standards.md" ] && ((compliance_score++))
    
    local compliance_percent=$((compliance_score * 100 / total_checks))
    
    echo "✅ Organization Compliance: $compliance_percent% ($compliance_score/$total_checks checks passed)"
    echo ""
}

# Main execution
main() {
    if [ "$DRY_RUN" = true ]; then
        print_info "Running in dry-run mode (no files will be moved)"
    elif [ "$FIX_FILES" = true ]; then
        print_info "Running in fix mode (files will be moved)"
    else
        print_info "Running in analysis mode (use --dry-run or --fix to make changes)"
    fi
    
    echo ""
    
    # Create required directories
    create_directories
    
    # Check for misplaced files
    check_root_files
    
    # Validate structure
    validate_structure
    
    # Check test organization
    check_test_organization
    
    # Check executable permissions
    check_executable_permissions
    
    # Generate report
    generate_report
    
    echo ""
    if [ "$FIX_FILES" = true ]; then
        print_success "File organization completed!"
    else
        print_info "Analysis completed. Use --fix to apply changes or --dry-run to preview."
    fi
}

# Run main function
main