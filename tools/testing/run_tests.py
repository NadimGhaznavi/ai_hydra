#!/usr/bin/env python3
"""
AI Hydra Test Runner

This script provides convenient commands for running different types of tests
in the AI Hydra project. It wraps pytest with commonly used configurations.

Usage:
    python run_tests.py [command] [options]

Commands:
    all         Run all tests with coverage
    fast        Run fast tests only (exclude slow tests)
    unit        Run unit tests only
    property    Run property-based tests only
    integration Run integration tests only
    performance Run performance tests only
    coverage    Generate detailed coverage report
    specific    Run specific test file or function
    
Examples:
    python run_tests.py all
    python run_tests.py fast
    python run_tests.py unit -v
    python run_tests.py specific tests/test_game_logic.py
    python run_tests.py coverage --html
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    if description:
        print(f"\n🧪 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install with: pip install -e '.[dev]'")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="AI Hydra Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "command",
        choices=["all", "fast", "unit", "property", "integration", "performance", 
                "coverage", "specific", "help"],
        help="Test command to run"
    )
    
    parser.add_argument(
        "target",
        nargs="?",
        help="Specific test file or function (for 'specific' command)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report (for 'coverage' command)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Test timeout in seconds (default: 600)"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        help="Number of parallel workers (requires pytest-xdist)"
    )
    
    args = parser.parse_args()
    
    if args.command == "help":
        parser.print_help()
        return True
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("❌ Tests directory not found. Please run from the project root.")
        return False
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    # Add common options
    if args.verbose:
        base_cmd.append("-v")
    
    if args.timeout:
        base_cmd.extend(["--timeout", str(args.timeout)])
    
    if args.parallel:
        base_cmd.extend(["-n", str(args.parallel)])
    
    # Command-specific configurations
    if args.command == "all":
        cmd = base_cmd + ["--cov=ai_hydra", "--cov-report=term-missing"]
        return run_command(cmd, "Running all tests with coverage")
    
    elif args.command == "fast":
        cmd = base_cmd + ["-m", "not slow"]
        return run_command(cmd, "Running fast tests only")
    
    elif args.command == "unit":
        cmd = base_cmd + ["-m", "unit"]
        return run_command(cmd, "Running unit tests")
    
    elif args.command == "property":
        cmd = base_cmd + ["-m", "property", "--hypothesis-show-statistics"]
        return run_command(cmd, "Running property-based tests")
    
    elif args.command == "integration":
        cmd = base_cmd + ["-m", "integration"]
        return run_command(cmd, "Running integration tests")
    
    elif args.command == "performance":
        cmd = base_cmd + ["-m", "performance"]
        return run_command(cmd, "Running performance tests")
    
    elif args.command == "coverage":
        coverage_opts = ["--cov=ai_hydra", "--cov-report=term-missing"]
        if args.html:
            coverage_opts.append("--cov-report=html")
        cmd = base_cmd + coverage_opts
        success = run_command(cmd, "Generating coverage report")
        if success and args.html:
            print("\n📊 HTML coverage report generated in htmlcov/index.html")
        return success
    
    elif args.command == "specific":
        if not args.target:
            print("❌ Please specify a test file or function for 'specific' command")
            print("Example: python run_tests.py specific tests/test_game_logic.py")
            return False
        cmd = base_cmd + [args.target]
        return run_command(cmd, f"Running specific test: {args.target}")
    
    else:
        print(f"❌ Unknown command: {args.command}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)