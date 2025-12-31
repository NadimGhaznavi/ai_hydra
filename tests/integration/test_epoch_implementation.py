#!/usr/bin/env python3
"""
Test runner for epoch display implementation.
This script runs the tests to verify the epoch display feature works correctly.
"""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run all tests for the epoch display feature."""
    print("🧪 Testing Epoch Display Implementation")
    print("=" * 50)
    
    # Test files to run
    test_files = [
        "tests/unit/test_tui_status_display.py",
        "tests/property/test_tui_epoch_display.py", 
        "tests/integration/test_tui_epoch_integration.py"
    ]
    
    all_passed = True
    
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"❌ Test file not found: {test_file}")
            all_passed = False
            continue
            
        print(f"\n📋 Running {test_file}...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_file, 
                "-v", 
                "--tb=short"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"✅ {test_file} - All tests passed!")
                print(result.stdout)
            else:
                print(f"❌ {test_file} - Some tests failed!")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                all_passed = False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file} - Tests timed out!")
            all_passed = False
        except Exception as e:
            print(f"💥 {test_file} - Error running tests: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All epoch display tests passed!")
        print("\n✨ Implementation Summary:")
        print("   • Added current_epoch reactive variable")
        print("   • Added Epoch display to status widget")
        print("   • Added epoch tracking in status updates")
        print("   • Added epoch reset functionality")
        print("   • Added epoch watcher for UI updates")
        print("   • Created comprehensive test suite")
        print("\n🚀 The epoch display feature is ready to use!")
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False
    
    return True

def demo_epoch_display():
    """Show how to use the epoch display feature."""
    print("\n🎮 Epoch Display Demo")
    print("=" * 30)
    print("To see the epoch display in action:")
    print("1. Run: python demo_tui.py")
    print("2. Click 'Start' to begin simulation")
    print("3. Watch the 'Epoch' field in the status panel")
    print("4. The epoch will increment every 20 moves in the demo")
    print("5. Click 'Reset' to see epoch return to 0")
    print("\nThe epoch display shows the current game number,")
    print("helping you track AI training progress across multiple games.")

if __name__ == "__main__":
    success = run_tests()
    demo_epoch_display()
    
    if not success:
        sys.exit(1)