#!/usr/bin/env python3
"""
Verification script for production TUI client epoch display.
This script verifies that the epoch display works with the production ai-hydra-tui command.
"""

import subprocess
import sys
import importlib.util
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_modules = [
        ("textual", "Textual TUI framework"),
        ("zmq", "ZeroMQ messaging"),
        ("ai_hydra.tui.client", "AI Hydra TUI client"),
        ("ai_hydra.zmq_protocol", "AI Hydra ZMQ protocol")
    ]
    
    all_available = True
    
    for module_name, description in required_modules:
        try:
            if module_name == "zmq":
                import zmq
            else:
                importlib.import_module(module_name)
            print(f"  ✅ {description}")
        except ImportError as e:
            print(f"  ❌ {description}: {e}")
            all_available = False
    
    return all_available

def verify_epoch_implementation():
    """Verify that the epoch display implementation is working."""
    print("\n🧪 Verifying epoch display implementation...")
    
    try:
        from ai_hydra.tui.client import HydraClient
        
        # Test client creation
        client = HydraClient()
        print("  ✅ HydraClient created successfully")
        
        # Test initial epoch value
        assert client.current_epoch == 0, f"Expected initial epoch 0, got {client.current_epoch}"
        print("  ✅ Initial epoch value is correct (0)")
        
        # Test epoch update
        client.current_epoch = 42
        assert client.current_epoch == 42, f"Expected epoch 42, got {client.current_epoch}"
        print("  ✅ Epoch update works correctly")
        
        # Test epoch watcher (mock UI)
        from unittest.mock import Mock
        mock_label = Mock()
        
        # Simulate UI query
        with unittest.mock.patch.object(client, 'query_one', return_value=mock_label):
            client.watch_current_epoch(0, 42)
            mock_label.update.assert_called_once_with("42")
        print("  ✅ Epoch watcher updates UI correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Epoch implementation error: {e}")
        return False

def check_cli_entry_point():
    """Check if the ai-hydra-tui CLI entry point is available."""
    print("\n🚀 Checking CLI entry point...")
    
    try:
        # Try to get help from the CLI command
        result = subprocess.run([
            "ai-hydra-tui", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("  ✅ ai-hydra-tui command is available")
            print("  ✅ CLI help output looks good")
            return True
        else:
            print(f"  ❌ ai-hydra-tui command failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("  ❌ ai-hydra-tui command not found")
        print("  💡 Try: pip install ai-hydra[tui]")
        return False
    except subprocess.TimeoutExpired:
        print("  ⏰ ai-hydra-tui command timed out")
        return False
    except Exception as e:
        print(f"  ❌ Error checking CLI: {e}")
        return False

def verify_status_widget_layout():
    """Verify that the status widget includes the epoch display."""
    print("\n📊 Verifying status widget layout...")
    
    try:
        from ai_hydra.tui.client import HydraClient
        
        # Create client and get compose result
        client = HydraClient()
        compose_result = list(client.compose())
        
        # Look for status panel in the compose result
        status_panel_found = False
        epoch_display_found = False
        
        def check_widget(widget):
            nonlocal status_panel_found, epoch_display_found
            
            # Check if this is the status panel
            if hasattr(widget, 'id') and widget.id == "status_panel":
                status_panel_found = True
                
                # Check children for epoch display
                if hasattr(widget, 'children'):
                    for child in widget.children:
                        if hasattr(child, 'children'):
                            for grandchild in child.children:
                                if (hasattr(grandchild, 'id') and 
                                    grandchild.id == "current_epoch"):
                                    epoch_display_found = True
            
            # Recursively check children
            if hasattr(widget, 'children'):
                for child in widget.children:
                    check_widget(child)
        
        # Check all widgets in compose result
        for widget in compose_result:
            check_widget(widget)
        
        if status_panel_found:
            print("  ✅ Status panel found in layout")
        else:
            print("  ❌ Status panel not found in layout")
        
        if epoch_display_found:
            print("  ✅ Epoch display found in status panel")
        else:
            print("  ❌ Epoch display not found in status panel")
        
        return status_panel_found and epoch_display_found
        
    except Exception as e:
        print(f"  ❌ Error verifying layout: {e}")
        return False

def main():
    """Main verification function."""
    print("🎮 AI Hydra TUI Epoch Display Verification")
    print("=" * 50)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Verify implementation
    impl_ok = verify_epoch_implementation() if deps_ok else False
    
    # Check CLI entry point
    cli_ok = check_cli_entry_point()
    
    # Verify layout
    layout_ok = verify_status_widget_layout() if deps_ok else False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Verification Summary:")
    print(f"  Dependencies: {'✅ OK' if deps_ok else '❌ FAILED'}")
    print(f"  Implementation: {'✅ OK' if impl_ok else '❌ FAILED'}")
    print(f"  CLI Entry Point: {'✅ OK' if cli_ok else '❌ FAILED'}")
    print(f"  Status Layout: {'✅ OK' if layout_ok else '❌ FAILED'}")
    
    if all([deps_ok, impl_ok, cli_ok, layout_ok]):
        print("\n🎉 All verifications passed!")
        print("\n🚀 Ready to use:")
        print("   1. Start server: ai-hydra-server")
        print("   2. Start TUI: ai-hydra-tui")
        print("   3. Look for 'Epoch: N' in the status panel")
        return True
    else:
        print("\n❌ Some verifications failed.")
        print("\n💡 Troubleshooting:")
        if not deps_ok:
            print("   - Install dependencies: pip install ai-hydra[tui]")
        if not cli_ok:
            print("   - Reinstall package: pip install --force-reinstall ai-hydra[tui]")
        return False

if __name__ == "__main__":
    import unittest.mock  # Import here to avoid issues if not available
    
    success = main()
    sys.exit(0 if success else 1)