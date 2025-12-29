#!/usr/bin/env python3
"""
Test script to verify that ai-hydra-server shuts down properly with Ctrl+C.
"""

import subprocess
import time
import signal
import sys
import os

def test_server_shutdown():
    """Test that the server shuts down properly with Ctrl+C."""
    print("🧪 Testing AI Hydra server shutdown behavior...")
    
    # Start the server as a subprocess
    print("🚀 Starting ai-hydra-server...")
    try:
        # Start server with minimal output
        process = subprocess.Popen(
            [sys.executable, "-m", "ai_hydra.headless_server", "--log-level", "WARNING"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give server time to start
        time.sleep(2)
        
        # Check if server is running
        if process.poll() is None:
            print("✅ Server started successfully")
        else:
            print("❌ Server failed to start")
            stdout, stderr = process.communicate()
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
        
        # Send SIGINT (Ctrl+C equivalent)
        print("🛑 Sending SIGINT (Ctrl+C) to server...")
        process.send_signal(signal.SIGINT)
        
        # Wait for graceful shutdown (max 10 seconds)
        try:
            stdout, stderr = process.communicate(timeout=10)
            exit_code = process.returncode
            
            print(f"📊 Server exit code: {exit_code}")
            if stdout:
                print(f"📝 Server output: {stdout}")
            if stderr:
                print(f"⚠️  Server errors: {stderr}")
            
            if exit_code == 0:
                print("✅ Server shut down gracefully!")
                return True
            else:
                print(f"❌ Server exit code was {exit_code}, expected 0")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Server did not shut down within 10 seconds, force killing...")
            process.kill()
            process.communicate()
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def main():
    """Run the shutdown test."""
    print("AI Hydra Server Shutdown Test")
    print("=" * 40)
    
    success = test_server_shutdown()
    
    if success:
        print("\n🎉 Test PASSED: Server shuts down properly with Ctrl+C")
        sys.exit(0)
    else:
        print("\n💥 Test FAILED: Server does not shut down properly")
        sys.exit(1)

if __name__ == "__main__":
    main()