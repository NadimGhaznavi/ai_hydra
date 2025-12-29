#!/usr/bin/env python3
"""
Manual test script - run this and press Ctrl+C to test shutdown.
"""

import sys
from ai_hydra.headless_server import main

if __name__ == "__main__":
    print("🧪 Manual test: Start the server and press Ctrl+C to test shutdown")
    print("Expected behavior: Server should shut down gracefully with exit code 0")
    print("=" * 60)
    main()