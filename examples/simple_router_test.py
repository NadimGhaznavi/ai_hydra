#!/usr/bin/env python3
"""
Simple Router Test

Basic test to verify the router can start and handle connections.
"""

import asyncio
import logging
from ai_hydra.router import HydraRouter


async def test_router():
    """Test basic router functionality."""
    print("🧪 Testing basic router functionality...")

    # Create router
    router = HydraRouter(
        router_address="127.0.0.1", router_port=5557, log_level="DEBUG"
    )
    print("✅ Router created")

    try:
        # Start router
        print("🚀 Starting router...")
        await router.start()
        print("✅ Router started successfully")

        # Start message handling
        print("📡 Starting message handling...")
        handle_task = asyncio.create_task(router.handle_requests())

        # Let it run for a few seconds
        print("⏱️  Running for 5 seconds...")
        await asyncio.sleep(5)

        print("✅ Router test completed successfully!")

    except Exception as e:
        print(f"❌ Router test failed: {e}")
        raise
    finally:
        print("🧹 Shutting down router...")
        if "handle_task" in locals():
            handle_task.cancel()
            try:
                await handle_task
            except asyncio.CancelledError:
                pass
        await router.shutdown()
        print("✅ Router shutdown complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_router())
