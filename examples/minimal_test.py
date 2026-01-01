#!/usr/bin/env python3
import asyncio
from ai_hydra.router import HydraRouter


async def main():
    router = HydraRouter(router_address="127.0.0.1", router_port=5560)
    print("Router created")
    await router.start()
    print("Router started")
    await router.shutdown()
    print("Router stopped")


if __name__ == "__main__":
    asyncio.run(main())
