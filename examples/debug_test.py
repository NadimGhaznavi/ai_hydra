#!/usr/bin/env python3
import asyncio
from ai_hydra.router import HydraRouter


async def main():
    router = HydraRouter(router_address="127.0.0.1", router_port=5560)
    print("Router created")
    print("Router type:", type(router))
    print("Has start method:", hasattr(router, "start"))

    if hasattr(router, "start"):
        print("Start method:", getattr(router, "start"))
        await router.start()
        print("Router started")
    else:
        print("ERROR: No start method found!")
        print("Available methods:", [m for m in dir(router) if not m.startswith("_")])

    await router.shutdown()
    print("Router stopped")


if __name__ == "__main__":
    asyncio.run(main())
