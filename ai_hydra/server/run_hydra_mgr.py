import argparse
import asyncio
import traceback

from ai_hydra.constants.DHydra import (
    DHydraLogDef,
    DHydraRouterDef,
    DHydraServerDef,
    DModule,
)
from ai_hydra.server.HydraMgr import HydraMgr


async def amain() -> None:
    print("run_hydra_mgr: entered amain()")
    p = argparse.ArgumentParser(description="AI Hydra Manager Server")
    p.add_argument("--address", default="*", help="Bind address")
    p.add_argument("--port", type=int, default=DHydraServerDef.PORT)
    p.add_argument("--router-address", default=DHydraRouterDef.HOSTNAME)
    p.add_argument("--router-port", type=int, default=DHydraRouterDef.PORT)
    p.add_argument("--identity", default=DModule.HYDRA_MGR)
    p.add_argument("--log_level", default=DHydraLogDef.DEFAULT_LOG_LEVEL)
    args = p.parse_args()

    print("run_hydra_mgr: parsed args", args)

    server = HydraMgr(
        address=args.address,
        port=args.port,
        router_address=args.router_address,
        router_port=args.router_port,
        identity=args.identity,
        log_level=args.log_level,
    )

    print("run_hydra_mgr: constructed erver, calling run()")
    await server.run()
    print("run_hydra_mgr: server.run() returned (unexpected)")


def main() -> None:
    try:
        asyncio.run(amain())
    # except KeyboardInterrupt:
    #    pass
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
