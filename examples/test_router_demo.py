#!/usr/bin/env python3
"""
Hydra Router Demo

Demonstrates the hydra-router working with multiple clients and a server.
This example shows:
1. Starting the router
2. Connecting multiple clients
3. Connecting a server
4. Message routing between clients and server
5. Heartbeat monitoring
"""

import asyncio
import logging
import time
from typing import Dict, Any

from ai_hydra.router import HydraRouter
from ai_hydra.mq_client import MQClient
from ai_hydra.zmq_protocol import ZMQMessage, MessageType
from ai_hydra.router_constants import RouterConstants


class DemoServer:
    """Demo server that responds to client commands."""

    def __init__(self, router_address: str = "tcp://localhost:5556"):
        self.client = MQClient(
            router_address=router_address,
            client_type="HydraServer",
            client_id="demo-server-001",
        )
        self.is_running = False

    async def start(self):
        """Start the demo server."""
        print("🖥️  Starting demo server...")
        await self.client.connect()
        self.is_running = True

        # Start message handling loop
        asyncio.create_task(self._handle_messages())
        print("✅ Demo server connected and ready")

    async def _handle_messages(self):
        """Handle incoming messages from clients."""
        while self.is_running:
            try:
                message = await self.client.receive_message()
                if message:
                    await self._process_message(message)
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"❌ Server error: {e}")

    async def _process_message(self, message: Dict[str, Any]):
        """Process a message from a client."""
        elem = message.get(RouterConstants.ELEM)
        data = message.get(RouterConstants.DATA, {})

        print(f"🖥️  Server received: {elem} with data: {data}")

        # Respond based on message type
        if elem == RouterConstants.START_SIMULATION:
            response = ZMQMessage.create_response(
                message_type=MessageType.SIMULATION_STARTED,
                client_id=self.client.client_id,
                data={
                    "status": "Simulation started successfully",
                    "sim_id": "demo-001",
                },
            )
            await self.client.send_message(response)
            print("🖥️  Server sent: SIMULATION_STARTED")

        elif elem == RouterConstants.GET_STATUS:
            response = ZMQMessage.create_response(
                message_type=MessageType.STATUS_RESPONSE,
                client_id=self.client.client_id,
                data={
                    "status": "running",
                    "uptime": time.time(),
                    "active_simulations": 1,
                },
            )
            await self.client.send_message(response)
            print("🖥️  Server sent: STATUS_RESPONSE")

    async def stop(self):
        """Stop the demo server."""
        print("🖥️  Stopping demo server...")
        self.is_running = False
        await self.client.disconnect()


class DemoClient:
    """Demo client that sends commands to server."""

    def __init__(self, client_id: str, router_address: str = "tcp://localhost:5556"):
        self.client = MQClient(
            router_address=router_address,
            client_type="HydraClient",
            client_id=client_id,
        )
        self.client_id = client_id

    async def start(self):
        """Start the demo client."""
        print(f"👤 Starting demo client {self.client_id}...")
        await self.client.connect()
        print(f"✅ Demo client {self.client_id} connected")

    async def send_start_simulation(self):
        """Send start simulation command."""
        message = ZMQMessage.create_command(
            message_type=MessageType.START_SIMULATION,
            client_id=self.client_id,
            data={"config": "demo_config", "seed": 42},
        )
        await self.client.send_message(message)
        print(f"👤 Client {self.client_id} sent: START_SIMULATION")

    async def send_get_status(self):
        """Send get status command."""
        message = ZMQMessage.create_command(
            message_type=MessageType.GET_STATUS, client_id=self.client_id, data={}
        )
        await self.client.send_message(message)
        print(f"👤 Client {self.client_id} sent: GET_STATUS")

    async def listen_for_responses(self, duration: float = 5.0):
        """Listen for server responses."""
        print(f"👤 Client {self.client_id} listening for responses...")
        start_time = time.time()

        while time.time() - start_time < duration:
            message = await self.client.receive_message()
            if message:
                elem = message.get(RouterConstants.ELEM, "unknown")
                data = message.get(RouterConstants.DATA, {})
                print(f"👤 Client {self.client_id} received: {elem} with data: {data}")
            await asyncio.sleep(0.1)

    async def stop(self):
        """Stop the demo client."""
        print(f"👤 Stopping demo client {self.client_id}...")
        await self.client.disconnect()


async def run_demo():
    """Run the complete router demo."""
    print("🚀 Starting Hydra Router Demo")
    print("=" * 50)

    # Start the router
    print("📡 Starting router on localhost:5556...")
    router = HydraRouter(router_address="127.0.0.1", router_port=5556, log_level="INFO")

    try:
        await router.start()

        # Start router message handling in background
        router_task = asyncio.create_task(router.handle_requests())

        # Give router time to start
        await asyncio.sleep(1)

        # Start demo server
        server = DemoServer()
        await server.start()

        # Give server time to connect
        await asyncio.sleep(1)

        # Start demo clients
        client1 = DemoClient("demo-client-001")
        client2 = DemoClient("demo-client-002")

        await client1.start()
        await client2.start()

        # Give clients time to connect
        await asyncio.sleep(1)

        print("\n🎬 Starting demo scenario...")
        print("-" * 30)

        # Demo scenario
        print("\n📋 Step 1: Client 1 requests server status")
        await client1.send_get_status()

        await asyncio.sleep(1)

        print("\n📋 Step 2: Client 2 starts simulation")
        await client2.send_start_simulation()

        await asyncio.sleep(1)

        print("\n📋 Step 3: Both clients listen for server responses")
        await asyncio.gather(
            client1.listen_for_responses(3), client2.listen_for_responses(3)
        )

        print("\n📋 Step 4: Demonstrate heartbeat monitoring")
        print("💓 Clients are sending heartbeats every 5 seconds...")
        await asyncio.sleep(6)  # Let heartbeats happen

        print("\n🎉 Demo completed successfully!")

    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise
    finally:
        print("\n🧹 Cleaning up...")

        # Stop clients
        try:
            await client1.stop()
            await client2.stop()
        except:
            pass

        # Stop server
        try:
            await server.stop()
        except:
            pass

        # Stop router
        router_task.cancel()
        try:
            await router_task
        except asyncio.CancelledError:
            pass

        await router.shutdown()
        print("✅ Cleanup complete")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("🎯 Hydra Router Demo")
    print("This demo shows the router working with multiple clients and a server")
    print("Press Ctrl+C to stop\n")

    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        raise
