#!/usr/bin/env python3
"""
Basic Client-Server Example for Hydra Router

This example demonstrates basic communication between a client and server
through the Hydra Router system.
"""

import asyncio
import logging
from hydra_router import MQClient, MessageType, ZMQMessage


async def run_server():
    """Run a basic server that responds to client messages."""
    server = MQClient(
        router_address="tcp://localhost:5556",
        client_type="HydraServer",
        client_id="example-server",
    )

    print("🖥️  Starting server...")
    await server.connect()
    print("✅ Server connected to router")

    try:
        while True:
            # Receive messages from clients via router
            message = await server.receive_message_blocking(timeout=1.0)

            if message:
                print(f"📨 Server received: {message}")

                # Send response back through router (will be broadcast to clients)
                response = ZMQMessage.create_broadcast(
                    MessageType.STATUS_UPDATE,
                    {
                        "server_response": f"Processed message: {message.get('elem', 'unknown')}"
                    },
                )
                await server.send_message(response)
                print("📤 Server sent response")

    except KeyboardInterrupt:
        print("\n🛑 Server shutting down...")
    finally:
        await server.disconnect()
        print("✅ Server disconnected")


async def run_client(client_id: str):
    """Run a basic client that sends messages to the server."""
    client = MQClient(
        router_address="tcp://localhost:5556",
        client_type="HydraClient",
        client_id=client_id,
    )

    print(f"👤 Starting client {client_id}...")
    await client.connect()
    print(f"✅ Client {client_id} connected to router")

    try:
        # Send a few test messages
        for i in range(3):
            message = ZMQMessage.create_command(
                message_type=MessageType.GET_STATUS,
                client_id=client_id,
                request_id=f"req-{i}",
                data={"message_number": i + 1},
            )

            await client.send_message(message)
            print(f"📤 Client {client_id} sent message {i + 1}")

            # Wait for server response
            response = await client.receive_message_blocking(timeout=2.0)
            if response:
                print(f"📨 Client {client_id} received: {response}")

            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print(f"\n🛑 Client {client_id} shutting down...")
    finally:
        await client.disconnect()
        print(f"✅ Client {client_id} disconnected")


async def main():
    """Main function to run the example."""
    print("🚀 Hydra Router Basic Client-Server Example")
    print("=" * 50)
    print("📋 Make sure to start the router first:")
    print("   ai-hydra-router --log-level INFO")
    print("=" * 50)

    # Wait a moment for user to read
    await asyncio.sleep(2)

    # Start server and clients concurrently
    tasks = [
        asyncio.create_task(run_server()),
        asyncio.create_task(run_client("client-001")),
        asyncio.create_task(run_client("client-002")),
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all components...")
        for task in tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Example completed!")
