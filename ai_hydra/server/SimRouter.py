"""
ai_snake_lab/server/SimRouter.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

import asyncio
import zmq
import zmq.asyncio
import time
import argparse

from copy import deepcopy

from ai_snake_lab.utils.LabLogger import LabLogger

from ai_snake_lab.constants.DSim import DSim
from ai_snake_lab.constants.DMQ import DMQ, DMQ_Label
from ai_snake_lab.constants.DLabLogger import DLog


class SimRouter:
    """Pure MQ router between TUIs and the Simulation Server."""

    def __init__(self, router, loglevel):

        # Setup a logger
        self.log = LabLogger(client_id=DMQ.SIM_ROUTER)
        self.log.loglevel(loglevel)

        # Initialize ZMQ context
        self.ctx = zmq.asyncio.Context()

        # Create a ROUTER socket to manage multiple clients
        self.socket = self.ctx.socket(zmq.ROUTER)
        mq_port = DSim.MQ_PORT
        protocol = DSim.PROTOCOL
        sim_service = f"{protocol}://{router}:{mq_port}"
        try:
            self.socket.bind(sim_service)
        except zmq.error.ZMQError as e:
            self.log.critical(f"Failed to bind router to {sim_service}: {e}")
            raise
        self.log.info(DMQ_Label.STARTUP_MSG % sim_service)

        self.clients = {}
        self.client_count = 0
        self.server_count = 0

        # We have two async processes that modify the same dictionary, we need a lock to
        # prevent concurrent writes.
        self.clients_lock = asyncio.Lock()
        # Start the process that prunes inactive clients
        asyncio.create_task(self.prune_dead_clients_bg())

    async def broadcast_to_simclients(self, elem, data, sender):
        """Broadcast simulation updates (from SimServer) to all connected TUIs."""

        client_ids = []
        clients = deepcopy(self.clients)

        for client_id in clients.keys():
            if clients[client_id][0] == DMQ.SIM_CLIENT:
                client_ids.append(client_id)

        # Nothing to do, just return
        if not client_ids:
            return

        msg = {DMQ.SENDER: DMQ.SIM_SERVER, DMQ.ELEM: elem, DMQ.DATA: data}
        msg_bytes = zmq.utils.jsonapi.dumps(msg)

        for client_id in client_ids:
            if client_id != sender:
                # print(f"Sending MQ message to client: {client_id_str}: {msg}")
                await self.socket.send_multipart([client_id.encode(), msg_bytes])

    async def handle_requests(self):
        """Continuously route messages between SimClients and the SimServer."""
        while True:
            try:
                # ROUTER sockets prepend an identity frame
                frames = await self.socket.recv_multipart()
                identity = frames[0]
                identity_str = identity.decode()
                msg_bytes = frames[1]

                if len(frames) != 2:
                    self.log.error(f"{DMQ_Label.MALFORMED_MESSAGE}: {frames}")
                    continue

                msg = zmq.utils.jsonapi.loads(msg_bytes)

            except asyncio.CancelledError:
                self.log.info("SimRouter shutting down...")
                break

            except KeyboardInterrupt:
                self.log.info(DMQ_Label.SHUTDOWN_MSG)
                break

            except zmq.ZMQError as e:
                self.log.error(f"ZMQ error in router: {e}")
                await asyncio.sleep(0.1)
                continue

            except Exception as e:
                self.log.error(DMQ_Label.ROUTER_ERROR % e)
                continue

            # Parse message
            sender_type = msg.get(DMQ.SENDER)  # SimClient or SimServer
            elem = msg.get(DMQ.ELEM)
            data = msg.get(DMQ.DATA, {})

            ### Uncomment for debugging
            # self.log.debug(f"MQ DEBUG {sender_type}({identity_str}): {msg}")

            # Validate message
            if not sender_type or elem is None:
                self.log.error(f"{DMQ_Label.MALFORMED_MESSAGE}: {msg}")
                continue

            # Capture heartbeat messages sender identity
            if elem == DMQ.HEARTBEAT:
                async with self.clients_lock:
                    self.clients[identity_str] = (sender_type, time.time())
                self.log.debug(f"Received heartbeat from {sender_type}({identity_str})")
                continue

            # Log start/stop/pause messages
            if elem == DMQ.CMD:
                self.log.info(
                    f"Received {data} command from {sender_type}/{identity_str}"
                )

            ### Routing logic

            # Forward all SimClient commands to the SimServer
            if sender_type == DMQ.SIM_CLIENT:
                await self.forward_to_simserver(elem=elem, data=data, sender=identity)

            ## Routing rules for messages from the SimServer vary...
            elif sender_type == DMQ.SIM_SERVER:

                # Drop STATUS or ERROR messages
                if elem in (DMQ.STATUS, DMQ.ERROR):
                    continue

                # Send these messages only to a specific SimClient
                if elem in (
                    DMQ.CUR_SIM_STATE,
                    DMQ.CUR_HIGHSCORE,
                    DMQ.AVG_LOSS_DATA,
                    DMQ.OLD_GAME_SCORE_DATA,
                    DMQ.OLD_HIGHSCORE_EVENTS,
                ):
                    await self.send_to_simclient(elem=elem, data=data)

                # All remaining messages are broadcast to all SimClients
                else:
                    await self.broadcast_to_simclients(
                        elem=elem, data=data, sender=identity
                    )

            else:
                self.log.error(f"{DMQ_Label.UNKNOWN_SENDER}: {sender_type}")

    async def forward_to_simserver(self, elem, data, sender):
        """Forward TUI command to the simulation server."""

        # Find all connected SimServers
        sim_servers = []
        self.clients = deepcopy(self.clients)

        for identity in self.clients.keys():
            if self.clients[identity][0] == DMQ.SIM_SERVER:
                sim_servers.append(identity)

        # No SimServer connected - inform the client
        if not sim_servers:
            await self.socket.send_multipart(
                [
                    sender,
                    zmq.utils.jsonapi.dumps(
                        {DMQ.ERROR: DMQ_Label.NO_SIMSERVER_CONNECTED}
                    ),
                ]
            )
            return

        # Construct atomic message
        msg = {DMQ.SENDER: DMQ.SIM_CLIENT, DMQ.ELEM: elem, DMQ.DATA: data}
        msg_bytes = zmq.utils.jsonapi.dumps(msg)

        # Send message to all connected SimServers
        for server_id in sim_servers:
            await self.socket.send_multipart([server_id.encode(), msg_bytes])

        # Acknowledge sender (SimClient)
        await self.socket.send_multipart(
            [
                sender,
                zmq.utils.jsonapi.dumps({DMQ.STATUS: DMQ.OK}),
            ]
        )

    async def prune_dead_clients_bg(self):
        while True:
            await self.prune_dead_clients()
            await asyncio.sleep(DSim.HEARTBEAT_INTERVAL * 4)

    async def prune_dead_clients(self):
        """Periodic cleanup loop that removes clients whose heartbeats have stopped."""
        async with self.clients_lock:
            now = time.time()
            client_count = 0
            server_count = 0

            clients_copy = deepcopy(self.clients)
            for identity in clients_copy.keys():
                sender_type, last = self.clients[identity]
                if now - last > (DSim.HEARTBEAT_INTERVAL * 3):
                    self.log.info(f"Removing inactive client: {identity}")
                    del self.clients[identity]
                else:
                    if sender_type == DMQ.SIM_SERVER:
                        server_count += 1
                    elif sender_type == DMQ.SIM_CLIENT:
                        client_count += 1

            await self.forward_to_simserver(
                elem=DMQ.CUR_NUM_CLIENTS,
                data=client_count,
                sender=DMQ.SIM_ROUTER.encode(),
            )

            if client_count != self.client_count or server_count != self.server_count:
                self.log.info(
                    f"Connected client(s): {client_count}, server(s): {server_count}"
                )

    async def send_to_simclient(self, elem, data):
        client_id = data[0]
        payload = data[1]
        msg = {DMQ.SENDER: DMQ.SIM_SERVER, DMQ.ELEM: elem, DMQ.DATA: payload}
        msg_bytes = zmq.utils.jsonapi.dumps(msg)
        await self.socket.send_multipart([client_id.encode(), msg_bytes])
        self.log.debug(f"Targeted MQ message to {client_id}: {elem}/{data}")


async def main_async(router, loglevel):
    router = SimRouter(router=router, loglevel=loglevel)
    # Start the router
    await router.handle_requests()


def main():
    # Process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--router",
        type=str,
        default=DSim.HOST,
        help="IP or hostname of the AI Snake Lab router",
    )
    parser.add_argument("-v", "--verbose", help="Show additional information")
    args = parser.parse_args()
    if args.verbose:
        loglevel = DLog.DEBUG
    else:
        loglevel = DLog.INFO
    asyncio.run(main_async(router=args.router, loglevel=loglevel))


if __name__ == "__main__":
    main()
