# ai_hydra/network/MQClient.py
#
#    AI Snake Lab
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
#    Website: https://snakelab.osoyalce.com
#    License: GPL 3.0
#

import asyncio
import zmq
import zmq.asyncio
import numpy as np
import sys
import random
from datetime import datetime

from ai_hydra.network.MQHelper import mq_srv_msg
from ai_hydra.constants.DSim import DSim
from ai_hydra.constants.DMQ import DMQ, DMQ_Label


class MQClient:
    """Manage the connection to the SimRouter."""

    def __init__(self, router=DSim.HOST, loglevel=None):
        self.router = router
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.identity_str = f"{DMQ.MQ_CLIENT}-{np.random.randint(0,10000)}"

        random.seed(DSim.RANDOM_SEED)

        self.identity = self.identity_str.encode()

        # self.lablog = LabLogger(client_id=f"{self.identity_str}")
        self.socket.setsockopt(zmq.IDENTITY, self.identity)
        if not self.router:
            self.router = DSim.HOST
        self.router_addr = f"{DSim.PROTOCOL}://{self.router}:{DSim.MQ_PORT}"

        self.stop_event = asyncio.Event()
        self.heartbeat_stop_event = asyncio.Event()

        self.socket.connect(self.router_addr)

        # Handy alias
        self.send = self.socket.send_json

        # Start sending heartbeat messages
        self.heartbeat_task = asyncio.create_task(self.send_heartbeat())
        # self.lablog.loglevel(loglevel)
        # self.lablog.info(f"{DMQ_Label.CONNECTED_TO_ROUTER} {self.router_addr}")

    async def send_heartbeat(self):
        """Periodic heartbeat to let the SimRouter know this client is alive."""
        while not self.heartbeat_stop_event.is_set():
            self.lablog.debug(
                f"Sending heartbeat: {DMQ.HEARTBEAT}/{self.identity.decode()}"
            )
            await self.send(mq_srv_msg(DMQ.HEARTBEAT, self.identity.decode()))
            await asyncio.sleep(DSim.HEARTBEAT_INTERVAL)

    async def quit(self):
        self.heartbeat_stop_event.set()
        try:
            await self.socket.disconnect(self.router_addr)
            self.socket.close(linger=0)
        finally:
            self.ctx.term()

    async def recv_multipart(self):
        return await self.socket.recv_multipart()
