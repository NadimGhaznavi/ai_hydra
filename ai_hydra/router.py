"""
AI Hydra Router

Pure MQ router between TUI clients and the AI Hydra server.
Based on the ai_snake_lab SimRouter pattern.
"""

import asyncio
import logging
import time
import argparse
from copy import deepcopy
from typing import Dict, Any, List

import zmq
import zmq.asyncio

from ai_hydra.router_constants import RouterConstants, RouterLabels


class HydraRouter:
    """Pure MQ router between TUI clients and the AI Hydra server."""

    def __init__(self, router_address: str = "0.0.0.0", router_port: int = 5556, log_level: str = "INFO"):
        """
        Initialize the AI Hydra router.
        
        Args:
            router_address: Address to bind the router to
            router_port: Port to bind the router to
            log_level: Logging level
        """
        # Setup logging
        self.logger = logging.getLogger("HydraRouter")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Initialize ZMQ context
        self.ctx = zmq.asyncio.Context()
        
        # Create a ROUTER socket to manage multiple clients
        self.socket = self.ctx.socket(zmq.ROUTER)
        
        # Bind to the router service
        router_service = f"tcp://{router_address}:{router_port}"
        try:
            self.socket.bind(router_service)
            self.logger.info(f"Router started on {router_service}")
        except zmq.error.ZMQError as e:
            self.logger.critical(f"Failed to bind router to {router_service}: {e}")
            raise
        
        # Client tracking
        self.clients: Dict[str, tuple] = {}  # client_id -> (client_type, last_heartbeat)
        self.client_count = 0
        self.server_count = 0
        
        # Lock for concurrent client dictionary access
        self.clients_lock = asyncio.Lock()
        
        # Start background client pruning
        asyncio.create_task(self.prune_dead_clients_bg())

    async def broadcast_to_clients(self, elem: str, data: Any, sender_id: str) -> None:
        """Broadcast messages from server to all connected clients."""
        client_ids = []
        clients = deepcopy(self.clients)
        
        for client_id in clients.keys():
            if clients[client_id][0] == RouterConstants.HYDRA_CLIENT:
                client_ids.append(client_id)
        
        # Nothing to do if no clients
        if not client_ids:
            return
        
        msg = {
            RouterConstants.SENDER: RouterConstants.HYDRA_SERVER,
            RouterConstants.ELEM: elem,
            RouterConstants.DATA: data
        }
        msg_bytes = zmq.utils.jsonapi.dumps(msg)
        
        for client_id in client_ids:
            if client_id != sender_id:
                try:
                    await self.socket.send_multipart([client_id.encode(), msg_bytes])
                    self.logger.debug(f"Broadcast message to client {client_id}: {elem}")
                except Exception as e:
                    self.logger.error(f"Failed to send message to client {client_id}: {e}")

    async def handle_requests(self) -> None:
        """Continuously route messages between clients and servers."""
        self.logger.info("Router message handling started")
        
        while True:
            try:
                # ROUTER sockets prepend an identity frame
                frames = await self.socket.recv_multipart()
                identity = frames[0]
                identity_str = identity.decode()
                msg_bytes = frames[1]
                
                if len(frames) != 2:
                    self.logger.error(f"Malformed message: {frames}")
                    continue
                
                msg = zmq.utils.jsonapi.loads(msg_bytes)
                
            except asyncio.CancelledError:
                self.logger.info("Router shutting down...")
                break
            except KeyboardInterrupt:
                self.logger.info("Router shutdown requested")
                break
            except zmq.ZMQError as e:
                self.logger.error(f"ZMQ error in router: {e}")
                await asyncio.sleep(0.1)
                continue
            except Exception as e:
                self.logger.error(f"Router error: {e}")
                continue
            
            # Parse message
            sender_type = msg.get(RouterConstants.SENDER)
            elem = msg.get(RouterConstants.ELEM)
            data = msg.get(RouterConstants.DATA, {})
            
            # Debug logging
            self.logger.debug(f"Message from {sender_type}({identity_str}): {elem}")
            
            # Validate message
            if not sender_type or elem is None:
                self.logger.error(f"Malformed message: {msg}")
                continue
            
            # Handle heartbeat messages
            if elem == RouterConstants.HEARTBEAT:
                async with self.clients_lock:
                    self.clients[identity_str] = (sender_type, time.time())
                self.logger.debug(f"Heartbeat from {sender_type}({identity_str})")
                continue
            
            # Log important commands
            if elem in [RouterConstants.START_SIMULATION, RouterConstants.STOP_SIMULATION, 
                       RouterConstants.PAUSE_SIMULATION, RouterConstants.RESUME_SIMULATION,
                       RouterConstants.RESET_SIMULATION]:
                self.logger.info(f"Command {elem} from {sender_type}/{identity_str}")
            
            ### Routing Logic ###
            
            # Forward client commands to server
            if sender_type == RouterConstants.HYDRA_CLIENT:
                await self.forward_to_server(elem=elem, data=data, sender=identity)
            
            # Handle server messages
            elif sender_type == RouterConstants.HYDRA_SERVER:
                # Drop status/error messages (they're handled locally)
                if elem in [RouterConstants.STATUS, RouterConstants.ERROR]:
                    continue
                
                # Broadcast all other messages to clients
                await self.broadcast_to_clients(elem=elem, data=data, sender_id=identity_str)
            
            else:
                self.logger.error(f"Unknown sender type: {sender_type}")

    async def forward_to_server(self, elem: str, data: Any, sender: bytes) -> None:
        """Forward client command to the AI Hydra server."""
        # Find all connected servers
        servers = []
        clients = deepcopy(self.clients)
        
        for identity in clients.keys():
            if clients[identity][0] == RouterConstants.HYDRA_SERVER:
                servers.append(identity)
        
        # No server connected - inform the client
        if not servers:
            error_msg = {RouterConstants.ERROR: "No AI Hydra server connected"}
            await self.socket.send_multipart([
                sender,
                zmq.utils.jsonapi.dumps(error_msg)
            ])
            self.logger.warning("No server connected for client request")
            return
        
        # Construct message
        msg = {
            RouterConstants.SENDER: RouterConstants.HYDRA_CLIENT,
            RouterConstants.ELEM: elem,
            RouterConstants.DATA: data
        }
        msg_bytes = zmq.utils.jsonapi.dumps(msg)
        
        # Send to all connected servers (usually just one)
        for server_id in servers:
            try:
                await self.socket.send_multipart([server_id.encode(), msg_bytes])
                self.logger.debug(f"Forwarded {elem} to server {server_id}")
            except Exception as e:
                self.logger.error(f"Failed to forward message to server {server_id}: {e}")
        
        # Acknowledge the client
        ack_msg = {RouterConstants.STATUS: RouterConstants.OK}
        await self.socket.send_multipart([
            sender,
            zmq.utils.jsonapi.dumps(ack_msg)
        ])

    async def prune_dead_clients_bg(self) -> None:
        """Background task to prune dead clients."""
        while True:
            await self.prune_dead_clients()
            await asyncio.sleep(RouterConstants.HEARTBEAT_INTERVAL * 4)

    async def prune_dead_clients(self) -> None:
        """Remove clients that haven't sent heartbeats recently."""
        async with self.clients_lock:
            now = time.time()
            client_count = 0
            server_count = 0
            
            clients_copy = deepcopy(self.clients)
            for identity in clients_copy.keys():
                sender_type, last_heartbeat = self.clients[identity]
                
                # Remove clients that haven't sent heartbeat in 3x the interval
                if now - last_heartbeat > (RouterConstants.HEARTBEAT_INTERVAL * 3):
                    self.logger.info(f"Removing inactive client: {identity}")
                    del self.clients[identity]
                else:
                    if sender_type == RouterConstants.HYDRA_SERVER:
                        server_count += 1
                    elif sender_type == RouterConstants.HYDRA_CLIENT:
                        client_count += 1
            
            # Update counts if changed
            if client_count != self.client_count or server_count != self.server_count:
                self.client_count = client_count
                self.server_count = server_count
                self.logger.info(f"Connected clients: {client_count}, servers: {server_count}")

    async def shutdown(self) -> None:
        """Gracefully shutdown the router."""
        self.logger.info("Shutting down router...")
        
        # Close socket
        if self.socket:
            self.socket.close(linger=0)
        
        # Terminate context
        if self.ctx:
            self.ctx.term()
        
        self.logger.info("Router shutdown complete")


async def main_async(router_address: str, router_port: int, log_level: str) -> None:
    """Main async function for the router."""
    router = HydraRouter(router_address=router_address, router_port=router_port, log_level=log_level)
    
    try:
        await router.handle_requests()
    except KeyboardInterrupt:
        pass
    finally:
        await router.shutdown()


def main():
    """Main entry point for the AI Hydra router."""
    parser = argparse.ArgumentParser(
        description="AI Hydra Router - Routes messages between clients and servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start router on default port
  ai-hydra-router
  
  # Start router on custom port with debug logging
  ai-hydra-router --port 6666 --log-level DEBUG
  
  # Start router bound to specific interface
  ai-hydra-router --address 192.168.1.100 --port 5556
        """
    )
    
    parser.add_argument(
        "-a", "--address",
        default="0.0.0.0",
        help="IP address to bind router to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=5556,
        help="Port to bind router to (default: 5556)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print(f"🚀 Starting AI Hydra Router on {args.address}:{args.port}")
    print(f"📊 Log level: {args.log_level}")
    print("📡 Press Ctrl+C to stop")
    
    try:
        asyncio.run(main_async(args.address, args.port, args.log_level))
    except KeyboardInterrupt:
        print("\n🛑 Router stopped by user")
    except Exception as e:
        print(f"❌ Router failed: {e}")
        raise


if __name__ == "__main__":
    main()