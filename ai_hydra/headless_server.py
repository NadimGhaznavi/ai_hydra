#!/usr/bin/env python3
"""
Headless server entry point for AI Hydra.

This script starts the AI Hydra server that connects to the router,
making the AI agent completely controllable via ZeroMQ messages 
without any GUI dependencies.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from .mq_client import MQClient
from .router_constants import RouterConstants
from .zmq_protocol import MessageType
from .zmq_server import ZMQServer


class HeadlessServer:
    """
    Headless server wrapper that connects to the AI Hydra router.
    """
    
    def __init__(self, router_address: str = "tcp://localhost:5556", 
                 heartbeat_interval: float = 5.0,
                 log_level: str = "INFO",
                 log_file: str = None):
        """
        Initialize the headless server.
        
        Args:
            router_address: Address of the AI Hydra router
            heartbeat_interval: Seconds between heartbeat messages
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional log file path
        """
        self.router_address = router_address
        self.heartbeat_interval = heartbeat_interval
        self.log_level = log_level
        self.log_file = log_file
        
        self.mq_client = None
        self.zmq_server = None
        self.shutdown_event = asyncio.Event()
        
        # Setup logging
        self._setup_logging()
        
        self.logger = logging.getLogger("ai_hydra.headless_server")
    
    def _setup_logging(self):
        """Configure logging for the headless server."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Add file handler if specified
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        # For async code, we should use asyncio signal handling
        loop = asyncio.get_running_loop()
        
        def signal_handler():
            self.logger.info("Received shutdown signal, initiating graceful shutdown...")
            self.shutdown_event.set()
        
        # Handle SIGINT (Ctrl+C) and SIGTERM using asyncio
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)
    
    async def start(self):
        """Start the headless server."""
        self.logger.info("Starting AI Hydra Headless Server")
        self.logger.info(f"Router Address: {self.router_address}")
        self.logger.info(f"Heartbeat Interval: {self.heartbeat_interval}s")
        self.logger.info(f"Log Level: {self.log_level}")
        if self.log_file:
            self.logger.info(f"Log File: {self.log_file}")
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Create MQ client to connect to router
        self.mq_client = MQClient(
            router_address=self.router_address,
            client_type=RouterConstants.HYDRA_SERVER,
            heartbeat_interval=self.heartbeat_interval
        )
        
        # Create ZMQ server for actual simulation logic
        self.zmq_server = ZMQServer(
            bind_address="tcp://127.0.0.1:0",  # Bind to random local port
            heartbeat_interval=self.heartbeat_interval,
            log_level=self.log_level
        )
        
        try:
            # Connect to router
            if not await self.mq_client.connect():
                raise RuntimeError("Failed to connect to router")
            
            # Start ZMQ server in background
            await self.zmq_server.start_background()
            
            # Start message handling
            message_task = asyncio.create_task(self.handle_router_messages())
            
            # Wait for shutdown signal
            self.logger.info("Server running. Press Ctrl+C to stop.")
            await self.shutdown_event.wait()
            
            self.logger.info("Shutdown signal received, stopping server...")
            
            # Cancel message handling
            message_task.cancel()
            
            # Stop components
            if self.zmq_server:
                await self.zmq_server.stop()
            
            if self.mq_client:
                await self.mq_client.disconnect()
            
        except Exception as e:
            # Only log as error if it's not a graceful shutdown
            if not self.shutdown_event.is_set():
                self.logger.error(f"Server error: {e}")
                raise
            else:
                self.logger.info(f"Server stopped during shutdown: {e}")
        finally:
            self.logger.info("Headless server stopped")
    
    async def handle_router_messages(self):
        """Handle messages from the router."""
        while not self.shutdown_event.is_set():
            try:
                # Receive message from router
                message = await self.mq_client.receive_message()
                
                if message:
                    await self.process_router_message(message)
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Error handling router message: {e}")
                await asyncio.sleep(1.0)
    
    async def process_router_message(self, message: dict):
        """Process a message received from the router."""
        try:
            elem = message.get(RouterConstants.ELEM)
            data = message.get(RouterConstants.DATA, {})
            
            self.logger.debug(f"Processing router message: {elem}")
            
            # Map router messages to ZMQ server calls
            if elem == RouterConstants.START_SIMULATION:
                # Forward to ZMQ server
                response = await self.zmq_server.handle_start_simulation(data)
                # Send response back through router
                await self.send_router_response(RouterConstants.SIMULATION_STARTED, response)
                
            elif elem == RouterConstants.STOP_SIMULATION:
                response = await self.zmq_server.handle_stop_simulation(data)
                await self.send_router_response(RouterConstants.SIMULATION_STOPPED, response)
                
            elif elem == RouterConstants.PAUSE_SIMULATION:
                response = await self.zmq_server.handle_pause_simulation(data)
                await self.send_router_response(RouterConstants.SIMULATION_PAUSED, response)
                
            elif elem == RouterConstants.RESUME_SIMULATION:
                response = await self.zmq_server.handle_resume_simulation(data)
                await self.send_router_response(RouterConstants.SIMULATION_RESUMED, response)
                
            elif elem == RouterConstants.RESET_SIMULATION:
                response = await self.zmq_server.handle_reset_simulation(data)
                await self.send_router_response(RouterConstants.SIMULATION_RESET, response)
                
            elif elem == RouterConstants.GET_STATUS:
                response = await self.zmq_server.handle_get_status(data)
                await self.send_router_response(RouterConstants.STATUS_UPDATE, response)
                
            else:
                self.logger.warning(f"Unknown router message: {elem}")
                
        except Exception as e:
            self.logger.error(f"Error processing router message: {e}")
    
    async def send_router_response(self, elem: str, data: dict):
        """Send a response back through the router."""
        try:
            from .zmq_protocol import ZMQMessage
            
            message = ZMQMessage.create_response(
                message_type=getattr(MessageType, elem.upper(), MessageType.STATUS_UPDATE),
                client_id=self.mq_client.client_id,
                data=data
            )
            
            await self.mq_client.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending router response: {e}")
    
    async def stop(self):
        """Stop the headless server."""
        self.shutdown_event.set()


def main():
    """Main entry point for the headless server."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI Hydra Headless Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server on default port
  python -m ai_hydra.headless_server
  
  # Start server on custom port with debug logging
  python -m ai_hydra.headless_server --bind "tcp://*:6666" --log-level DEBUG
  
  # Start server with log file
  python -m ai_hydra.headless_server --log-file /var/log/snake_ai.log
  
  # Start server for remote connections
  python -m ai_hydra.headless_server --bind "tcp://0.0.0.0:5555"
        """
    )
    
    parser.add_argument(
        "--bind", 
        default="tcp://*:5555",
        help="ZeroMQ bind address (default: tcp://*:5555)"
    )
    
    parser.add_argument(
        "--heartbeat", 
        type=float, 
        default=5.0,
        help="Heartbeat interval in seconds (default: 5.0)"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        help="Optional log file path"
    )
    
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon (detach from terminal)"
    )
    
    args = parser.parse_args()
    
    # Handle daemon mode
    if args.daemon:
        try:
            import daemon
            import daemon.pidfile
            
            # Create daemon context
            with daemon.DaemonContext(
                pidfile=daemon.pidfile.PIDLockFile('/var/run/snake_ai_server.pid'),
                stdout=open('/var/log/snake_ai_stdout.log', 'w+'),
                stderr=open('/var/log/snake_ai_stderr.log', 'w+'),
            ):
                # Run server as daemon
                server = HeadlessServer(
                    bind_address=args.bind,
                    heartbeat_interval=args.heartbeat,
                    log_level=args.log_level,
                    log_file=args.log_file or '/var/log/snake_ai.log'
                )
                asyncio.run(server.start())
                
        except ImportError:
            print("Daemon mode requires 'python-daemon' package: pip install python-daemon")
            sys.exit(1)
        except Exception as e:
            print(f"Failed to start daemon: {e}")
            sys.exit(1)
    else:
        # Run in foreground
        server = HeadlessServer(
            bind_address=args.bind,
            heartbeat_interval=args.heartbeat,
            log_level=args.log_level,
            log_file=args.log_file
        )
        
        try:
            print("🚀 Starting AI Hydra Server...")
            print("📡 Press Ctrl+C to stop")
            asyncio.run(server.start())
        except KeyboardInterrupt:
            print("\n🛑 Server interrupted by user")
            sys.exit(0)  # Exit with success code for Ctrl+C
        except Exception as e:
            print(f"❌ Server failed: {e}")
            sys.exit(1)
        finally:
            print("👋 Server shutdown complete")


if __name__ == "__main__":
    main()