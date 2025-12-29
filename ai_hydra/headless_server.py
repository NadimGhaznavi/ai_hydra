#!/usr/bin/env python3
"""
Headless server entry point for AI Hydra.

This script starts the ZeroMQ server in headless mode, making the AI agent
completely controllable via ZeroMQ messages without any GUI dependencies.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from .zmq_server import ZMQServer


class HeadlessServer:
    """
    Headless server wrapper that handles graceful shutdown and logging.
    """
    
    def __init__(self, bind_address: str = "tcp://*:5555", 
                 heartbeat_interval: float = 5.0,
                 log_level: str = "INFO",
                 log_file: str = None):
        """
        Initialize the headless server.
        
        Args:
            bind_address: ZeroMQ bind address
            heartbeat_interval: Seconds between heartbeat broadcasts
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional log file path
        """
        self.bind_address = bind_address
        self.heartbeat_interval = heartbeat_interval
        self.log_level = log_level
        self.log_file = log_file
        
        self.server = None
        self.shutdown_event = asyncio.Event()
        
        # Setup logging
        self._setup_logging()
        
        self.logger = logging.getLogger("headless_server")
    
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
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        # Handle SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self):
        """Start the headless server."""
        self.logger.info("Starting AI Hydra Headless Server")
        self.logger.info(f"Bind Address: {self.bind_address}")
        self.logger.info(f"Heartbeat Interval: {self.heartbeat_interval}s")
        self.logger.info(f"Log Level: {self.log_level}")
        if self.log_file:
            self.logger.info(f"Log File: {self.log_file}")
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Create and start ZeroMQ server
        self.server = ZMQServer(
            bind_address=self.bind_address,
            heartbeat_interval=self.heartbeat_interval
        )
        
        try:
            # Start server in background
            server_task = asyncio.create_task(self.server.start())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            self.logger.info("Shutdown signal received, stopping server...")
            
            # Cancel server task
            server_task.cancel()
            
            try:
                await server_task
            except asyncio.CancelledError:
                pass
            
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise
        finally:
            if self.server:
                await self.server.stop()
            
            self.logger.info("Headless server stopped")
    
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
            asyncio.run(server.start())
        except KeyboardInterrupt:
            print("\nServer interrupted by user")
        except Exception as e:
            print(f"Server failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()