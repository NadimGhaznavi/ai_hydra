"""
Hydra Router Command Line Interface

Provides command-line interface for starting and managing the Hydra Router.
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

from .router import HydraRouter, main_async


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai-hydra-router",
        description="Hydra Router - Standalone ZeroMQ message routing system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start router with default settings
  ai-hydra-router

  # Start router on custom address and port
  ai-hydra-router --address 192.168.1.100 --port 6666

  # Start router with debug logging
  ai-hydra-router --log-level DEBUG

  # Start router with custom configuration
  ai-hydra-router -a 0.0.0.0 -p 5556 --log-level INFO

For more information, visit: https://github.com/ai-hydra/hydra-router
        """,
    )

    parser.add_argument(
        "-a",
        "--address",
        default="0.0.0.0",
        help="IP address to bind router to (default: 0.0.0.0)",
        metavar="ADDRESS",
    )

    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=5556,
        help="Port to bind router to (default: 5556)",
        metavar="PORT",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
        metavar="LEVEL",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
        help="Show version information and exit",
    )

    return parser


def setup_logging(log_level: str) -> None:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level string
    """
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set ZMQ logger to WARNING to reduce noise
    zmq_logger = logging.getLogger("zmq")
    zmq_logger.setLevel(logging.WARNING)

    # Set asyncio logger to INFO to reduce debug noise
    asyncio_logger = logging.getLogger("asyncio")
    asyncio_logger.setLevel(logging.INFO)


def validate_arguments(args: argparse.Namespace) -> Optional[str]:
    """
    Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Error message if validation fails, None if valid
    """
    # Validate port range
    if not (1 <= args.port <= 65535):
        return f"Port must be between 1 and 65535, got {args.port}"

    # Validate address format (basic validation)
    if not args.address:
        return "Address cannot be empty"

    # Check for common invalid addresses
    if args.address in ["localhost", "127.0.0.1"] and args.port == 5556:
        # This is fine, just a note
        pass

    return None


def print_startup_banner(address: str, port: int, log_level: str) -> None:
    """
    Print startup banner with configuration information.

    Args:
        address: Router bind address
        port: Router bind port
        log_level: Logging level
    """
    print("=" * 60)
    print("🚀 Hydra Router - Standalone Message Routing System")
    print("=" * 60)
    print(f"📡 Binding to: {address}:{port}")
    print(f"📊 Log level: {log_level}")
    print(f"🔧 ZeroMQ Router Pattern: ROUTER socket")
    print(f"💡 Client connection: tcp://{address}:{port}")
    print("=" * 60)
    print("📋 Status: Starting router...")
    print("⏹️  Press Ctrl+C to stop")
    print()


def print_shutdown_message() -> None:
    """Print shutdown message."""
    print()
    print("=" * 60)
    print("🛑 Hydra Router Shutdown")
    print("=" * 60)
    print("✅ Router stopped successfully")
    print("👋 Thank you for using Hydra Router!")
    print()


async def run_router(address: str, port: int, log_level: str) -> int:
    """
    Run the router with error handling.

    Args:
        address: Router bind address
        port: Router bind port
        log_level: Logging level

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        await main_async(address, port, log_level)
        return 0
    except KeyboardInterrupt:
        logging.getLogger("HydraRouter").info("Router shutdown requested by user")
        return 0
    except Exception as e:
        logging.getLogger("HydraRouter").error(f"Router failed: {e}", exc_info=True)
        print(f"\n❌ Router failed: {e}")
        return 1


def main() -> int:
    """
    Main entry point for the CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    validation_error = validate_arguments(args)
    if validation_error:
        print(f"❌ Error: {validation_error}", file=sys.stderr)
        return 1

    # Setup logging
    setup_logging(args.log_level)

    # Print startup information
    print_startup_banner(args.address, args.port, args.log_level)

    try:
        # Run the router
        exit_code = asyncio.run(run_router(args.address, args.port, args.log_level))

        # Print shutdown message
        print_shutdown_message()

        return exit_code

    except KeyboardInterrupt:
        print_shutdown_message()
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
