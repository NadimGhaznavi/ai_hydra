#!/usr/bin/env python3
"""
Test script to verify that log level propagation works correctly
throughout the AI Hydra server components.
"""

import asyncio
import logging
import sys
import time
from ai_hydra.headless_server import HeadlessServer
from ai_hydra.zmq_server import ZMQServer
from ai_hydra.config import LoggingConfig
from ai_hydra.logging_config import SimulationLogger


def test_log_level_propagation():
    """Test that log levels propagate correctly through all components."""
    print("Testing log level propagation...")
    
    # Test 1: Direct SimulationLogger with different levels
    print("\n1. Testing SimulationLogger with different levels:")
    
    # Test DEBUG level
    debug_config = LoggingConfig(level="DEBUG")
    debug_logger = SimulationLogger("test_debug", debug_config)
    
    # Test INFO level  
    info_config = LoggingConfig(level="INFO")
    info_logger = SimulationLogger("test_info", info_config)
    
    # Test WARNING level
    warning_config = LoggingConfig(level="WARNING")
    warning_logger = SimulationLogger("test_warning", warning_config)
    
    print(f"DEBUG logger level: {debug_logger.logger.level} (should be {logging.DEBUG})")
    print(f"INFO logger level: {info_logger.logger.level} (should be {logging.INFO})")
    print(f"WARNING logger level: {warning_logger.logger.level} (should be {logging.WARNING})")
    
    # Test 2: Root logger inheritance
    print("\n2. Testing root logger inheritance:")
    
    # Set root logger to WARNING
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.WARNING)
    
    # Create new logger - should inherit WARNING level
    inherited_logger = logging.getLogger("ai_hydra.test_inherited")
    print(f"Root logger level: {root_logger.level} (WARNING = {logging.WARNING})")
    print(f"Inherited logger effective level: {inherited_logger.getEffectiveLevel()}")
    
    # Restore original level
    root_logger.setLevel(original_level)
    
    # Test 3: HeadlessServer configuration
    print("\n3. Testing HeadlessServer log level configuration:")
    
    # Test different log levels
    for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
        print(f"\nTesting {level} level:")
        server = HeadlessServer(log_level=level)
        
        # Check that root logger was configured correctly
        root_level = logging.getLogger().level
        expected_level = getattr(logging, level)
        print(f"  Root logger level: {root_level} (expected: {expected_level})")
        print(f"  Match: {root_level == expected_level}")
        
        # Check server's own logger
        server_level = server.logger.getEffectiveLevel()
        print(f"  Server logger effective level: {server_level}")
    
    print("\n4. Testing ZMQServer log level inheritance:")
    
    # Set root logger to DEBUG for this test
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Create ZMQServer with DEBUG level
    zmq_server = ZMQServer(log_level="DEBUG")
    print(f"ZMQServer logger level: {zmq_server.logger.getEffectiveLevel()}")
    print(f"ZMQServer log_level attribute: {zmq_server.log_level}")
    
    print("\nLog level propagation test completed!")


def test_actual_logging_output():
    """Test that different log levels actually produce different output."""
    print("\n" + "="*50)
    print("TESTING ACTUAL LOG OUTPUT")
    print("="*50)
    
    # Configure root logger to INFO
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    
    # Create loggers with hierarchical names
    debug_logger = logging.getLogger("ai_hydra.test_debug")
    info_logger = logging.getLogger("ai_hydra.test_info") 
    warning_logger = logging.getLogger("ai_hydra.test_warning")
    
    print("\nWith root logger at INFO level, testing different message levels:")
    print("(DEBUG messages should NOT appear, INFO and above should appear)")
    
    debug_logger.debug("This DEBUG message should NOT appear")
    debug_logger.info("This INFO message should appear")
    debug_logger.warning("This WARNING message should appear")
    
    info_logger.debug("This DEBUG message should NOT appear")
    info_logger.info("This INFO message should appear") 
    info_logger.warning("This WARNING message should appear")
    
    warning_logger.debug("This DEBUG message should NOT appear")
    warning_logger.info("This INFO message should appear")
    warning_logger.warning("This WARNING message should appear")
    
    print("\nNow changing root logger to WARNING level:")
    logging.getLogger().setLevel(logging.WARNING)
    
    print("(Only WARNING and above should appear)")
    debug_logger.debug("This DEBUG message should NOT appear")
    debug_logger.info("This INFO message should NOT appear")
    debug_logger.warning("This WARNING message should appear")
    debug_logger.error("This ERROR message should appear")


if __name__ == "__main__":
    test_log_level_propagation()
    test_actual_logging_output()