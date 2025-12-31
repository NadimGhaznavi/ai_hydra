Router Message Protocol Fix
===========================

This document describes the design and implementation of the router message protocol fix that addresses the message format mismatch between the AI Hydra router and MQClient components.

Problem Statement
-----------------

The AI Hydra system experienced a message format mismatch between the router and MQClient components:

* **Router Expectation**: RouterConstants format with ``elem`` field
* **MQClient Behavior**: ZMQMessage format with ``message_type`` field
* **Impact**: "Malformed message" errors when heartbeat messages were sent to the router
* **Root Cause**: Inconsistent message format standards between components

Solution Overview
-----------------

The fix implements a message format adapter within the MQClient that transparently converts between formats while maintaining backward compatibility:

.. code-block:: text

    Internal Components → ZMQMessage Format → MQClient → RouterConstants Format → Router
                                                ↑
                                        Format Conversion
                                            Layer

**Key Design Principles:**

* **Transparency**: Format conversion is handled automatically by MQClient
* **Backward Compatibility**: Internal components continue using ZMQMessage format unchanged
* **Centralized Conversion**: All format conversion logic is contained within MQClient
* **Error Resilience**: Comprehensive error handling and validation

Architecture
------------

Message Format Mapping
~~~~~~~~~~~~~~~~~~~~~~~

The format adapter maps between two message structures:

.. list-table:: Message Format Mapping
   :header-rows: 1
   :widths: 30 30 40

   * - ZMQMessage Field
     - RouterConstants Field
     - Notes
   * - ``message_type``
     - ``elem``
     - Direct mapping of message type
   * - ``client_id``
     - ``client_id``
     - Preserved unchanged
   * - ``timestamp``
     - ``timestamp``
     - Preserved unchanged
   * - ``data``
     - ``data``
     - Preserved unchanged
   * - N/A
     - ``sender``
     - Added from client configuration

Format Conversion Process
~~~~~~~~~~~~~~~~~~~~~~~~~

**Outgoing Message Conversion (ZMQMessage → RouterConstants):**

.. code-block:: python

    def _convert_to_router_format(self, message: ZMQMessage) -> Dict[str, Any]:
        """Convert ZMQMessage to RouterConstants format."""
        return {
            "sender": self.client_type,  # Added from client config
            "elem": self._map_message_type_to_elem(message.message_type),
            "data": message.data or {},
            "client_id": self.client_id,
            "timestamp": message.timestamp,
            "request_id": message.request_id
        }

**Incoming Message Conversion (RouterConstants → ZMQMessage):**

.. code-block:: python

    def _convert_from_router_format(self, router_message: Dict[str, Any]) -> ZMQMessage:
        """Convert RouterConstants format to ZMQMessage."""
        return ZMQMessage(
            message_type=self._map_elem_to_message_type(router_message["elem"]),
            client_id=router_message["client_id"],
            timestamp=router_message["timestamp"],
            request_id=router_message.get("request_id"),
            data=router_message.get("data", {})
        )

Message Type Mapping
~~~~~~~~~~~~~~~~~~~~~

The adapter maintains a bidirectional mapping between message types:

.. code-block:: python

    MESSAGE_TYPE_MAPPING = {
        MessageType.HEARTBEAT.value: RouterConstants.HEARTBEAT,
        MessageType.START_SIMULATION.value: RouterConstants.START_SIMULATION,
        MessageType.STOP_SIMULATION.value: RouterConstants.STOP_SIMULATION,
        MessageType.PAUSE_SIMULATION.value: RouterConstants.PAUSE_SIMULATION,
        MessageType.RESUME_SIMULATION.value: RouterConstants.RESUME_SIMULATION,
        MessageType.GET_STATUS.value: RouterConstants.GET_STATUS,
        # Additional mappings as needed
    }

Validation and Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Message Validation:**

.. code-block:: python

    def _validate_router_message(self, message: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate RouterConstants format compliance."""
        required_fields = ["sender", "elem", "client_id", "timestamp"]
        
        for field in required_fields:
            if field not in message:
                return False, f"Missing required field: {field}"
        
        if message["elem"] not in self.VALID_ELEMENTS:
            return False, f"Invalid elem value: {message['elem']}"
        
        return True, None

**Error Recovery:**

* **Format Conversion Errors**: Log error and raise exception with context
* **Validation Failures**: Provide specific error details for troubleshooting
* **Unknown Message Types**: Clear error messages for unsupported types
* **Network Errors**: Graceful handling with retry mechanisms

Implementation Details
----------------------

Heartbeat Message Fix
~~~~~~~~~~~~~~~~~~~~~

The primary issue was with heartbeat messages not using the correct format:

**Before (Incorrect):**

.. code-block:: python

    heartbeat_message = {
        "message_type": "HEARTBEAT",  # Wrong field name
        "client_id": self.client_id,
        "timestamp": time.time(),
        "data": {"status": "active"}
    }

**After (Correct):**

.. code-block:: python

    heartbeat_message = {
        "sender": self.client_type,   # Added sender field
        "elem": "HEARTBEAT",          # Correct field name
        "client_id": self.client_id,
        "timestamp": time.time(),
        "data": {"status": "active"}
    }

MQClient Integration
~~~~~~~~~~~~~~~~~~~~

The format conversion is integrated into MQClient's send and receive methods:

.. code-block:: python

    async def send_message(self, message: ZMQMessage) -> None:
        """Send message with automatic format conversion."""
        try:
            # Convert to router format
            router_message = self._convert_to_router_format(message)
            
            # Validate converted message
            is_valid, error = self._validate_router_message(router_message)
            if not is_valid:
                raise ValueError(f"Message validation failed: {error}")
            
            # Send to router
            await self._send_raw_message(router_message)
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            raise

    async def receive_message(self) -> Optional[ZMQMessage]:
        """Receive message with automatic format conversion."""
        try:
            # Receive from router
            router_message = await self._receive_raw_message()
            if not router_message:
                return None
            
            # Convert to internal format
            return self._convert_from_router_format(router_message)
            
        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            return None

Testing Strategy
----------------

The fix includes comprehensive testing at multiple levels:

Unit Tests
~~~~~~~~~~

* Test message format conversion functions
* Test validation logic for both formats
* Test error handling for invalid messages
* Test message type mapping bidirectionality

Property-Based Tests
~~~~~~~~~~~~~~~~~~~~

* Generate random ZMQMessages and test round-trip conversion
* Generate random RouterConstants messages and test validation
* Test heartbeat message processing with various client types
* Test error handling with malformed messages

Integration Tests
~~~~~~~~~~~~~~~~~

* Test full communication flow between MQClient and Router
* Test heartbeat message exchange without errors
* Test command/response cycles with format conversion
* Test error recovery and retry mechanisms

Migration and Deployment
------------------------

Backward Compatibility
~~~~~~~~~~~~~~~~~~~~~~

The fix maintains full backward compatibility:

* **Internal Components**: Continue using ZMQMessage format unchanged
* **Existing APIs**: No changes to public interfaces
* **Configuration**: No new configuration requirements
* **Data Formats**: All data payloads remain identical

Deployment Process
~~~~~~~~~~~~~~~~~~

1. **Update MQClient**: Deploy updated MQClient with format conversion
2. **Verify Heartbeats**: Confirm heartbeat messages no longer generate errors
3. **Test All Message Types**: Validate all command/response cycles work correctly
4. **Monitor Logs**: Check for any remaining format-related errors
5. **Performance Validation**: Ensure format conversion doesn't impact performance

Monitoring and Troubleshooting
-------------------------------

Logging Enhancements
~~~~~~~~~~~~~~~~~~~~

The fix includes enhanced logging for troubleshooting:

.. code-block:: python

    # Format conversion logging
    self.logger.debug(f"Converting message: {message.message_type} -> {router_message['elem']}")
    
    # Validation logging
    self.logger.debug(f"Message validation: {is_valid}, error: {error}")
    
    # Error context logging
    self.logger.error(f"Format conversion failed: {e}", extra={
        "original_message": message,
        "conversion_stage": "to_router_format"
    })

Performance Impact
~~~~~~~~~~~~~~~~~~

The format conversion has minimal performance impact:

* **Conversion Time**: < 1ms per message for typical payloads
* **Memory Overhead**: Negligible additional memory usage
* **Network Impact**: No change to network traffic patterns
* **CPU Usage**: Minimal additional CPU for format conversion

Troubleshooting Guide
~~~~~~~~~~~~~~~~~~~~~

**Common Issues:**

1. **"Missing required field" errors**: Check message structure and required fields
2. **"Invalid elem value" errors**: Verify message type mapping is complete
3. **Conversion failures**: Check for unsupported message types
4. **Heartbeat errors**: Verify client_type configuration is correct

**Diagnostic Commands:**

.. code-block:: bash

    # Check router logs for format errors
    grep -i "malformed" router.log
    
    # Verify heartbeat message format
    grep -i "heartbeat" client.log | tail -5
    
    # Monitor format conversion activity
    grep -i "converting message" client.log

This design ensures reliable, transparent message format conversion while maintaining system performance and backward compatibility.