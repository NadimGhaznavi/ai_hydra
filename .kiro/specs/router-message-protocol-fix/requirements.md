# Requirements Document

## Introduction

The AI Hydra system currently has a message format mismatch between the router and MQClient components. The router expects messages with an `elem` field following the RouterConstants format, but the MQClient is sending messages with a `message_type` field following the ZMQMessage format. This causes "Malformed message" errors when the server tries to send heartbeat messages to the router.

## Glossary

- **Router**: The AI Hydra router that routes messages between clients and servers using RouterConstants format
- **MQClient**: The ZeroMQ client that connects to the router and sends/receives messages
- **ZMQMessage**: The message format used by the ZMQ protocol with `message_type` field
- **RouterConstants**: The message format expected by the router with `elem` field
- **Heartbeat_Message**: Periodic messages sent to indicate client/server is alive
- **Message_Format_Adapter**: Component that converts between ZMQMessage and RouterConstants formats
- **Protocol_Validator**: Component that validates message format compliance

## Requirements

### Requirement 1: Message Format Standardization

**User Story:** As a system integrator, I want consistent message formats between router and clients, so that all components can communicate without format errors.

#### Acceptance Criteria

1. THE Router SHALL accept messages in RouterConstants format with `sender`, `elem`, and `data` fields
2. THE MQClient SHALL send messages in RouterConstants format when communicating with the router
3. THE Message_Format_Adapter SHALL convert ZMQMessage format to RouterConstants format before sending to router
4. THE Protocol_Validator SHALL validate message format compliance before processing
5. THE system SHALL maintain backward compatibility with existing ZMQMessage format for internal components

### Requirement 2: Heartbeat Message Protocol Compliance

**User Story:** As a system administrator, I want heartbeat messages to be properly formatted and processed, so that the router can track client/server connectivity without errors.

#### Acceptance Criteria

1. WHEN MQClient sends a heartbeat message, THE message SHALL use RouterConstants.HEARTBEAT as the `elem` field
2. THE heartbeat message SHALL include `sender`, `client_id`, `timestamp`, and `data` fields in RouterConstants format
3. THE Router SHALL process heartbeat messages without generating "Malformed message" errors
4. THE Router SHALL update client tracking information when receiving valid heartbeat messages
5. THE heartbeat message SHALL maintain the same timing interval and content as before the format fix

### Requirement 3: Bidirectional Message Format Conversion

**User Story:** As a developer, I want seamless conversion between message formats, so that internal components can continue using ZMQMessage while external communication uses RouterConstants.

#### Acceptance Criteria

1. THE Message_Format_Adapter SHALL convert outgoing ZMQMessage to RouterConstants format for router communication
2. THE Message_Format_Adapter SHALL convert incoming RouterConstants format to ZMQMessage for internal processing
3. THE conversion SHALL preserve all message content including data, timestamps, and identifiers
4. THE Message_Format_Adapter SHALL handle all message types including commands, responses, and broadcasts
5. THE conversion process SHALL not introduce message loss or corruption

### Requirement 4: Error Handling and Validation

**User Story:** As a system operator, I want clear error messages and validation, so that I can quickly identify and resolve communication issues.

#### Acceptance Criteria

1. WHEN a message fails format validation, THE Protocol_Validator SHALL provide specific error details about missing or incorrect fields
2. THE Router SHALL log detailed information about malformed messages including expected vs actual format
3. THE MQClient SHALL retry message sending with correct format when receiving format error responses
4. THE system SHALL gracefully handle format conversion failures without crashing components
5. THE error messages SHALL include enough context to identify the source component and message type

### Requirement 5: Backward Compatibility and Migration

**User Story:** As a system maintainer, I want smooth migration to the new message format, so that existing functionality continues to work during the transition.

#### Acceptance Criteria

1. THE Router SHALL continue to accept both old and new message formats during migration period
2. THE MQClient SHALL detect router capabilities and use appropriate message format
3. THE system SHALL log format conversion activities for monitoring migration progress
4. THE internal ZMQMessage format SHALL remain unchanged for components not communicating with router
5. THE migration SHALL not require simultaneous updates to all system components