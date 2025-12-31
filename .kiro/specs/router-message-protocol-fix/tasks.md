# Implementation Plan: Router Message Protocol Fix

## Overview

This implementation plan fixes the message format mismatch between the AI Hydra router and MQClient components. The router expects RouterConstants format with `elem` field, but MQClient sends ZMQMessage format with `message_type` field. The fix is primarily implemented in MQClient with minimal router changes.

## Tasks

- [x] 1. Update MQClient message format conversion
  - Add message type to RouterConstants elem mapping
  - Implement format conversion methods in MQClient
  - Update heartbeat message format to use `elem` field
  - _Requirements: 1.2, 2.1, 2.2_

- [x] 1.1 Write property test for message format conversion
  - **Property 1: Message Format Round-Trip Conversion**
  - **Validates: Requirements 1.3, 3.1, 3.2, 3.3, 3.5**

- [x] 2. Fix heartbeat message format in MQClient
  - Modify `_send_heartbeat()` method to use RouterConstants format
  - Ensure heartbeat includes `sender`, `elem`, `data`, `client_id`, `timestamp` fields
  - Test heartbeat messages are processed without "Malformed message" errors
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 2.1 Write property test for heartbeat message processing
  - **Property 3: Heartbeat Message Processing**
  - **Validates: Requirements 2.3, 2.4, 2.5**

- [ ] 3. Add message validation and error handling
  - Implement message format validation in MQClient
  - Add error handling for conversion failures
  - Ensure graceful handling of unsupported message types
  - _Requirements: 1.4, 4.1, 4.4_

- [ ] 3.1 Write property test for format validation
  - **Property 2: RouterConstants Format Compliance**
  - **Validates: Requirements 1.1, 1.2, 2.1, 2.2**

- [ ] 3.2 Write property test for error handling
  - **Property 4: Format Validation Error Reporting**
  - **Validates: Requirements 4.1, 4.2, 4.5**

- [ ] 4. Enhance router error logging
  - Improve malformed message error messages in router
  - Add detailed logging for debugging format issues
  - Include expected vs actual format information in logs
  - _Requirements: 4.2_

- [ ] 5. Test integration and backward compatibility
  - Verify MQClient can send all message types in RouterConstants format
  - Test that internal components still use ZMQMessage format unchanged
  - Ensure no breaking changes to existing functionality
  - _Requirements: 1.5, 5.4_

- [ ] 5.1 Write property test for backward compatibility
  - **Property 5: Backward Compatibility Preservation**
  - **Validates: Requirements 1.5, 5.1, 5.4**

- [ ] 5.2 Write property test for error resilience
  - **Property 6: Error Resilience**
  - **Validates: Requirements 4.3, 4.4**

- [ ] 5.3 Write property test for message type coverage
  - **Property 7: Message Type Coverage**
  - **Validates: Requirements 3.4**

- [ ] 6. Checkpoint - Verify heartbeat messages work correctly
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Integration testing with router
  - Test complete message flow between MQClient and Router
  - Verify heartbeat messages no longer generate "Malformed message" errors
  - Test command/response cycles work correctly
  - _Requirements: 2.3, 2.4_

- [ ] 7.1 Write integration tests for router communication
  - Test full communication flow with format conversion
  - Test error recovery and retry mechanisms

- [ ] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks include comprehensive testing from the start for robust implementation
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- The main fix is updating MQClient's `_send_heartbeat()` method to use RouterConstants format