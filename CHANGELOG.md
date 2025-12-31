# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Token Tracker System**: Complete implementation of comprehensive token usage tracking system
  - Core data models (TokenTransaction, TrackerConfig) with validation and CSV serialization
  - Thread-safe CSV operations with file locking and concurrent access protection
  - Token tracker service with validation, error handling, and transaction history
  - Special character and Unicode handling for international text support
  - Comprehensive property-based testing with universal correctness properties
  - Metadata collection system for workspace and execution context gathering
  - Agent hook integration with automatic token tracking on AI interactions
  - Configuration management with runtime updates and state consistency
  - Debug utilities for file patterns preservation testing
  - **Maintenance Utilities**: Comprehensive maintenance system for token tracking
    - MaintenanceManager class for CSV file rotation and cleanup operations
    - Automatic file rotation based on configurable size and row count limits (50MB or 100k rows)
    - Data compression support for archived files with gzip compression
    - Cleanup functionality for old data based on retention policies
    - Archive statistics and maintenance recommendations system
    - Thread-safe operations with proper error handling and recovery
    - Comprehensive maintenance API with rotation, cleanup, and monitoring capabilities
  - **System Monitoring**: Complete monitoring and health checking system
    - SystemMonitor class for performance metrics collection and health checks
    - Real-time monitoring with configurable intervals and alert callbacks
    - Performance metrics tracking including memory usage, CPU utilization, and transaction rates
    - Health check utilities with comprehensive system validation
    - Export capabilities for monitoring data in multiple formats
  - **Documentation Restructuring**: Complete documentation system reorganization
    - Restructured docs/_source/ with architecture/, end_user/, and runbook/ directories
    - Comprehensive token tracking documentation with usage guides and troubleshooting
    - Updated main documentation index with clear navigation structure
    - Property-based testing documentation with requirements traceability
- **Integration Testing**: Complete integration test suite for token tracking system
  - Comprehensive system workflow tests covering end-to-end token tracking operations
  - Token tracker integration tests with hook validation and metadata accuracy
  - Error scenario testing and recovery mechanism validation
  - Hook integration testing with real Kiro IDE event simulation

### Changed
- **Documentation**: Comprehensive updates for token tracking system
  - Updated token tracking documentation with configuration management features
  - Enhanced testing documentation with new property-based tests
  - Updated requirements documentation with extended agent hook capabilities
  - Added API reference documentation for TokenTrackingHook class
  - Complete comprehensive token tracking documentation runbook with usage examples, troubleshooting guides, FAQ, and diagnostic tools
  - Added deployment and maintenance procedures documentation
  - Created SDLC procedures documentation with operational guidelines
  - **Documentation Restructuring**: Complete reorganization of documentation system
    - Restructured docs/_source/ directory with architecture/, end_user/, and runbook/ categories
    - Updated main index.rst with clear navigation and project overview
    - Migrated existing documentation to appropriate categories without content loss
    - Enhanced documentation build process and cross-reference validation
- **Project Management**: Updated token tracker implementation task status
  - Tasks 1-14 (Core implementation through final integration) marked as completed
  - Task 15 (Final system validation) currently in progress
  - Comprehensive documentation implementation finished
  - All major system components implemented and tested
  - Final validation phase initiated for production readiness assessment

### Fixed
- **Token Tracker**: Enhanced error recovery and resilience mechanisms
  - Graceful handling of file system errors and permission issues
  - Robust CSV integrity validation and corruption prevention
  - Improved metadata collection with fallback for missing context

## [Release 0.7.1] - 2025-12-30 23:37


### Changed
- **Project Structure**: Complete cleanup of root directory removing legacy files
  - Removed `setup.py` (legacy packaging file, redundant with `pyproject.toml`)
  - Removed `qasync.sh`, `manual_test_server.py`, `test.log`, `accounting.txt`
  - Moved test files from root to `tests/` directory for better organization
- **Documentation**: Updated version update procedure to remove setup.py references
  - Updated `docs/_source/version_update_procedure.rst` to reflect current packaging approach
  - Removed all legacy setup file handling from documentation
- **Version Scripts**: Enhanced `update_version.sh` to remove setup.py handling
  - Added deprecation warnings if setup.py is found
  - Focused script on modern `pyproject.toml` packaging approach
- **Development Environment**: Enhanced `.gitignore` with comprehensive exclusions
  - Added coverage reports, testing artifacts, build artifacts
  - Added IDE files and OS generated files exclusions

## [Release 0.7.0] - 2025-12-30 23:04


### Added

#### Router Architecture System
- **Complete Router Implementation**: Comprehensive client/server router system based on ai_snake_lab SimRouter pattern
  - `HydraRouter` class (`ai_hydra/router.py`) with ZeroMQ ROUTER socket for message routing
  - `MQClient` generic class (`ai_hydra/mq_client.py`) for router communication with both client and server support
  - `RouterConstants` (`ai_hydra/router_constants.py`) for centralized message types and network configuration
  - Support for multiple clients connecting to single server through centralized router
  - Heartbeat-based client registration and automatic management (5-second intervals)
  - Automatic inactive client detection and removal (15-second timeout)
  - `ai-hydra-router` CLI entry point for standalone router operation with configurable address/port
  - Network architecture: `[TUI Client] ←→ [Router:5556] ←→ [Headless Server]`

#### Router Features
- **Client Registration**: Automatic client registration via heartbeat messages
- **Message Routing**: Intelligent routing based on sender type (HydraClient/HydraServer)
- **Heartbeat Management**: Background task for inactive client detection and removal
- **Error Handling**: Graceful error handling with informative error messages
- **Scalability**: Support for multiple clients per server with distributed deployment
- **Background Tasks**: Proper async task management with cleanup

#### MQClient Features
- **Connection Management**: Automatic connection and heartbeat with reconnection support
- **Message Types**: Commands, responses, and broadcasts with structured protocol
- **Timeout Handling**: Configurable operation timeouts with fallback behavior
- **Context Management**: Python context manager support for resource cleanup
- **Error Recovery**: Graceful error handling and automatic cleanup

#### Message Protocol
- **Standardized Format**: JSON message structure with sender, client_id, message_type, timestamp, request_id, data
- **Control Commands**: start_simulation, stop_simulation, pause_simulation, resume_simulation, reset_simulation
- **Status Messages**: get_status, status_update, game_state_update, performance_update
- **System Messages**: heartbeat, error, ok acknowledgments

#### TUI Epoch Display Feature
- **Epoch Counter Display**: Added "Epoch: N" display in TUI status widget between Snake Length and Runtime
- **Reactive Updates**: Real-time epoch updates using Textual's reactive variable system
- **Status Processing**: Enhanced status update processing to extract epoch from game_state data
- **Reset Integration**: Epoch resets to 0 when simulation is reset
- **Demo Enhancement**: Updated `demo_tui.py` with epoch progression (increments every 20 moves)
- **Production Integration**: Works with production `ai-hydra-tui` command via `pip install ai-hydra[tui]`

#### Comprehensive Test Suite
- **Unit Tests**: Individual component functionality testing
  - `tests/unit/test_mq_client.py`: MQClient functionality validation
  - `tests/unit/test_router.py`: HydraRouter functionality validation
  - `tests/unit/test_router_constants.py`: Constants and configuration validation
  - `tests/unit/test_tui_status_display.py`: TUI status display functionality
- **Property-Based Tests**: Universal behavior validation with Hypothesis
  - `tests/property/test_router_properties.py`: Router behavior properties
  - `tests/property/test_tui_epoch_display.py`: Epoch display properties (100+ test cases)
- **Integration Tests**: Component interaction validation
  - `tests/integration/test_router_integration.py`: Router component integration
  - `tests/integration/test_tui_epoch_integration.py`: End-to-end TUI epoch workflow
- **End-to-End Tests**: Complete workflow validation
  - `tests/e2e/test_router_system.py`: Complete router system workflows

#### CLI Commands
- **Router Command**: `ai-hydra-router --address 0.0.0.0 --port 5556 --log-level INFO`
- **Updated Server Command**: `ai-hydra-server --router tcp://localhost:5556` (connects to router)
- **Updated TUI Command**: `ai-hydra-tui --router tcp://localhost:5556` (connects to router)
- **Remote Deployment Support**: Full support for distributed deployment across multiple machines

#### Enhanced Error Handling
- **TUI Dependencies**: Better error messages for missing TUI dependencies
  - Graceful fallback when `textual` package not installed
  - Helpful installation instructions for optional dependencies (`pip install ai-hydra[tui]`)
- **Router Error Handling**: Connection failures, malformed messages, resource cleanup
- **Client Error Handling**: Connection loss recovery, timeout handling, message validation

### Changed

#### Network Architecture Transformation
- **Headless Server**: Updated `ai_hydra/headless_server.py` to use MQClient and connect to router instead of direct binding
- **TUI Client**: Updated `ai_hydra/tui/client.py` to use MQClient for router communication instead of direct server connection
- **Connection Model**: Transformed from direct client-server to router-based messaging system
  - Clients now connect to router at port 5556 instead of directly to server
  - Router handles intelligent message routing between clients and servers
  - Supports distributed deployment across multiple machines with remote router access

#### Status Display Enhancement
- **Status Widget Layout**: Added epoch display between Snake Length and Runtime in TUI status panel
- **Data Flow**: Enhanced `process_status_update()` to extract and process epoch information
- **UI Architecture**: Integrated epoch display with Textual's reactive variable system

#### CLI Integration
- **Entry Points**: Added `ai-hydra-router` to `pyproject.toml` CLI entry points
- **Command Arguments**: Enhanced server and TUI commands with `--router` parameter for router address
- **Deployment Flexibility**: Support for local and remote router deployments

### Fixed

#### Dependency Management
- **TUI Dependencies**: Fixed ModuleNotFoundError when textual package not installed
- **Import Handling**: Added graceful error handling for optional TUI dependencies
- **Installation Instructions**: Clear guidance for installing TUI support via `pip install ai-hydra[tui]`

#### Communication Protocol
- **Message Protocol**: Standardized message format across all components with proper JSON structure
- **Connection Management**: Improved connection handling and cleanup with proper resource management
- **Error Propagation**: Enhanced error handling and informative error messages throughout the system

#### Testing and Quality
- **Test Coverage**: 13/13 router constants tests passing with comprehensive edge case coverage
- **Property Testing**: Universal behavior validation with Hypothesis for robust edge case discovery
- **Integration Validation**: Complete component interaction testing with mock-based isolation
- **Requirements Compliance**: Full validation of Requirements 3.5 and 3.6 for epoch display feature

### Technical Details

#### Performance and Scalability
- **Message Throughput**: Efficient message routing with minimal latency
- **Resource Management**: Automatic cleanup of inactive clients and proper memory management
- **Concurrent Clients**: Support for multiple concurrent clients with heartbeat-based tracking
- **Async Operations**: Full async/await support for non-blocking operations

#### Security and Reliability
- **Network Security**: Configurable bind addresses and controlled error information disclosure
- **Input Validation**: Message validation to prevent malformed messages and resource exhaustion
- **Connection Security**: Framework for future authentication and encryption features
- **Monitoring**: Comprehensive logging for security monitoring and debugging

#### Code Quality Standards
- **Type Hints**: Comprehensive type annotations throughout all new components
- **Documentation**: Complete docstrings with Google style formatting
- **Error Handling**: Robust error handling with graceful degradation
- **Testing Standards**: Property-based testing with requirements traceability
- **Async Patterns**: Proper async programming patterns with resource cleanup

---

## [0.1.0] - Initial Release

### Added
- Initial AI Hydra implementation
- Neural network + tree search hybrid system
- Game board and logic implementation
- Configuration management system
- Basic TUI client interface
- Headless server implementation
- Comprehensive testing framework
- Documentation and code style standards

---
