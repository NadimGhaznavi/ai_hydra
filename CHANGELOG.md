# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Router Enhancement**: Enhanced router error logging and debugging capabilities
  - Added comprehensive message format validation with detailed error reporting
  - Implemented enhanced malformed message logging with expected vs actual format comparison
  - Added detailed frame error logging for ZMQ communication issues
  - Implemented JSON parsing error logging with debugging hints
  - Added unknown sender type error logging with valid options display
  - Provided comprehensive debugging information for all error types
  - Enhanced router troubleshooting capabilities for message format issues
- **Documentation**: Router message protocol fix documentation
  - Created comprehensive router message protocol fix design document (docs/_source/architecture/router_message_protocol_fix.rst)
  - Added detailed problem statement, solution overview, and architecture documentation
  - Documented message format mapping between ZMQMessage and RouterConstants formats
  - Included format conversion process, validation, error handling, and testing strategy
  - Added migration and deployment guidance with backward compatibility considerations
- **Documentation Updates**: Comprehensive documentation status tracking and implementation progress
  - Updated architecture documentation with implementation status indicators
  - Added "Current Implementation Status" section to router message protocol fix documentation
  - Updated heartbeat message format section with "IN PROGRESS" status tracking
  - Enhanced MQClient integration documentation with current implementation details
  - Added implementation notes showing actual RouterConstants format usage
  - Updated architecture.rst with MQClient and Message Format Adapter status indicators
  - Enhanced ZMQ protocol documentation with implementation status and completion checkmarks
  - Updated specification documents with comprehensive implementation status tracking
  - Added status indicators to requirements documentation (Requirements 31-35)
  - Implemented consistent status tracking across all documentation files
  - Added progress visibility with ✅ **IMPLEMENTED**, 🔄 **IN PROGRESS**, and ⏳ **PENDING** indicators
  - Enhanced requirement tracking with completion status for all acceptance criteria
  - Updated CHANGELOG.md with router message protocol fix implementation status
  - Provided clear implementation progress visibility across all documentation

### Changed
- **Documentation**: Updated ZeroMQ protocol documentation for message format standardization
  - Enhanced message structure documentation to include both RouterConstants and ZMQMessage formats
  - Added message format conversion section with automatic conversion details
  - Updated heartbeat message documentation to specify RouterConstants format requirement
  - Added Message Format Adapter section with conversion process and error handling details
- **Documentation**: Updated requirements specification with router message protocol fix requirements
  - Added Requirements 31-35 covering message format standardization, heartbeat protocol compliance
  - Defined bidirectional message format conversion and error handling requirements
  - Established message format migration and backward compatibility requirements
- **Documentation**: Updated architecture documentation with message format adapter information
- **Specifications**: Updated router message protocol fix task status to reflect active implementation progress
  - Marked task 2 (heartbeat message format fix) as in-progress in router-message-protocol-fix spec
  - Indicates ongoing work on MQClient _send_heartbeat() method RouterConstants format implementation
  - Enhanced MQClient description to include built-in message format conversion capabilities
  - Added Message Format Adapter component description with transparent operation details
  - Updated RouterConstants description to clarify router message format expectations
  - **Implementation Status**: Task 2 heartbeat message format fix is currently being implemented in MQClient
- **Specifications**: Added comprehensive requirements documentation for router message protocol fix
  - Created detailed requirements document with 5 main requirements covering message format standardization
  - Defined acceptance criteria for heartbeat protocol compliance and bidirectional message conversion
  - Established error handling, validation, and backward compatibility requirements
  - Added glossary of key terms and components for router-MQClient communication protocol
- **Specifications**: Updated task 3.1 status in router message protocol fix specification
  - Changed task 3.1 from completed to in-progress status to reflect current implementation state
  - Updated format validation property test status tracking for accurate project management
  - Task 3.1 (Property test for format validation) status updated to reflect ongoing implementation work
  - Property 2: RouterConstants Format Compliance validation requires additional implementation

## [Release 0.8.0] - 2025-12-31 17:08


### Added
- **Project Organization**: Comprehensive directory structure reorganization and standardization
  - Created standardized directory structure with `scripts/`, `tools/`, and organized `tests/` directories
  - Established `tools/` directory with categorized development utilities:
    - `debug/` - Debugging utilities (debug_file_patterns.py)
    - `testing/` - Testing utilities (run_tests.py)
    - `documentation/` - Documentation tools (test_documentation.py, validate_docs.py)
    - `analysis/` - Analysis utilities (ready for future use)
  - Reorganized test suite by type for better maintainability:
    - `unit/` - Unit tests (6 existing + 6 moved files)
    - `property/` - Property-based tests (10 moved files)
    - `integration/` - Integration tests (4 existing + 5 moved files)
    - `e2e/` - End-to-end tests (ready for future use)
  - Created `scripts/` directory for build and maintenance scripts (update_version.sh)
  - Added comprehensive file organization utility script (`scripts/organize_files.sh`)
  - Updated file permissions for all executable scripts and tools
  - Moved documentation files to appropriate locations (DOCUMENTATION_UPDATE_SUMMARY.md → docs/)
  - Removed empty/unused files (debug_property_test.py)
- **Directory Layout Standards**: Comprehensive project organization standards and guidelines
  - Created directory-layout-standards.md in .kiro/steering/ with complete project structure definitions
  - Established standard directory structure for ai_hydra/ main package and ai_snake_lab/ legacy code
  - Defined testing organization with unit/, property/, integration/, and e2e/ test categories
  - Documented file naming conventions for Python files, tests, documentation, and configuration
  - Provided directory creation guidelines and migration procedures for reorganization
  - Included automation tools for structure validation and import path checking
- **Development Standards Documentation**: Comprehensive development standards and guidelines documentation
  - Created docs/_source/runbook/development_standards.rst with complete development standards
  - Integrated directory layout standards with existing SDLC procedures documentation
  - Documented code organization standards, import organization, and module structure guidelines
  - Provided configuration management standards and file migration procedures
  - Included automation and tooling documentation for structure validation and maintenance
  - Updated main documentation index to include development standards in operations runbook
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
- **Documentation Dependencies**: Added myst-parser>=2.0.0 for Markdown support in Sphinx documentation
- **Project Structure**: Complete reorganization following directory layout standards
  - Moved development tools from root directory to categorized `tools/` subdirectories
  - Reorganized 21 test files by type (unit, property-based, integration) for better test management
  - Relocated build and maintenance scripts to dedicated `scripts/` directory
  - Enhanced development workflow with standardized tool locations and clear categorization
  - Improved discoverability and maintainability through consistent directory structure
  - Updated all file permissions for executable scripts and development tools
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
