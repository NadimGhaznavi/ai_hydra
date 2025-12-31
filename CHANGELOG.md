# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Router Architecture**: Complete client/server router system based on ai_snake_lab pattern
  - `HydraRouter` class for routing messages between clients and servers
  - `MQClient` generic class for router communication
  - `RouterConstants` for centralized message types and constants
  - Support for multiple clients connecting to single server through router
  - Heartbeat-based client registration and management
  - Automatic inactive client detection and removal
  - `ai-hydra-router` CLI entry point for standalone router operation
- **TUI Epoch Display**: Added epoch counter to TUI status widget
  - Shows "Epoch: N" between Snake Length and Runtime in status display
  - Reactive updates when epoch changes
  - Comprehensive test coverage including unit, property-based, and integration tests
- **Enhanced Error Handling**: Better error messages for missing TUI dependencies
  - Graceful fallback when `textual` package not installed
  - Helpful installation instructions for optional dependencies

### Changed
- **Headless Server**: Updated to use MQClient and connect to router instead of direct binding
- **TUI Client**: Updated to use MQClient for router communication instead of direct server connection
- **Network Architecture**: Transformed from direct client-server to router-based messaging system
  - Clients now connect to router at port 5556 instead of directly to server
  - Router handles message routing between clients and servers
  - Supports distributed deployment across multiple machines

### Fixed
- **TUI Dependencies**: Fixed ModuleNotFoundError when textual package not installed
- **Message Protocol**: Standardized message format across all components
- **Connection Management**: Improved connection handling and cleanup

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