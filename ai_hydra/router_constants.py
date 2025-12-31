"""
AI Hydra Router Constants

Constants for the AI Hydra router system, based on ai_snake_lab DMQ pattern.
"""


class RouterConstants:
    """Router message constants."""
    
    # Client/Server Types
    HYDRA_CLIENT = "HydraClient"
    HYDRA_SERVER = "HydraServer"
    HYDRA_ROUTER = "HydraRouter"
    
    # Message Structure Keys
    SENDER = "sender"
    ELEM = "elem"
    DATA = "data"
    CLIENT_ID = "client_id"
    TIMESTAMP = "timestamp"
    REQUEST_ID = "request_id"
    MESSAGE_TYPE = "message_type"
    
    # System Messages
    HEARTBEAT = "heartbeat"
    STATUS = "status"
    ERROR = "error"
    OK = "ok"
    
    # Simulation Control Commands
    START_SIMULATION = "start_simulation"
    STOP_SIMULATION = "stop_simulation"
    PAUSE_SIMULATION = "pause_simulation"
    RESUME_SIMULATION = "resume_simulation"
    RESET_SIMULATION = "reset_simulation"
    
    # Simulation Status Messages
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_STOPPED = "simulation_stopped"
    SIMULATION_PAUSED = "simulation_paused"
    SIMULATION_RESUMED = "simulation_resumed"
    SIMULATION_RESET = "simulation_reset"
    
    # Data Messages
    GET_STATUS = "get_status"
    STATUS_UPDATE = "status_update"
    GAME_STATE_UPDATE = "game_state_update"
    PERFORMANCE_UPDATE = "performance_update"
    
    # Configuration
    GET_CONFIG = "get_config"
    SET_CONFIG = "set_config"
    CONFIG_UPDATE = "config_update"
    
    # Timing
    HEARTBEAT_INTERVAL = 5  # seconds
    
    # Network
    DEFAULT_ROUTER_PORT = 5556
    DEFAULT_SERVER_PORT = 5555
    PROTOCOL = "tcp"


class RouterLabels:
    """Human-readable labels for router messages."""
    
    STARTUP_MSG = "Router running on %s"
    SHUTDOWN_MSG = "Router shutting down..."
    MALFORMED_MESSAGE = "Malformed message"
    ROUTER_ERROR = "Router error: %s"
    UNKNOWN_SENDER = "Unknown sender type"
    NO_SERVER_CONNECTED = "No AI Hydra server connected"
    CLIENT_DISCONNECTED = "Client disconnected"
    SERVER_DISCONNECTED = "Server disconnected"