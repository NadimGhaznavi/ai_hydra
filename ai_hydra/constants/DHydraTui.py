# ai_hydra/constants/DHydraTui.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Final


class DField:
    """
    Machine readable field names.
    """

    BUTTONS: Final[str] = "buttons"
    CLIENTS: Final[str] = "clients"
    CLIENTS_SCREEN: Final[str] = "clients_screen"
    CONNECTED: Final[str] = "connected"
    CONFIG: Final[str] = "config"
    CONSOLE: Final[str] = "console"
    CONSOLE_SCREEN: Final[str] = "console_screen"
    QUIT: Final[str] = "quit"
    RUNNING: Final[str] = "running"
    STATUS: Final[str] = "status"
    TITLE: Final[str] = "title"


class DFile:
    """
    Filenames.
    """

    CLIENT_CSS_PATH: Final[str] = "HydraClient.tcss"
    ROUTER_CSS_PATH: Final[str] = "HydraRouter.tcss"


class DLabel:
    """
    Human readable text.
    """

    CLIENT_TITLE: Final[str] = "Hydra Client"
    CLIENTS: Final[str] = "Clients"
    CONFIG: Final[str] = "Configuration"
    CONNECTED: Final[str] = "Connected"
    DISCONNECTED: Final[str] = "Disonnected"
    ERROR: Final[str] = "ERROR"
    LISTEN_PORT: Final[str] = "Listening Port"
    PING_ROUTER: Final[str] = "Ping Router"
    PING_SERVER: Final[str] = "Ping Server"
    QUIT: Final[str] = "Quit"
    ROUTER_TITLE: Final[str] = "Hydra Router"
    SPACE: Final[str] = " "
    START: Final[str] = "Start"
    STATUS: Final[str] = "Status"
    TARGET_HOST: Final[str] = "Target Host"
    TARGET_PORT: Final[str] = "Target Port"
    VERSION: Final[str] = "Version"


class DStatus:
    """
    Status indicators.
    """

    GOOD: Final[str] = "🟢"
    OK: Final[str] = "🟡"
    BAD: Final[str] = "🔴"
