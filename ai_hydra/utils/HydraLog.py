# ai_hydra/utils/HydraLog.py
#
#   Hydra Router
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

import logging
from typing import Any, Dict, Optional

from ai_hydra.constants.DHydra import LOG_LEVELS, DHydraLog, DHydraLogDef


class HydraLog:
    """
    Centralized logging utility for HydraRouter components.

    HydraLog provides a standardized logging interface with configurable
    output destinations (console and/or file) and log levels. It wraps
    Python's standard logging module with HydraRouter-specific formatting
    and configuration.
    """

    def __init__(
        self,
        client_id: str,
        log_file: Optional[str] = None,
        to_console: bool = True,
        log_level: DHydraLog = DHydraLogDef.DEFAULT_LOG_LEVEL,
    ) -> None:
        """
        Initialize the HydraLog instance with specified configuration.

        Args:
            client_id (str): Unique identifier for the logging client
            log_file (Optional[str]): Path to log file for file output
            to_console (bool): Whether to output logs to console

        Returns:
            None
        """

        # Get a logging object
        self._logger = logging.getLogger(client_id)

        # The default logger log level
        self._logger.setLevel(LOG_LEVELS[log_level])

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Optional file handler
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(LOG_LEVELS[log_level])
            fh.setFormatter(formatter)
            self._logger.addHandler(fh)

        # Optional console handler
        if to_console:
            ch = logging.StreamHandler()
            ch.setLevel(LOG_LEVELS[log_level])
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)

        self._logger.propagate = False

    def loglevel(self, loglevel: DHydraLog) -> None:
        """
        Set the logging level for this logger instance.

        Args:
            loglevel (DHydraLog): Log level string from DHydraLog constants

        Returns:
            None

        Raises:
            KeyError: If loglevel is not a valid log level constant
        """
        self._logger.setLevel(LOG_LEVELS[loglevel])

    def shutdown(self) -> None:
        """
        Cleanly shutdown the logging system and flush all handlers.

        Returns:
            None
        """
        # Exit cleanly
        logging.shutdown()  # Flush all handler

    # Basic log message handling, wraps Python's logging object
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an informational message.

        Args:
            message (str): The message to log
            extra (Optional[Dict[str, Any]]): Extra context data for logging

        Returns:
            None
        """
        self._logger.info(message, extra=extra)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a debug message.

        Args:
            message (str): The message to log
            extra (Optional[Dict[str, Any]]): Extra context data for logging

        Returns:
            None
        """
        self._logger.debug(message, extra=extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a warning message.

        Args:
            message (str): The message to log
            extra (Optional[Dict[str, Any]]): Extra context data for logging

        Returns:
            None
        """
        self._logger.warning(message, extra=extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error message.

        Args:
            message (str): The message to log
            extra (Optional[Dict[str, Any]]): Extra context data for logging

        Returns:
            None
        """
        self._logger.error(message, extra=extra)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a critical error message.

        Args:
            message (str): The message to log
            extra (Optional[Dict[str, Any]]): Extra context data for logging

        Returns:
            None
        """
        self._logger.critical(message, extra=extra)
