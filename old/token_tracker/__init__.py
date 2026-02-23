"""
Token Tracker Package

This package provides comprehensive token usage tracking for the Kiro IDE,
including data models, CSV operations, and agent hook integration.
"""

from .models import TokenTransaction, TrackerConfig
from .tracker import TokenTracker
from .csv_writer import CSVWriter
from .metadata_collector import MetadataCollector
from .hook import TokenTrackingHook
from .error_handler import (
    TokenTrackerError,
    CSVWriteError,
    ValidationError,
    HookExecutionError,
)

__all__ = [
    "TokenTransaction",
    "TrackerConfig",
    "TokenTracker",
    "CSVWriter",
    "MetadataCollector",
    "TokenTrackingHook",
    "TokenTrackerError",
    "CSVWriteError",
    "ValidationError",
    "HookExecutionError",
]

__version__ = "1.0.0"
