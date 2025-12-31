"""
Core data models for the Token Tracker system.

This module defines the fundamental data structures used throughout the token
tracking system, including transaction records and configuration objects.
All models are designed to be immutable and thread-safe.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import uuid


@dataclass(frozen=True)
class TokenTransaction:
    """
    Immutable representation of a single token transaction record.

    This class encapsulates all information about a token usage event,
    including timing, context, and metadata for comprehensive tracking.

    Attributes:
        timestamp: When the transaction occurred (ISO format)
        prompt_text: The prompt text sent to the AI model
        tokens_used: Number of tokens consumed
        elapsed_time: Time taken for the operation in seconds
        session_id: Unique identifier for the current session
        workspace_folder: Name of the workspace folder
        hook_trigger_type: Type of hook that triggered this transaction
        agent_execution_id: Unique identifier for the agent execution
        file_patterns: List of file patterns that triggered the interaction
        hook_name: Name of the specific hook that initiated tracking
        error_occurred: Whether an error occurred during the transaction
        error_message: Error message if an error occurred
    """

    timestamp: datetime
    prompt_text: str
    tokens_used: int
    elapsed_time: float
    session_id: str
    workspace_folder: str
    hook_trigger_type: str
    agent_execution_id: str
    file_patterns: Optional[List[str]] = None
    hook_name: str = "unknown"
    error_occurred: bool = False
    error_message: Optional[str] = None

    def __post_init__(self):
        """Validate transaction data after initialization."""
        if self.tokens_used < 0:
            raise ValueError("tokens_used must be non-negative")

        if self.elapsed_time < 0:
            raise ValueError("elapsed_time must be non-negative")

        if not self.prompt_text.strip():
            raise ValueError("prompt_text cannot be empty")

        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty")

        if not self.workspace_folder.strip():
            raise ValueError("workspace_folder cannot be empty")

        if not self.hook_trigger_type.strip():
            raise ValueError("hook_trigger_type cannot be empty")

        if not self.agent_execution_id.strip():
            raise ValueError("agent_execution_id cannot be empty")

    @classmethod
    def create_new(
        cls,
        prompt_text: str,
        tokens_used: int,
        elapsed_time: float,
        workspace_folder: str,
        hook_trigger_type: str,
        session_id: Optional[str] = None,
        agent_execution_id: Optional[str] = None,
        file_patterns: Optional[List[str]] = None,
        hook_name: str = "unknown",
        error_occurred: bool = False,
        error_message: Optional[str] = None,
    ) -> "TokenTransaction":
        """
        Create a new token transaction with auto-generated IDs.

        Args:
            prompt_text: The prompt text sent to the AI model
            tokens_used: Number of tokens consumed
            elapsed_time: Time taken for the operation in seconds
            workspace_folder: Name of the workspace folder
            hook_trigger_type: Type of hook that triggered this transaction
            session_id: Session ID (auto-generated if not provided)
            agent_execution_id: Agent execution ID (auto-generated if not provided)
            file_patterns: List of file patterns that triggered the interaction
            hook_name: Name of the specific hook
            error_occurred: Whether an error occurred
            error_message: Error message if an error occurred

        Returns:
            TokenTransaction: New transaction instance
        """
        return cls(
            timestamp=datetime.now(),
            prompt_text=prompt_text,
            tokens_used=tokens_used,
            elapsed_time=elapsed_time,
            session_id=session_id or str(uuid.uuid4()),
            workspace_folder=workspace_folder,
            hook_trigger_type=hook_trigger_type,
            agent_execution_id=agent_execution_id or str(uuid.uuid4()),
            file_patterns=file_patterns,
            hook_name=hook_name,
            error_occurred=error_occurred,
            error_message=error_message,
        )

    def to_csv_row(self) -> List[str]:
        """
        Convert transaction to CSV row format.

        Returns:
            List[str]: CSV row data with proper escaping
        """
        # Handle file patterns - convert list to semicolon-separated string
        file_patterns_str = ""
        if self.file_patterns:
            file_patterns_str = ";".join(self.file_patterns)

        # Escape and sanitize prompt text for CSV
        sanitized_prompt = self._sanitize_csv_text(self.prompt_text)

        # Format timestamp as ISO string
        timestamp_str = self.timestamp.isoformat()

        return [
            timestamp_str,
            sanitized_prompt,
            str(self.tokens_used),
            str(self.elapsed_time),
            self.session_id,
            self.workspace_folder,
            self.hook_trigger_type,
            self.agent_execution_id,
            file_patterns_str,
            self.hook_name,
            str(self.error_occurred).lower(),
            self.error_message or "",
        ]

    @classmethod
    def from_csv_row(cls, row: List[str]) -> "TokenTransaction":
        """
        Create transaction from CSV row data.

        Args:
            row: CSV row data

        Returns:
            TokenTransaction: Transaction instance

        Raises:
            ValueError: If row data is invalid
        """
        if len(row) != 12:
            raise ValueError(f"Expected 12 columns, got {len(row)}")

        # Parse timestamp
        timestamp = datetime.fromisoformat(row[0])

        # Parse file patterns
        file_patterns = None
        if row[8].strip():
            file_patterns = row[8].split(";")

        # Parse boolean values
        error_occurred = row[10].lower() == "true"

        return cls(
            timestamp=timestamp,
            prompt_text=row[1],
            tokens_used=int(row[2]),
            elapsed_time=float(row[3]),
            session_id=row[4],
            workspace_folder=row[5],
            hook_trigger_type=row[6],
            agent_execution_id=row[7],
            file_patterns=file_patterns,
            hook_name=row[9],
            error_occurred=error_occurred,
            error_message=row[11] if row[11] else None,
        )

    def _sanitize_csv_text(self, text: str) -> str:
        """
        Sanitize text for CSV storage.

        Args:
            text: Text to sanitize

        Returns:
            str: Sanitized text safe for CSV storage
        """
        # Replace newlines with escaped sequences
        sanitized = text.replace("\n", "\\n").replace("\r", "\\r")

        # Escape quotes
        sanitized = sanitized.replace('"', '""')

        # Limit length to prevent extremely large CSV cells
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "...[truncated]"

        return sanitized

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the transaction for logging/display.

        Returns:
            Dict[str, Any]: Transaction summary
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "tokens_used": self.tokens_used,
            "elapsed_time": self.elapsed_time,
            "workspace": self.workspace_folder,
            "hook_type": self.hook_trigger_type,
            "hook_name": self.hook_name,
            "prompt_length": len(self.prompt_text),
            "has_error": self.error_occurred,
            "file_patterns_count": len(self.file_patterns) if self.file_patterns else 0,
        }


@dataclass
class TrackerConfig:
    """
    Configuration for the token tracking system.

    This class encapsulates all configuration options for the token tracker,
    including file paths, limits, and behavior settings.
    """

    enabled: bool = True
    csv_file_path: Union[str, Path] = ".kiro/token_transactions.csv"
    max_prompt_length: int = 1000
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    compression_enabled: bool = False
    retention_days: int = 365
    auto_create_directories: bool = True
    file_lock_timeout_seconds: float = 5.0
    max_concurrent_writes: int = 10
    enable_validation: bool = True
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_prompt_length < 10:
            raise ValueError("max_prompt_length must be at least 10")

        if self.backup_interval_hours < 1:
            raise ValueError("backup_interval_hours must be at least 1")

        if self.retention_days < 1:
            raise ValueError("retention_days must be at least 1")

        if self.file_lock_timeout_seconds <= 0:
            raise ValueError("file_lock_timeout_seconds must be positive")

        if self.max_concurrent_writes < 1:
            raise ValueError("max_concurrent_writes must be at least 1")

        # Convert string path to Path object
        if isinstance(self.csv_file_path, str):
            object.__setattr__(self, "csv_file_path", Path(self.csv_file_path))

        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ValueError(f"log_level must be one of: {valid_levels}")

    @classmethod
    def create_default(cls) -> "TrackerConfig":
        """
        Create default configuration.

        Returns:
            TrackerConfig: Default configuration instance
        """
        return cls()

    @classmethod
    def create_for_testing(cls) -> "TrackerConfig":
        """
        Create configuration optimized for testing.

        Returns:
            TrackerConfig: Testing configuration instance
        """
        return cls(
            csv_file_path=Path("test_token_transactions.csv"),
            backup_enabled=False,
            retention_days=1,
            file_lock_timeout_seconds=1.0,
            max_concurrent_writes=5,
            log_level="DEBUG",
        )

    @classmethod
    def create_for_production(cls) -> "TrackerConfig":
        """
        Create configuration optimized for production use.

        Returns:
            TrackerConfig: Production configuration instance
        """
        return cls(
            csv_file_path=Path(".kiro/token_transactions.csv"),
            backup_enabled=True,
            backup_interval_hours=12,
            compression_enabled=True,
            retention_days=90,
            file_lock_timeout_seconds=10.0,
            max_concurrent_writes=20,
            log_level="INFO",
        )

    def get_csv_file_path(self) -> Path:
        """
        Get the CSV file path as a Path object.

        Returns:
            Path: CSV file path
        """
        return self.csv_file_path

    def get_backup_file_path(self) -> Path:
        """
        Get the backup file path.

        Returns:
            Path: Backup file path
        """
        csv_path = self.get_csv_file_path()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{csv_path.stem}_backup_{timestamp}{csv_path.suffix}"
        return csv_path.parent / backup_name

    def should_create_backup(self, last_backup_time: Optional[datetime] = None) -> bool:
        """
        Check if a backup should be created based on interval.

        Args:
            last_backup_time: Time of last backup (None if never backed up)

        Returns:
            bool: True if backup should be created
        """
        if not self.backup_enabled:
            return False

        if last_backup_time is None:
            return True

        time_since_backup = datetime.now() - last_backup_time
        backup_interval = timedelta(hours=self.backup_interval_hours)

        return time_since_backup >= backup_interval

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List[str]: List of validation issues (empty if valid)
        """
        issues = []

        # Check CSV file path
        csv_path = self.get_csv_file_path()
        if not csv_path.parent.exists() and not self.auto_create_directories:
            issues.append(f"CSV directory does not exist: {csv_path.parent}")

        # Check file permissions if file exists
        if csv_path.exists():
            try:
                # Test write access
                with open(csv_path, "a") as f:
                    pass
            except PermissionError:
                issues.append(f"No write permission for CSV file: {csv_path}")

        # Validate numeric ranges
        if self.max_prompt_length > 10000:
            issues.append(
                "max_prompt_length is very large, may cause performance issues"
            )

        if self.retention_days > 3650:  # 10 years
            issues.append("retention_days is very large, may cause storage issues")

        if self.file_lock_timeout_seconds > 60:
            issues.append("file_lock_timeout_seconds is very large, may cause delays")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return {
            "enabled": self.enabled,
            "csv_file_path": str(self.csv_file_path),
            "max_prompt_length": self.max_prompt_length,
            "backup_enabled": self.backup_enabled,
            "backup_interval_hours": self.backup_interval_hours,
            "compression_enabled": self.compression_enabled,
            "retention_days": self.retention_days,
            "auto_create_directories": self.auto_create_directories,
            "file_lock_timeout_seconds": self.file_lock_timeout_seconds,
            "max_concurrent_writes": self.max_concurrent_writes,
            "enable_validation": self.enable_validation,
            "log_level": self.log_level,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrackerConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            TrackerConfig: Configuration instance
        """
        # Convert csv_file_path back to Path if it's a string
        if "csv_file_path" in config_dict:
            config_dict["csv_file_path"] = Path(config_dict["csv_file_path"])

        return cls(**config_dict)


@dataclass(frozen=True)
class CSVSchema:
    """
    Schema definition for the token transactions CSV file.

    This class defines the structure and validation rules for the CSV file
    used to store token transaction data.
    """

    headers: List[str] = field(
        default_factory=lambda: [
            "timestamp",
            "prompt_text",
            "tokens_used",
            "elapsed_time",
            "session_id",
            "workspace_folder",
            "hook_trigger_type",
            "agent_execution_id",
            "file_patterns",
            "hook_name",
            "error_occurred",
            "error_message",
        ]
    )

    required_columns: List[str] = field(
        default_factory=lambda: [
            "timestamp",
            "prompt_text",
            "tokens_used",
            "elapsed_time",
            "session_id",
            "workspace_folder",
            "hook_trigger_type",
            "agent_execution_id",
        ]
    )

    column_types: Dict[str, type] = field(
        default_factory=lambda: {
            "timestamp": str,
            "prompt_text": str,
            "tokens_used": int,
            "elapsed_time": float,
            "session_id": str,
            "workspace_folder": str,
            "hook_trigger_type": str,
            "agent_execution_id": str,
            "file_patterns": str,
            "hook_name": str,
            "error_occurred": bool,
            "error_message": str,
        }
    )

    def validate_headers(self, headers: List[str]) -> List[str]:
        """
        Validate CSV headers against schema.

        Args:
            headers: Headers from CSV file

        Returns:
            List[str]: List of validation issues
        """
        issues = []

        # Check for missing required headers
        for required_header in self.headers:
            if required_header not in headers:
                issues.append(f"Missing required header: {required_header}")

        # Check for extra headers
        for header in headers:
            if header not in self.headers:
                issues.append(f"Unknown header: {header}")

        # Check header order
        if headers != self.headers:
            issues.append("Headers are not in expected order")

        return issues

    def validate_row(self, row: List[str]) -> List[str]:
        """
        Validate a CSV row against schema.

        Args:
            row: CSV row data

        Returns:
            List[str]: List of validation issues
        """
        issues = []

        # Check column count
        if len(row) != len(self.headers):
            issues.append(f"Expected {len(self.headers)} columns, got {len(row)}")
            return issues  # Can't validate further without correct column count

        # Validate each column
        for i, (header, value) in enumerate(zip(self.headers, row)):
            column_type = self.column_types.get(header, str)

            # Check required columns are not empty
            if header in self.required_columns and not value.strip():
                issues.append(f"Required column '{header}' is empty")
                continue

            # Type validation
            try:
                if column_type == int:
                    int(value)
                elif column_type == float:
                    float(value)
                elif column_type == bool:
                    if value.lower() not in ["true", "false"]:
                        issues.append(
                            f"Column '{header}' must be 'true' or 'false', got '{value}'"
                        )
            except ValueError:
                issues.append(
                    f"Column '{header}' has invalid type, expected {column_type.__name__}"
                )

        return issues

    def get_csv_header_row(self) -> str:
        """
        Get CSV header row as string.

        Returns:
            str: CSV header row
        """
        return ",".join(self.headers)
