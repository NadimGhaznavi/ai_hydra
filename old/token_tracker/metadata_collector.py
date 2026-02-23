"""
Metadata collection system for token tracking.

This module provides comprehensive metadata collection from the Kiro IDE
environment, including workspace information, hook context, and execution details.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
import platform
import psutil

from .error_handler import TokenTrackerErrorHandler, MetadataError


class MetadataCollector:
    """
    Collects contextual metadata from the Kiro IDE environment.

    This class gathers information about the current execution context,
    workspace state, and system environment for comprehensive token tracking.
    """

    def __init__(
        self,
        error_handler: Optional[TokenTrackerErrorHandler] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the metadata collector.

        Args:
            error_handler: Error handler instance
            logger: Logger instance
        """
        self.error_handler = error_handler or TokenTrackerErrorHandler()
        self.logger = logger or logging.getLogger(__name__)

        # Cache for expensive operations
        self._workspace_cache: Dict[str, Any] = {}
        self._system_cache: Dict[str, Any] = {}
        self._cache_timestamp = datetime.now()
        self._cache_ttl_seconds = 300  # 5 minutes

        self.logger.info("MetadataCollector initialized")

    def collect_execution_metadata(
        self, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Collect metadata about the current execution context.

        Args:
            context: Additional context information from the caller

        Returns:
            Dict[str, Any]: Execution metadata
        """
        metadata = {}

        try:
            # Basic execution information
            metadata.update(
                {
                    "timestamp": datetime.now().isoformat(),
                    "session_id": self._get_or_create_session_id(),
                    "agent_execution_id": str(uuid.uuid4()),
                    "process_id": os.getpid(),
                    "thread_id": self._get_thread_id(),
                }
            )

            # Workspace information
            workspace_info = self.get_workspace_info()
            metadata.update(workspace_info)

            # Hook context
            hook_context = self.get_hook_context(context)
            metadata.update(hook_context)

            # System information
            system_info = self.get_system_info()
            metadata.update(system_info)

            # Performance metrics
            performance_info = self.get_performance_info()
            metadata.update(performance_info)

            self.logger.debug(f"Collected execution metadata: {len(metadata)} fields")

        except Exception as e:
            recovery_result = self.error_handler.handle_metadata_error("execution", e)
            if recovery_result.success and recovery_result.recovered_data:
                metadata.update(recovery_result.recovered_data)
            else:
                # Provide minimal fallback metadata
                metadata.update(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "session_id": "unknown",
                        "agent_execution_id": str(uuid.uuid4()),
                        "workspace_folder": "unknown",
                        "hook_trigger_type": "unknown",
                        "hook_name": "unknown",
                    }
                )
                self.logger.error(f"Failed to collect execution metadata: {e}")

        return metadata

    def get_workspace_info(self) -> Dict[str, Any]:
        """
        Get current workspace information.

        Returns:
            Dict[str, Any]: Workspace information
        """
        if self._is_cache_valid() and "workspace" in self._workspace_cache:
            return self._workspace_cache["workspace"]

        workspace_info = {}

        try:
            # Current working directory
            current_dir = Path.cwd()
            workspace_info["current_directory"] = str(current_dir)

            # Workspace folder name
            workspace_info["workspace_folder"] = current_dir.name

            # Look for workspace indicators
            workspace_info.update(self._detect_workspace_type(current_dir))

            # Git information
            git_info = self._get_git_info(current_dir)
            if git_info:
                workspace_info.update(git_info)

            # Project files
            project_info = self._get_project_info(current_dir)
            workspace_info.update(project_info)

            # Cache the result
            self._workspace_cache["workspace"] = workspace_info

        except Exception as e:
            recovery_result = self.error_handler.handle_metadata_error("workspace", e)
            if recovery_result.success and recovery_result.recovered_data:
                workspace_info = recovery_result.recovered_data
            else:
                workspace_info = {
                    "workspace_folder": "unknown",
                    "current_directory": str(Path.cwd()),
                    "workspace_type": "unknown",
                }
                self.logger.error(f"Failed to collect workspace info: {e}")

        return workspace_info

    def get_hook_context(
        self, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get information about the triggering hook context.

        Args:
            context: Hook context information from the caller

        Returns:
            Dict[str, Any]: Hook context information
        """
        hook_info = {}

        try:
            if context:
                # Extract hook information from provided context
                hook_info.update(
                    {
                        "hook_trigger_type": context.get("trigger_type", "unknown"),
                        "hook_name": context.get("hook_name", "unknown"),
                        "file_patterns": context.get("file_patterns"),
                        "trigger_event": context.get("event", "unknown"),
                        "hook_config": context.get("config", {}),
                    }
                )
            else:
                # Try to infer hook context from environment
                hook_info.update(self._infer_hook_context())

            # Add file context if available
            file_context = self._get_file_context(context)
            if file_context:
                hook_info.update(file_context)

        except Exception as e:
            recovery_result = self.error_handler.handle_metadata_error(
                "hook_context", e
            )
            if recovery_result.success and recovery_result.recovered_data:
                hook_info = recovery_result.recovered_data
            else:
                hook_info = {
                    "hook_trigger_type": "unknown",
                    "hook_name": "unknown",
                    "file_patterns": None,
                }
                self.logger.error(f"Failed to collect hook context: {e}")

        return hook_info

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.

        Returns:
            Dict[str, Any]: System information
        """
        if self._is_cache_valid() and "system" in self._system_cache:
            return self._system_cache["system"]

        system_info = {}

        try:
            system_info.update(
                {
                    "platform": platform.system(),
                    "platform_version": platform.version(),
                    "architecture": platform.architecture()[0],
                    "python_version": platform.python_version(),
                    "hostname": platform.node(),
                    "username": os.getenv("USER") or os.getenv("USERNAME", "unknown"),
                }
            )

            # Cache the result
            self._system_cache["system"] = system_info

        except Exception as e:
            recovery_result = self.error_handler.handle_metadata_error("system", e)
            if recovery_result.success and recovery_result.recovered_data:
                system_info = recovery_result.recovered_data
            else:
                system_info = {"platform": "unknown", "username": "unknown"}
                self.logger.error(f"Failed to collect system info: {e}")

        return system_info

    def get_performance_info(self) -> Dict[str, Any]:
        """
        Get performance and resource usage information.

        Returns:
            Dict[str, Any]: Performance information
        """
        performance_info = {}

        try:
            # Process information
            process = psutil.Process()
            performance_info.update(
                {
                    "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                    "cpu_percent": process.cpu_percent(),
                    "num_threads": process.num_threads(),
                    "process_create_time": process.create_time(),
                }
            )

            # System information
            performance_info.update(
                {
                    "system_cpu_percent": psutil.cpu_percent(interval=0.1),
                    "system_memory_percent": psutil.virtual_memory().percent,
                    "system_disk_usage_percent": (
                        psutil.disk_usage("/").percent
                        if os.name != "nt"
                        else psutil.disk_usage("C:").percent
                    ),
                }
            )

        except Exception as e:
            recovery_result = self.error_handler.handle_metadata_error("performance", e)
            if recovery_result.success and recovery_result.recovered_data:
                performance_info = recovery_result.recovered_data
            else:
                performance_info = {"memory_usage_mb": 0, "cpu_percent": 0}
                self.logger.error(f"Failed to collect performance info: {e}")

        return performance_info

    def collect_token_usage_metadata(
        self, prompt_text: str, execution_time: float
    ) -> Dict[str, Any]:
        """
        Collect metadata specific to token usage events.

        Args:
            prompt_text: The prompt text that was processed
            execution_time: Time taken to process the prompt

        Returns:
            Dict[str, Any]: Token usage metadata
        """
        metadata = {}

        try:
            metadata.update(
                {
                    "prompt_length": len(prompt_text),
                    "prompt_word_count": len(prompt_text.split()),
                    "prompt_line_count": prompt_text.count("\n") + 1,
                    "execution_time": execution_time,
                    "execution_timestamp": datetime.now().isoformat(),
                }
            )

            # Analyze prompt characteristics
            prompt_analysis = self._analyze_prompt(prompt_text)
            metadata.update(prompt_analysis)

        except Exception as e:
            recovery_result = self.error_handler.handle_metadata_error("token_usage", e)
            if recovery_result.success and recovery_result.recovered_data:
                metadata = recovery_result.recovered_data
            else:
                metadata = {
                    "prompt_length": len(prompt_text) if prompt_text else 0,
                    "execution_time": execution_time,
                }
                self.logger.error(f"Failed to collect token usage metadata: {e}")

        return metadata

    def _get_or_create_session_id(self) -> str:
        """Get or create a session ID for the current session."""
        # Try to get session ID from environment or create new one
        session_id = os.getenv("KIRO_SESSION_ID")
        if not session_id:
            session_id = str(uuid.uuid4())
            os.environ["KIRO_SESSION_ID"] = session_id
        return session_id

    def _get_thread_id(self) -> int:
        """Get current thread ID."""
        import threading

        return threading.get_ident()

    def _detect_workspace_type(self, workspace_dir: Path) -> Dict[str, Any]:
        """Detect the type of workspace based on files present."""
        workspace_type = "unknown"
        project_files = []

        # Check for common project files
        project_indicators = {
            "python": ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile"],
            "node": ["package.json", "yarn.lock", "package-lock.json"],
            "rust": ["Cargo.toml", "Cargo.lock"],
            "go": ["go.mod", "go.sum"],
            "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
            "dotnet": ["*.csproj", "*.sln", "*.fsproj"],
            "kiro": [".kiro/"],
        }

        for proj_type, indicators in project_indicators.items():
            for indicator in indicators:
                if indicator.endswith("/"):
                    # Directory indicator
                    if (workspace_dir / indicator.rstrip("/")).is_dir():
                        workspace_type = proj_type
                        project_files.append(indicator)
                        break
                elif "*" in indicator:
                    # Glob pattern
                    matches = list(workspace_dir.glob(indicator))
                    if matches:
                        workspace_type = proj_type
                        project_files.extend([str(m.name) for m in matches])
                        break
                else:
                    # Regular file
                    if (workspace_dir / indicator).exists():
                        workspace_type = proj_type
                        project_files.append(indicator)
                        break

            if workspace_type != "unknown":
                break

        return {"workspace_type": workspace_type, "project_files": project_files}

    def _get_git_info(self, workspace_dir: Path) -> Optional[Dict[str, Any]]:
        """Get Git repository information if available."""
        git_dir = workspace_dir / ".git"
        if not git_dir.exists():
            return None

        git_info = {"is_git_repo": True}

        try:
            # Try to get branch name
            head_file = git_dir / "HEAD"
            if head_file.exists():
                head_content = head_file.read_text().strip()
                if head_content.startswith("ref: refs/heads/"):
                    git_info["branch"] = head_content.split("/")[-1]
                else:
                    git_info["branch"] = head_content[:8]  # Short commit hash

            # Try to get remote URL
            config_file = git_dir / "config"
            if config_file.exists():
                config_content = config_file.read_text()
                # Simple parsing for remote URL
                for line in config_content.split("\n"):
                    if "url =" in line:
                        git_info["remote_url"] = line.split("url =")[-1].strip()
                        break

        except Exception as e:
            self.logger.debug(f"Could not read Git info: {e}")

        return git_info

    def _get_project_info(self, workspace_dir: Path) -> Dict[str, Any]:
        """Get project-specific information."""
        project_info = {}

        # Check for common project files and extract info
        if (workspace_dir / "pyproject.toml").exists():
            project_info.update(
                self._parse_pyproject_toml(workspace_dir / "pyproject.toml")
            )
        elif (workspace_dir / "package.json").exists():
            project_info.update(
                self._parse_package_json(workspace_dir / "package.json")
            )

        return project_info

    def _parse_pyproject_toml(self, toml_file: Path) -> Dict[str, Any]:
        """Parse pyproject.toml for project information."""
        try:
            import tomllib

            with open(toml_file, "rb") as f:
                data = tomllib.load(f)

            project_data = data.get("project", {})
            return {
                "project_name": project_data.get("name", "unknown"),
                "project_version": project_data.get("version", "unknown"),
                "project_description": project_data.get("description", ""),
            }
        except Exception as e:
            self.logger.debug(f"Could not parse pyproject.toml: {e}")
            return {}

    def _parse_package_json(self, json_file: Path) -> Dict[str, Any]:
        """Parse package.json for project information."""
        try:
            import json

            with open(json_file, "r") as f:
                data = json.load(f)

            return {
                "project_name": data.get("name", "unknown"),
                "project_version": data.get("version", "unknown"),
                "project_description": data.get("description", ""),
            }
        except Exception as e:
            self.logger.debug(f"Could not parse package.json: {e}")
            return {}

    def _infer_hook_context(self) -> Dict[str, Any]:
        """Infer hook context from environment variables and process info."""
        hook_info = {
            "hook_trigger_type": "unknown",
            "hook_name": "unknown",
            "file_patterns": None,
        }

        # Check environment variables that might indicate hook context
        kiro_hook_type = os.getenv("KIRO_HOOK_TYPE")
        if kiro_hook_type:
            hook_info["hook_trigger_type"] = kiro_hook_type

        kiro_hook_name = os.getenv("KIRO_HOOK_NAME")
        if kiro_hook_name:
            hook_info["hook_name"] = kiro_hook_name

        kiro_file_patterns = os.getenv("KIRO_FILE_PATTERNS")
        if kiro_file_patterns:
            hook_info["file_patterns"] = kiro_file_patterns.split(";")

        return hook_info

    def _get_file_context(
        self, context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Get file context information from the hook context."""
        if not context:
            return None

        file_context = {}

        # Extract file-related information
        if "modified_files" in context:
            file_context["modified_files"] = context["modified_files"]

        if "current_file" in context:
            file_context["current_file"] = context["current_file"]

        if "file_extension" in context:
            file_context["file_extension"] = context["file_extension"]

        return file_context if file_context else None

    def _analyze_prompt(self, prompt_text: str) -> Dict[str, Any]:
        """Analyze prompt characteristics for metadata."""
        analysis = {}

        try:
            # Basic text analysis
            analysis.update(
                {
                    "has_code_blocks": "```" in prompt_text,
                    "has_file_paths": "/" in prompt_text or "\\" in prompt_text,
                    "has_urls": "http" in prompt_text.lower(),
                    "has_special_chars": any(
                        c in prompt_text for c in ["@", "#", "$", "%"]
                    ),
                    "language_indicators": self._detect_language_indicators(
                        prompt_text
                    ),
                }
            )

        except Exception as e:
            self.logger.debug(f"Prompt analysis failed: {e}")

        return analysis

    def _detect_language_indicators(self, text: str) -> List[str]:
        """Detect programming language indicators in text."""
        indicators = []

        language_keywords = {
            "python": ["def ", "import ", "class ", "if __name__"],
            "javascript": ["function ", "const ", "let ", "var "],
            "typescript": ["interface ", "type ", "export "],
            "java": ["public class", "private ", "public static"],
            "rust": ["fn ", "let mut", "impl "],
            "go": ["func ", "package ", "import "],
            "sql": ["SELECT ", "FROM ", "WHERE ", "INSERT "],
        }

        text_lower = text.lower()
        for language, keywords in language_keywords.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                indicators.append(language)

        return indicators

    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        return (
            datetime.now() - self._cache_timestamp
        ).total_seconds() < self._cache_ttl_seconds

    def clear_cache(self) -> None:
        """Clear all cached metadata."""
        self._workspace_cache.clear()
        self._system_cache.clear()
        self._cache_timestamp = datetime.now()
        self.logger.debug("Metadata cache cleared")
