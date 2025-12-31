"""
Maintenance utilities for the Token Tracker system.

This module provides file rotation, data cleanup, and maintenance
utilities for managing CSV files and system health.
"""

import logging
import os
import shutil
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import csv
import threading
import time

from .models import TokenTransaction, TrackerConfig, CSVSchema
from .error_handler import TokenTrackerErrorHandler


class MaintenanceManager:
    """
    Manages file rotation, cleanup, and maintenance operations for token tracking.

    This class provides utilities for:
    - CSV file rotation when files become too large
    - Data compression for archived files
    - Cleanup of old data based on retention policies
    - Maintenance utilities for data integrity
    """

    def __init__(
        self,
        config: TrackerConfig,
        error_handler: Optional[TokenTrackerErrorHandler] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the maintenance manager.

        Args:
            config: Tracker configuration
            error_handler: Error handler instance
            logger: Logger instance
        """
        self.config = config
        self.error_handler = error_handler or TokenTrackerErrorHandler()
        self.logger = logger or logging.getLogger(__name__)
        self.schema = CSVSchema()

        # Maintenance settings
        self.max_file_size_mb = 50  # Rotate when CSV exceeds 50MB
        self.max_rows_per_file = 100000  # Rotate when CSV exceeds 100k rows
        self.archive_directory = self.config.get_csv_file_path().parent / "archives"

        # Thread safety
        self._maintenance_lock = threading.RLock()

        self.logger.info("MaintenanceManager initialized")

    def check_rotation_needed(self) -> Dict[str, Any]:
        """
        Check if CSV file rotation is needed based on size and row count.

        Returns:
            Dict[str, Any]: Rotation check results
        """
        csv_path = self.config.get_csv_file_path()

        results = {
            "rotation_needed": False,
            "reason": None,
            "file_exists": csv_path.exists(),
            "file_size_mb": 0.0,
            "row_count": 0,
            "max_file_size_mb": self.max_file_size_mb,
            "max_rows_per_file": self.max_rows_per_file,
        }

        if not results["file_exists"]:
            return results

        try:
            # Check file size
            file_size_bytes = csv_path.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            results["file_size_mb"] = file_size_mb

            # Check row count
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                row_count = sum(1 for _ in reader) - 1  # Subtract header row
                results["row_count"] = row_count

            # Determine if rotation is needed
            if file_size_mb > self.max_file_size_mb:
                results["rotation_needed"] = True
                results["reason"] = (
                    f"File size ({file_size_mb:.1f}MB) exceeds limit ({self.max_file_size_mb}MB)"
                )
            elif row_count > self.max_rows_per_file:
                results["rotation_needed"] = True
                results["reason"] = (
                    f"Row count ({row_count}) exceeds limit ({self.max_rows_per_file})"
                )

            self.logger.debug(
                f"Rotation check: size={file_size_mb:.1f}MB, rows={row_count}, "
                f"needed={results['rotation_needed']}"
            )

        except Exception as e:
            self.logger.error(f"Failed to check rotation status: {e}")
            results["error"] = str(e)

        return results

    def rotate_csv_file(self) -> Dict[str, Any]:
        """
        Rotate the current CSV file by moving it to archives and creating a new one.

        Returns:
            Dict[str, Any]: Rotation operation results
        """
        results = {
            "success": False,
            "archived_file": None,
            "new_file_created": False,
            "compression_applied": False,
            "rows_archived": 0,
            "issues": [],
        }

        try:
            with self._maintenance_lock:
                csv_path = self.config.get_csv_file_path()

                if not csv_path.exists():
                    results["issues"].append("CSV file does not exist")
                    return results

                # Create archive directory
                self.archive_directory.mkdir(parents=True, exist_ok=True)

                # Generate archive filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_name = f"{csv_path.stem}_{timestamp}{csv_path.suffix}"
                archive_path = self.archive_directory / archive_name

                # Count rows before archiving
                try:
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        results["rows_archived"] = (
                            sum(1 for _ in reader) - 1
                        )  # Subtract header
                except Exception as e:
                    self.logger.warning(f"Could not count rows: {e}")

                # Move current file to archive
                shutil.move(str(csv_path), str(archive_path))
                results["archived_file"] = str(archive_path)

                # Apply compression if enabled
                if self.config.compression_enabled:
                    compressed_path = archive_path.with_suffix(
                        archive_path.suffix + ".gz"
                    )
                    with open(archive_path, "rb") as f_in:
                        with gzip.open(compressed_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    # Remove uncompressed file
                    archive_path.unlink()
                    results["archived_file"] = str(compressed_path)
                    results["compression_applied"] = True

                # Create new CSV file with headers
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.schema.headers)

                results["new_file_created"] = True
                results["success"] = True

                self.logger.info(
                    f"CSV file rotated: {results['rows_archived']} rows archived to {results['archived_file']}"
                )

        except Exception as e:
            results["issues"].append(f"Rotation failed: {e}")
            self.logger.error(f"CSV rotation failed: {e}")

        return results

    def cleanup_old_data(self, max_age_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Clean up old transaction data and archived files based on retention policy.

        Args:
            max_age_days: Maximum age of data to keep (uses config default if not provided)

        Returns:
            Dict[str, Any]: Cleanup results
        """
        max_age = max_age_days or self.config.retention_days
        cutoff_date = datetime.now() - timedelta(days=max_age)

        results = {
            "success": False,
            "archived_files_removed": 0,
            "space_freed_bytes": 0,
            "cleanup_errors": [],
        }

        try:
            with self._maintenance_lock:
                # Clean up archived files
                if self.archive_directory.exists():
                    for archive_file in self.archive_directory.iterdir():
                        if archive_file.is_file():
                            try:
                                # Check file modification time
                                file_mtime = datetime.fromtimestamp(
                                    archive_file.stat().st_mtime
                                )

                                if file_mtime < cutoff_date:
                                    file_size = archive_file.stat().st_size
                                    archive_file.unlink()

                                    results["archived_files_removed"] += 1
                                    results["space_freed_bytes"] += file_size

                                    self.logger.debug(
                                        f"Removed old archive: {archive_file}"
                                    )

                            except Exception as e:
                                results["cleanup_errors"].append(
                                    f"Failed to remove {archive_file}: {e}"
                                )

                results["success"] = len(results["cleanup_errors"]) == 0

                self.logger.info(
                    f"Cleanup completed: {results['archived_files_removed']} files removed, "
                    f"{results['space_freed_bytes']} bytes freed"
                )

        except Exception as e:
            results["cleanup_errors"].append(f"Cleanup operation failed: {e}")
            self.logger.error(f"Data cleanup failed: {e}")

        return results

    def get_archive_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about archived files.

        Returns:
            Dict[str, Any]: Archive statistics
        """
        stats = {
            "archive_directory_exists": self.archive_directory.exists(),
            "total_archives": 0,
            "total_size_bytes": 0,
            "compressed_archives": 0,
            "oldest_archive": None,
            "newest_archive": None,
            "archives_by_month": {},
        }

        if not stats["archive_directory_exists"]:
            return stats

        try:
            archive_files = []

            for archive_file in self.archive_directory.iterdir():
                if archive_file.is_file():
                    file_info = {
                        "path": archive_file,
                        "size": archive_file.stat().st_size,
                        "mtime": datetime.fromtimestamp(archive_file.stat().st_mtime),
                        "compressed": archive_file.suffix == ".gz",
                    }
                    archive_files.append(file_info)

                    stats["total_archives"] += 1
                    stats["total_size_bytes"] += file_info["size"]

                    if file_info["compressed"]:
                        stats["compressed_archives"] += 1

            if archive_files:
                # Sort by modification time
                archive_files.sort(key=lambda x: x["mtime"])

                stats["oldest_archive"] = {
                    "path": str(archive_files[0]["path"]),
                    "date": archive_files[0]["mtime"].isoformat(),
                    "size_bytes": archive_files[0]["size"],
                }

                stats["newest_archive"] = {
                    "path": str(archive_files[-1]["path"]),
                    "date": archive_files[-1]["mtime"].isoformat(),
                    "size_bytes": archive_files[-1]["size"],
                }

                # Group by month
                for file_info in archive_files:
                    month_key = file_info["mtime"].strftime("%Y-%m")
                    if month_key not in stats["archives_by_month"]:
                        stats["archives_by_month"][month_key] = {
                            "count": 0,
                            "total_size": 0,
                        }

                    stats["archives_by_month"][month_key]["count"] += 1
                    stats["archives_by_month"][month_key]["total_size"] += file_info[
                        "size"
                    ]

        except Exception as e:
            self.logger.error(f"Failed to get archive statistics: {e}")
            stats["error"] = str(e)

        return stats

    def perform_maintenance(self, force_rotation: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive maintenance operations.

        Args:
            force_rotation: Force file rotation regardless of size/row limits

        Returns:
            Dict[str, Any]: Maintenance operation results
        """
        results = {
            "maintenance_started": datetime.now().isoformat(),
            "rotation_performed": False,
            "cleanup_performed": False,
            "rotation_results": None,
            "cleanup_results": None,
            "issues": [],
        }

        try:
            self.logger.info("Starting maintenance operations")

            # Check if rotation is needed
            rotation_check = self.check_rotation_needed()

            if force_rotation or rotation_check.get("rotation_needed", False):
                self.logger.info(
                    f"Performing file rotation: {rotation_check.get('reason', 'forced')}"
                )
                rotation_results = self.rotate_csv_file()
                results["rotation_results"] = rotation_results
                results["rotation_performed"] = rotation_results.get("success", False)

                if not rotation_results.get("success", False):
                    results["issues"].extend(rotation_results.get("issues", []))

            # Perform cleanup
            self.logger.info("Performing data cleanup")
            cleanup_results = self.cleanup_old_data()
            results["cleanup_results"] = cleanup_results
            results["cleanup_performed"] = cleanup_results.get("success", False)

            if not cleanup_results.get("success", False):
                results["issues"].extend(cleanup_results.get("cleanup_errors", []))

            results["maintenance_completed"] = datetime.now().isoformat()

            self.logger.info(
                f"Maintenance completed: rotation={results['rotation_performed']}, "
                f"cleanup={results['cleanup_performed']}"
            )

        except Exception as e:
            results["issues"].append(f"Maintenance operation failed: {e}")
            self.logger.error(f"Maintenance failed: {e}")

        return results

    def get_maintenance_recommendations(self) -> Dict[str, Any]:
        """
        Get maintenance recommendations based on current system state.

        Returns:
            Dict[str, Any]: Maintenance recommendations
        """
        recommendations = {
            "rotation_recommended": False,
            "cleanup_recommended": False,
            "compression_recommended": False,
            "recommendations": [],
            "current_status": {},
        }

        try:
            # Check rotation status
            rotation_check = self.check_rotation_needed()
            recommendations["current_status"]["rotation"] = rotation_check

            if rotation_check.get("rotation_needed", False):
                recommendations["rotation_recommended"] = True
                recommendations["recommendations"].append(
                    {
                        "action": "rotate_csv_file",
                        "reason": rotation_check.get("reason"),
                        "priority": (
                            "high"
                            if rotation_check.get("file_size_mb", 0)
                            > self.max_file_size_mb * 1.5
                            else "medium"
                        ),
                    }
                )

            # Check archive statistics
            archive_stats = self.get_archive_statistics()
            recommendations["current_status"]["archives"] = archive_stats

            if archive_stats.get("total_archives", 0) > 10:
                recommendations["cleanup_recommended"] = True
                recommendations["recommendations"].append(
                    {
                        "action": "cleanup_old_data",
                        "reason": f"Found {archive_stats['total_archives']} archived files",
                        "priority": "medium",
                    }
                )

            # Check compression
            if (
                not self.config.compression_enabled
                and archive_stats.get("total_size_bytes", 0) > 100 * 1024 * 1024
            ):  # 100MB
                recommendations["compression_recommended"] = True
                recommendations["recommendations"].append(
                    {
                        "action": "enable_compression",
                        "reason": f"Archives use {archive_stats['total_size_bytes'] / (1024*1024):.1f}MB without compression",
                        "priority": "low",
                    }
                )

        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations["error"] = str(e)

        return recommendations
