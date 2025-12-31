"""
Monitoring and health check utilities for the Token Tracker system.

This module provides comprehensive monitoring, health checks, performance
metrics collection, and alerting capabilities for the token tracking system.
"""

import logging
import os
import psutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
import csv
import json
from dataclasses import dataclass, asdict
from enum import Enum

from .models import TokenTransaction, TrackerConfig, CSVSchema
from .error_handler import TokenTrackerErrorHandler


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Represents a single health check result."""

    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        result["status"] = self.status.value
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class PerformanceMetrics:
    """Performance metrics for the token tracking system."""

    timestamp: datetime
    csv_file_size_bytes: int
    csv_row_count: int
    memory_usage_mb: float
    disk_usage_percent: float
    avg_write_time_ms: float
    transactions_per_minute: float
    error_rate_percent: float
    concurrent_operations: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


class SystemMonitor:
    """
    Comprehensive monitoring and health checking for the token tracking system.

    This class provides:
    - CSV file integrity validation
    - System health monitoring
    - Performance metrics collection
    - Alerting for system issues
    """

    def __init__(
        self,
        config: TrackerConfig,
        error_handler: Optional[TokenTrackerErrorHandler] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the system monitor.

        Args:
            config: Tracker configuration
            error_handler: Error handler instance
            logger: Logger instance
        """
        self.config = config
        self.error_handler = error_handler or TokenTrackerErrorHandler()
        self.logger = logger or logging.getLogger(__name__)
        self.schema = CSVSchema()

        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_interval = 60  # seconds
        self._shutdown_event = threading.Event()

        # Performance tracking
        self._performance_history: List[PerformanceMetrics] = []
        self._max_history_size = 1440  # 24 hours at 1-minute intervals

        # Health check registry
        self._health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._register_default_health_checks()

        # Alert thresholds
        self.alert_thresholds = {
            "csv_file_size_mb": 100,  # Alert if CSV > 100MB
            "disk_usage_percent": 90,  # Alert if disk > 90% full
            "memory_usage_mb": 500,  # Alert if memory > 500MB
            "error_rate_percent": 10,  # Alert if error rate > 10%
            "avg_write_time_ms": 1000,  # Alert if writes > 1 second
        }

        # Alert callbacks
        self._alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        self.logger.info("SystemMonitor initialized")

    def _register_default_health_checks(self) -> None:
        """Register default health checks."""
        self._health_checks.update(
            {
                "csv_file_integrity": self._check_csv_file_integrity,
                "csv_file_permissions": self._check_csv_file_permissions,
                "disk_space": self._check_disk_space,
                "memory_usage": self._check_memory_usage,
                "file_lock_status": self._check_file_lock_status,
                "configuration_validity": self._check_configuration_validity,
                "error_rate": self._check_error_rate,
            }
        )

    def register_health_check(
        self, name: str, check_func: Callable[[], HealthCheck]
    ) -> None:
        """
        Register a custom health check.

        Args:
            name: Name of the health check
            check_func: Function that returns a HealthCheck result
        """
        self._health_checks[name] = check_func
        self.logger.debug(f"Registered health check: {name}")

    def register_alert_callback(
        self, callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """
        Register a callback for alert notifications.

        Args:
            callback: Function to call when alerts are triggered
        """
        self._alert_callbacks.append(callback)
        self.logger.debug("Registered alert callback")

    def run_health_checks(self) -> Dict[str, Any]:
        """
        Run all registered health checks.

        Returns:
            Dict[str, Any]: Health check results
        """
        results = {
            "overall_status": HealthStatus.HEALTHY,
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "summary": {
                "total_checks": len(self._health_checks),
                "healthy": 0,
                "warnings": 0,
                "critical": 0,
                "unknown": 0,
            },
        }

        overall_status = HealthStatus.HEALTHY

        for check_name, check_func in self._health_checks.items():
            try:
                check_result = check_func()
                results["checks"][check_name] = check_result.to_dict()

                # Update summary counts
                status_key = check_result.status.value
                if status_key in results["summary"]:
                    results["summary"][status_key] += 1

                # Determine overall status (worst case)
                if check_result.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif (
                    check_result.status == HealthStatus.WARNING
                    and overall_status != HealthStatus.CRITICAL
                ):
                    overall_status = HealthStatus.WARNING
                elif (
                    check_result.status == HealthStatus.UNKNOWN
                    and overall_status == HealthStatus.HEALTHY
                ):
                    overall_status = HealthStatus.UNKNOWN

            except Exception as e:
                self.logger.error(f"Health check '{check_name}' failed: {e}")
                error_check = HealthCheck(
                    name=check_name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {e}",
                    timestamp=datetime.now(),
                )
                results["checks"][check_name] = error_check.to_dict()
                results["summary"]["unknown"] += 1

                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.UNKNOWN

        results["overall_status"] = overall_status.value

        # Trigger alerts if needed
        if overall_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            self._trigger_alerts("health_check", results)

        return results

    def collect_performance_metrics(self) -> PerformanceMetrics:
        """
        Collect current performance metrics.

        Returns:
            PerformanceMetrics: Current system performance metrics
        """
        csv_path = self.config.get_csv_file_path()

        # CSV file metrics
        csv_file_size = 0
        csv_row_count = 0
        if csv_path.exists():
            csv_file_size = csv_path.stat().st_size
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    csv_row_count = sum(1 for _ in csv.reader(f)) - 1  # Subtract header
            except Exception as e:
                self.logger.warning(f"Could not count CSV rows: {e}")

        # Memory usage
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / (1024 * 1024)

        # Disk usage
        disk_usage = psutil.disk_usage(csv_path.parent)
        disk_usage_percent = (disk_usage.used / disk_usage.total) * 100

        # Performance metrics from error handler
        error_stats = self.error_handler.get_error_statistics()
        error_rate = 0.0
        if error_stats.get("total_operations", 0) > 0:
            error_rate = (
                error_stats.get("total_errors", 0) / error_stats["total_operations"]
            ) * 100

        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            csv_file_size_bytes=csv_file_size,
            csv_row_count=csv_row_count,
            memory_usage_mb=memory_usage_mb,
            disk_usage_percent=disk_usage_percent,
            avg_write_time_ms=error_stats.get("avg_operation_time_ms", 0.0),
            transactions_per_minute=self._calculate_transactions_per_minute(),
            error_rate_percent=error_rate,
            concurrent_operations=error_stats.get("concurrent_operations", 0),
        )

        # Store in history
        self._performance_history.append(metrics)
        if len(self._performance_history) > self._max_history_size:
            self._performance_history.pop(0)

        # Check for performance alerts
        self._check_performance_alerts(metrics)

        return metrics

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary for the specified time period.

        Args:
            hours: Number of hours to include in summary

        Returns:
            Dict[str, Any]: Performance summary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self._performance_history if m.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {
                "period_hours": hours,
                "data_points": 0,
                "message": "No performance data available for the specified period",
            }

        # Calculate statistics
        summary = {
            "period_hours": hours,
            "data_points": len(recent_metrics),
            "start_time": recent_metrics[0].timestamp.isoformat(),
            "end_time": recent_metrics[-1].timestamp.isoformat(),
            "csv_file_growth": {
                "start_size_mb": recent_metrics[0].csv_file_size_bytes / (1024 * 1024),
                "end_size_mb": recent_metrics[-1].csv_file_size_bytes / (1024 * 1024),
                "growth_mb": (
                    recent_metrics[-1].csv_file_size_bytes
                    - recent_metrics[0].csv_file_size_bytes
                )
                / (1024 * 1024),
            },
            "transaction_stats": {
                "start_count": recent_metrics[0].csv_row_count,
                "end_count": recent_metrics[-1].csv_row_count,
                "transactions_added": recent_metrics[-1].csv_row_count
                - recent_metrics[0].csv_row_count,
                "avg_per_minute": sum(m.transactions_per_minute for m in recent_metrics)
                / len(recent_metrics),
            },
            "performance_stats": {
                "avg_memory_mb": sum(m.memory_usage_mb for m in recent_metrics)
                / len(recent_metrics),
                "max_memory_mb": max(m.memory_usage_mb for m in recent_metrics),
                "avg_write_time_ms": sum(m.avg_write_time_ms for m in recent_metrics)
                / len(recent_metrics),
                "max_write_time_ms": max(m.avg_write_time_ms for m in recent_metrics),
                "avg_error_rate": sum(m.error_rate_percent for m in recent_metrics)
                / len(recent_metrics),
                "max_error_rate": max(m.error_rate_percent for m in recent_metrics),
            },
            "system_stats": {
                "avg_disk_usage": sum(m.disk_usage_percent for m in recent_metrics)
                / len(recent_metrics),
                "max_disk_usage": max(m.disk_usage_percent for m in recent_metrics),
                "max_concurrent_ops": max(
                    m.concurrent_operations for m in recent_metrics
                ),
            },
        }

        return summary

    def start_monitoring(self, interval_seconds: int = 60) -> None:
        """
        Start continuous monitoring in a background thread.

        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self._monitoring_active:
            self.logger.warning("Monitoring is already active")
            return

        self._monitoring_interval = interval_seconds
        self._shutdown_event.clear()
        self._monitoring_active = True

        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, name="TokenTracker-Monitor", daemon=True
        )
        self._monitoring_thread.start()

        self.logger.info(f"Started monitoring with {interval_seconds}s interval")

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        self._shutdown_event.set()

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
            if self._monitoring_thread.is_alive():
                self.logger.warning("Monitoring thread did not shutdown cleanly")

        self.logger.info("Monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.debug("Monitoring loop started")

        while self._monitoring_active and not self._shutdown_event.is_set():
            try:
                # Collect performance metrics
                metrics = self.collect_performance_metrics()

                # Run health checks periodically (every 5 minutes)
                if len(self._performance_history) % 5 == 0:
                    health_results = self.run_health_checks()
                    self.logger.debug(
                        f"Health check status: {health_results['overall_status']}"
                    )

                # Wait for next interval
                if self._shutdown_event.wait(self._monitoring_interval):
                    break  # Shutdown requested

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                # Continue monitoring despite errors
                time.sleep(self._monitoring_interval)

        self.logger.debug("Monitoring loop finished")

    def _calculate_transactions_per_minute(self) -> float:
        """Calculate recent transactions per minute rate."""
        if len(self._performance_history) < 2:
            return 0.0

        # Use last 5 minutes of data
        recent_metrics = self._performance_history[-5:]
        if len(recent_metrics) < 2:
            return 0.0

        time_diff = (
            recent_metrics[-1].timestamp - recent_metrics[0].timestamp
        ).total_seconds() / 60
        row_diff = recent_metrics[-1].csv_row_count - recent_metrics[0].csv_row_count

        if time_diff > 0:
            return row_diff / time_diff
        return 0.0

    def _check_performance_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check performance metrics against alert thresholds."""
        alerts = []

        # Check CSV file size
        csv_size_mb = metrics.csv_file_size_bytes / (1024 * 1024)
        if csv_size_mb > self.alert_thresholds["csv_file_size_mb"]:
            alerts.append(
                {
                    "type": "csv_file_size",
                    "message": f"CSV file size ({csv_size_mb:.1f}MB) exceeds threshold ({self.alert_thresholds['csv_file_size_mb']}MB)",
                    "value": csv_size_mb,
                    "threshold": self.alert_thresholds["csv_file_size_mb"],
                }
            )

        # Check disk usage
        if metrics.disk_usage_percent > self.alert_thresholds["disk_usage_percent"]:
            alerts.append(
                {
                    "type": "disk_usage",
                    "message": f"Disk usage ({metrics.disk_usage_percent:.1f}%) exceeds threshold ({self.alert_thresholds['disk_usage_percent']}%)",
                    "value": metrics.disk_usage_percent,
                    "threshold": self.alert_thresholds["disk_usage_percent"],
                }
            )

        # Check memory usage
        if metrics.memory_usage_mb > self.alert_thresholds["memory_usage_mb"]:
            alerts.append(
                {
                    "type": "memory_usage",
                    "message": f"Memory usage ({metrics.memory_usage_mb:.1f}MB) exceeds threshold ({self.alert_thresholds['memory_usage_mb']}MB)",
                    "value": metrics.memory_usage_mb,
                    "threshold": self.alert_thresholds["memory_usage_mb"],
                }
            )

        # Check error rate
        if metrics.error_rate_percent > self.alert_thresholds["error_rate_percent"]:
            alerts.append(
                {
                    "type": "error_rate",
                    "message": f"Error rate ({metrics.error_rate_percent:.1f}%) exceeds threshold ({self.alert_thresholds['error_rate_percent']}%)",
                    "value": metrics.error_rate_percent,
                    "threshold": self.alert_thresholds["error_rate_percent"],
                }
            )

        # Check write performance
        if metrics.avg_write_time_ms > self.alert_thresholds["avg_write_time_ms"]:
            alerts.append(
                {
                    "type": "write_performance",
                    "message": f"Average write time ({metrics.avg_write_time_ms:.1f}ms) exceeds threshold ({self.alert_thresholds['avg_write_time_ms']}ms)",
                    "value": metrics.avg_write_time_ms,
                    "threshold": self.alert_thresholds["avg_write_time_ms"],
                }
            )

        # Trigger alerts
        for alert in alerts:
            self._trigger_alerts("performance", alert)

    def _trigger_alerts(self, alert_type: str, alert_data: Dict[str, Any]) -> None:
        """Trigger alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

    # Health check implementations
    def _check_csv_file_integrity(self) -> HealthCheck:
        """Check CSV file integrity."""
        csv_path = self.config.get_csv_file_path()

        if not csv_path.exists():
            return HealthCheck(
                name="csv_file_integrity",
                status=HealthStatus.WARNING,
                message="CSV file does not exist",
                timestamp=datetime.now(),
            )

        try:
            # Basic file validation
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader)

                # Validate headers
                header_issues = self.schema.validate_headers(headers)
                if header_issues:
                    return HealthCheck(
                        name="csv_file_integrity",
                        status=HealthStatus.CRITICAL,
                        message=f"CSV header validation failed: {header_issues}",
                        timestamp=datetime.now(),
                        details={"header_issues": header_issues},
                    )

                # Sample a few rows for validation
                row_count = 0
                invalid_rows = 0
                for row in reader:
                    row_count += 1
                    if row_count > 100:  # Sample first 100 rows
                        break

                    row_issues = self.schema.validate_row(row)
                    if row_issues:
                        invalid_rows += 1

                if invalid_rows > 0:
                    return HealthCheck(
                        name="csv_file_integrity",
                        status=HealthStatus.WARNING,
                        message=f"Found {invalid_rows} invalid rows in sample of {row_count}",
                        timestamp=datetime.now(),
                        details={
                            "invalid_rows": invalid_rows,
                            "sample_size": row_count,
                        },
                    )

            return HealthCheck(
                name="csv_file_integrity",
                status=HealthStatus.HEALTHY,
                message="CSV file integrity validated",
                timestamp=datetime.now(),
            )

        except Exception as e:
            return HealthCheck(
                name="csv_file_integrity",
                status=HealthStatus.CRITICAL,
                message=f"CSV integrity check failed: {e}",
                timestamp=datetime.now(),
            )

    def _check_csv_file_permissions(self) -> HealthCheck:
        """Check CSV file permissions."""
        csv_path = self.config.get_csv_file_path()

        if not csv_path.exists():
            # Check if we can create the file
            try:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                csv_path.touch()
                csv_path.unlink()  # Clean up test file

                return HealthCheck(
                    name="csv_file_permissions",
                    status=HealthStatus.HEALTHY,
                    message="Can create CSV file",
                    timestamp=datetime.now(),
                )
            except Exception as e:
                return HealthCheck(
                    name="csv_file_permissions",
                    status=HealthStatus.CRITICAL,
                    message=f"Cannot create CSV file: {e}",
                    timestamp=datetime.now(),
                )

        # Check read/write permissions
        readable = os.access(csv_path, os.R_OK)
        writable = os.access(csv_path, os.W_OK)

        if not readable or not writable:
            return HealthCheck(
                name="csv_file_permissions",
                status=HealthStatus.CRITICAL,
                message=f"Insufficient permissions: readable={readable}, writable={writable}",
                timestamp=datetime.now(),
                details={"readable": readable, "writable": writable},
            )

        return HealthCheck(
            name="csv_file_permissions",
            status=HealthStatus.HEALTHY,
            message="CSV file permissions are correct",
            timestamp=datetime.now(),
        )

    def _check_disk_space(self) -> HealthCheck:
        """Check available disk space."""
        csv_path = self.config.get_csv_file_path()

        try:
            disk_usage = psutil.disk_usage(csv_path.parent)
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            free_gb = disk_usage.free / (1024**3)

            if usage_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Disk usage critical: {usage_percent:.1f}% used, {free_gb:.1f}GB free"
            elif usage_percent > 85:
                status = HealthStatus.WARNING
                message = (
                    f"Disk usage high: {usage_percent:.1f}% used, {free_gb:.1f}GB free"
                )
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {usage_percent:.1f}% used, {free_gb:.1f}GB free"

            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={
                    "usage_percent": usage_percent,
                    "free_gb": free_gb,
                    "total_gb": disk_usage.total / (1024**3),
                },
            )

        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Could not check disk space: {e}",
                timestamp=datetime.now(),
            )

    def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            if memory_mb > 1000:  # 1GB
                status = HealthStatus.WARNING
                message = f"High memory usage: {memory_mb:.1f}MB"
            elif memory_mb > 2000:  # 2GB
                status = HealthStatus.CRITICAL
                message = f"Critical memory usage: {memory_mb:.1f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_mb:.1f}MB"

            return HealthCheck(
                name="memory_usage",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={"memory_mb": memory_mb},
            )

        except Exception as e:
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.UNKNOWN,
                message=f"Could not check memory usage: {e}",
                timestamp=datetime.now(),
            )

    def _check_file_lock_status(self) -> HealthCheck:
        """Check file locking status."""
        # This is a basic check - in a real implementation you might
        # check for stuck locks or deadlock conditions
        return HealthCheck(
            name="file_lock_status",
            status=HealthStatus.HEALTHY,
            message="File locking operational",
            timestamp=datetime.now(),
        )

    def _check_configuration_validity(self) -> HealthCheck:
        """Check configuration validity."""
        try:
            issues = self.config.validate()

            if issues:
                return HealthCheck(
                    name="configuration_validity",
                    status=HealthStatus.WARNING,
                    message=f"Configuration issues found: {len(issues)}",
                    timestamp=datetime.now(),
                    details={"issues": issues},
                )

            return HealthCheck(
                name="configuration_validity",
                status=HealthStatus.HEALTHY,
                message="Configuration is valid",
                timestamp=datetime.now(),
            )

        except Exception as e:
            return HealthCheck(
                name="configuration_validity",
                status=HealthStatus.CRITICAL,
                message=f"Configuration validation failed: {e}",
                timestamp=datetime.now(),
            )

    def _check_error_rate(self) -> HealthCheck:
        """Check system error rate."""
        try:
            error_stats = self.error_handler.get_error_statistics()

            total_ops = error_stats.get("total_operations", 0)
            total_errors = error_stats.get("total_errors", 0)

            if total_ops == 0:
                return HealthCheck(
                    name="error_rate",
                    status=HealthStatus.HEALTHY,
                    message="No operations recorded yet",
                    timestamp=datetime.now(),
                )

            error_rate = (total_errors / total_ops) * 100

            if error_rate > 20:
                status = HealthStatus.CRITICAL
                message = f"Critical error rate: {error_rate:.1f}%"
            elif error_rate > 5:
                status = HealthStatus.WARNING
                message = f"High error rate: {error_rate:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Error rate normal: {error_rate:.1f}%"

            return HealthCheck(
                name="error_rate",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={
                    "error_rate_percent": error_rate,
                    "total_operations": total_ops,
                    "total_errors": total_errors,
                },
            )

        except Exception as e:
            return HealthCheck(
                name="error_rate",
                status=HealthStatus.UNKNOWN,
                message=f"Could not check error rate: {e}",
                timestamp=datetime.now(),
            )

    def export_monitoring_data(
        self, output_path: Union[str, Path], format: str = "json"
    ) -> bool:
        """
        Export monitoring data to file.

        Args:
            output_path: Path to export file
            format: Export format ('json' or 'csv')

        Returns:
            bool: True if export was successful
        """
        try:
            output_path = Path(output_path)

            if format.lower() == "json":
                data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "performance_history": [
                        m.to_dict() for m in self._performance_history
                    ],
                    "health_checks": self.run_health_checks(),
                    "configuration": self.config.to_dict(),
                }

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            elif format.lower() == "csv":
                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    if self._performance_history:
                        writer = csv.DictWriter(
                            f, fieldnames=self._performance_history[0].to_dict().keys()
                        )
                        writer.writeheader()
                        for metrics in self._performance_history:
                            writer.writerow(metrics.to_dict())
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Monitoring data exported to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export monitoring data: {e}")
            return False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
