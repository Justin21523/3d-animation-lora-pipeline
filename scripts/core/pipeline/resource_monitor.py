#!/usr/bin/env python3
"""
Resource Monitor for Pipeline Orchestrator

Tracks GPU/CPU resources and helps optimize batch sizes and memory usage.

Author: Claude Code
Date: 2025-01-17
"""

import logging
import psutil
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class ResourceStats:
    """Resource statistics snapshot."""
    gpu_memory_used: float  # GB
    gpu_memory_total: float  # GB
    gpu_memory_percent: float  # 0-100
    gpu_utilization: float  # 0-100
    cpu_percent: float  # 0-100
    ram_used: float  # GB
    ram_total: float  # GB
    ram_percent: float  # 0-100
    timestamp: float


class ResourceMonitor:
    """
    Monitor GPU and CPU resources for pipeline optimization.

    Features:
    - Real-time GPU memory tracking
    - CPU and RAM monitoring
    - Batch size recommendations
    - Resource warnings and limits
    """

    def __init__(self,
                 device: str = "cuda",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize resource monitor.

        Args:
            device: Device to monitor ('cuda' or 'cpu')
            logger: Logger instance
        """
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.gpu_available = self._check_gpu_available()

        # Resource limits (configurable)
        self.gpu_memory_limit = 0.90  # Use up to 90% of GPU memory
        self.ram_limit = 0.85  # Use up to 85% of system RAM

        self.logger.info(f"Resource Monitor initialized (device: {device})")
        if self.gpu_available:
            stats = self.get_current_stats()
            self.logger.info(f"GPU Memory: {stats.gpu_memory_total:.1f} GB total")

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available."""
        if self.device == "cpu":
            return False

        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            self.logger.warning("PyTorch not available, GPU monitoring disabled")
            return False

    def get_current_stats(self) -> ResourceStats:
        """
        Get current resource statistics.

        Returns:
            ResourceStats object with current measurements
        """
        # CPU and RAM stats (always available)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        ram_used = ram.used / (1024**3)  # GB
        ram_total = ram.total / (1024**3)  # GB
        ram_percent = ram.percent

        # GPU stats (if available)
        if self.gpu_available:
            gpu_mem_used, gpu_mem_total, gpu_mem_pct, gpu_util = self._get_gpu_stats()
        else:
            gpu_mem_used = gpu_mem_total = gpu_mem_pct = gpu_util = 0.0

        return ResourceStats(
            gpu_memory_used=gpu_mem_used,
            gpu_memory_total=gpu_mem_total,
            gpu_memory_percent=gpu_mem_pct,
            gpu_utilization=gpu_util,
            cpu_percent=cpu_percent,
            ram_used=ram_used,
            ram_total=ram_total,
            ram_percent=ram_percent,
            timestamp=time.time()
        )

    def _get_gpu_stats(self) -> Tuple[float, float, float, float]:
        """
        Get GPU statistics.

        Returns:
            (memory_used_gb, memory_total_gb, memory_percent, utilization_percent)
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return 0.0, 0.0, 0.0, 0.0

            # Get memory stats
            mem_used = torch.cuda.memory_allocated() / (1024**3)  # GB
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0.0

            # Try to get GPU utilization (requires nvidia-ml-py)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                pynvml.nvmlShutdown()
            except (ImportError, Exception):
                gpu_util = 0.0

            return mem_used, mem_total, mem_percent, gpu_util

        except Exception as e:
            self.logger.debug(f"Error getting GPU stats: {e}")
            return 0.0, 0.0, 0.0, 0.0

    def check_gpu_memory_available(self, required_gb: float) -> bool:
        """
        Check if sufficient GPU memory is available.

        Args:
            required_gb: Required memory in GB

        Returns:
            True if sufficient memory available
        """
        if not self.gpu_available:
            return False

        stats = self.get_current_stats()
        available = stats.gpu_memory_total - stats.gpu_memory_used

        return available >= required_gb

    def get_recommended_batch_size(self,
                                   base_batch_size: int,
                                   memory_per_item: float,
                                   min_batch_size: int = 1,
                                   max_batch_size: int = 128) -> int:
        """
        Recommend batch size based on available GPU memory.

        Args:
            base_batch_size: Desired batch size
            memory_per_item: Estimated memory per item in GB
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size

        Returns:
            Recommended batch size
        """
        if not self.gpu_available:
            return base_batch_size

        stats = self.get_current_stats()

        # Calculate available memory (accounting for limit)
        available_memory = stats.gpu_memory_total * self.gpu_memory_limit - stats.gpu_memory_used

        if available_memory <= 0:
            self.logger.warning("GPU memory exhausted, using minimum batch size")
            return min_batch_size

        # Calculate max batch size that fits in memory
        max_by_memory = int(available_memory / memory_per_item)

        # Recommend batch size
        recommended = min(base_batch_size, max_by_memory, max_batch_size)
        recommended = max(recommended, min_batch_size)

        if recommended != base_batch_size:
            self.logger.info(
                f"Batch size adjusted: {base_batch_size} → {recommended} "
                f"(available memory: {available_memory:.1f} GB)"
            )

        return recommended

    def log_current_stats(self, level: int = logging.INFO):
        """
        Log current resource statistics.

        Args:
            level: Logging level
        """
        stats = self.get_current_stats()

        msg_parts = [
            f"CPU: {stats.cpu_percent:.1f}%",
            f"RAM: {stats.ram_used:.1f}/{stats.ram_total:.1f} GB ({stats.ram_percent:.1f}%)"
        ]

        if self.gpu_available:
            msg_parts.insert(0,
                f"GPU: {stats.gpu_memory_used:.1f}/{stats.gpu_memory_total:.1f} GB "
                f"({stats.gpu_memory_percent:.1f}%)"
            )
            if stats.gpu_utilization > 0:
                msg_parts[0] += f", Util: {stats.gpu_utilization:.0f}%"

        self.logger.log(level, " | ".join(msg_parts))

    def check_resource_warnings(self) -> Dict[str, str]:
        """
        Check for resource warnings.

        Returns:
            Dictionary of warning messages (empty if no warnings)
        """
        warnings = {}
        stats = self.get_current_stats()

        # GPU memory warnings
        if self.gpu_available:
            if stats.gpu_memory_percent > 95:
                warnings['gpu_memory'] = (
                    f"GPU memory critical: {stats.gpu_memory_percent:.1f}% used"
                )
            elif stats.gpu_memory_percent > self.gpu_memory_limit * 100:
                warnings['gpu_memory'] = (
                    f"GPU memory high: {stats.gpu_memory_percent:.1f}% used "
                    f"(limit: {self.gpu_memory_limit*100:.0f}%)"
                )

        # RAM warnings
        if stats.ram_percent > 95:
            warnings['ram'] = f"System RAM critical: {stats.ram_percent:.1f}% used"
        elif stats.ram_percent > self.ram_limit * 100:
            warnings['ram'] = (
                f"System RAM high: {stats.ram_percent:.1f}% used "
                f"(limit: {self.ram_limit*100:.0f}%)"
            )

        return warnings

    def wait_for_memory(self,
                       required_gb: float,
                       timeout: float = 60.0,
                       check_interval: float = 1.0) -> bool:
        """
        Wait for sufficient GPU memory to become available.

        Args:
            required_gb: Required memory in GB
            timeout: Maximum wait time in seconds
            check_interval: Check interval in seconds

        Returns:
            True if memory became available, False if timeout
        """
        if not self.gpu_available:
            return True

        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.check_gpu_memory_available(required_gb):
                return True

            time.sleep(check_interval)

        self.logger.warning(
            f"Timeout waiting for {required_gb:.1f} GB GPU memory "
            f"(waited {timeout:.0f}s)"
        )
        return False

    def clear_gpu_cache(self):
        """Clear GPU cache to free memory."""
        if not self.gpu_available:
            return

        try:
            import torch
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared")
        except Exception as e:
            self.logger.debug(f"Error clearing GPU cache: {e}")

    def get_memory_summary(self) -> str:
        """
        Get formatted memory summary.

        Returns:
            Formatted string with memory statistics
        """
        stats = self.get_current_stats()

        lines = [
            "=== Resource Summary ===",
            f"RAM: {stats.ram_used:.1f}/{stats.ram_total:.1f} GB ({stats.ram_percent:.1f}%)",
            f"CPU: {stats.cpu_percent:.1f}%"
        ]

        if self.gpu_available:
            lines.insert(1,
                f"GPU: {stats.gpu_memory_used:.1f}/{stats.gpu_memory_total:.1f} GB "
                f"({stats.gpu_memory_percent:.1f}%)"
            )

        return "\n".join(lines)


def main():
    """Test resource monitor."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Resource Monitor")
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--monitor-duration', type=int, default=10,
                       help='Monitoring duration in seconds')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create monitor
    monitor = ResourceMonitor(device=args.device)

    print("\n" + monitor.get_memory_summary())

    # Monitor for duration
    print(f"\nMonitoring resources for {args.monitor_duration} seconds...")
    print("(Press Ctrl+C to stop)\n")

    try:
        for i in range(args.monitor_duration):
            monitor.log_current_stats()

            # Check warnings
            warnings = monitor.check_resource_warnings()
            for warn_type, warn_msg in warnings.items():
                logging.warning(f"[{warn_type}] {warn_msg}")

            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

    print("\n" + monitor.get_memory_summary())


if __name__ == "__main__":
    main()
