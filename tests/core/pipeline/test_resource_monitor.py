#!/usr/bin/env python3
"""
Unit Tests for ResourceMonitor

Tests GPU/CPU/RAM monitoring, batch size recommendations, and resource warnings.

Author: Claude Code
Date: 2025-01-17
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.core.pipeline.resource_monitor import ResourceMonitor, ResourceStats


class TestResourceStats(unittest.TestCase):
    """Test cases for ResourceStats dataclass"""

    def test_resource_stats_creation(self):
        """Test creating ResourceStats object"""
        stats = ResourceStats(
            gpu_memory_used=4.0,
            gpu_memory_total=16.0,
            gpu_memory_percent=25.0,
            gpu_utilization=75.0,
            cpu_percent=50.0,
            ram_used=8.0,
            ram_total=16.0,
            ram_percent=50.0,
            timestamp=123456.0
        )

        self.assertEqual(stats.gpu_memory_used, 4.0)
        self.assertEqual(stats.gpu_memory_total, 16.0)
        self.assertEqual(stats.cpu_percent, 50.0)
        self.assertEqual(stats.ram_used, 8.0)
        self.assertEqual(stats.timestamp, 123456.0)


class TestResourceMonitor(unittest.TestCase):
    """Test cases for ResourceMonitor class"""

    def setUp(self):
        """Set up test fixtures"""
        # Use CPU mode to avoid GPU dependencies in tests
        self.monitor = ResourceMonitor(device='cpu')

    # ====== Initialization Tests ======

    def test_monitor_initialization_cpu(self):
        """Test ResourceMonitor initializes with CPU device"""
        self.assertEqual(self.monitor.device, 'cpu')
        self.assertFalse(self.monitor.gpu_available)
        self.assertEqual(self.monitor.gpu_memory_limit, 0.90)
        self.assertEqual(self.monitor.ram_limit, 0.85)

    @patch('torch.cuda.is_available')
    def test_monitor_initialization_cuda(self, mock_cuda_available):
        """Test ResourceMonitor initializes with CUDA device"""
        mock_cuda_available.return_value = True

        monitor = ResourceMonitor(device='cuda')

        self.assertEqual(monitor.device, 'cuda')
        self.assertTrue(monitor.gpu_available)

    # ====== Resource Stats Tests ======

    def test_get_current_stats_cpu(self):
        """Test getting current resource statistics in CPU mode"""
        stats = self.monitor.get_current_stats()

        # Check stats object structure
        self.assertIsInstance(stats, ResourceStats)

        # CPU and RAM stats should be present
        self.assertGreaterEqual(stats.cpu_percent, 0)
        self.assertLessEqual(stats.cpu_percent, 100)
        self.assertGreater(stats.ram_total, 0)
        self.assertGreaterEqual(stats.ram_used, 0)
        self.assertGreaterEqual(stats.ram_percent, 0)
        self.assertLessEqual(stats.ram_percent, 100)

        # GPU stats should be 0 in CPU mode
        self.assertEqual(stats.gpu_memory_used, 0.0)
        self.assertEqual(stats.gpu_memory_total, 0.0)
        self.assertEqual(stats.gpu_utilization, 0.0)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.get_device_properties')
    def test_get_current_stats_gpu(self, mock_props, mock_mem_alloc, mock_cuda_avail):
        """Test getting current statistics with GPU"""
        # Mock CUDA available
        mock_cuda_avail.return_value = True
        mock_mem_alloc.return_value = 4 * (1024**3)  # 4GB used

        # Mock device properties
        mock_device = MagicMock()
        mock_device.total_memory = 16 * (1024**3)  # 16GB total
        mock_props.return_value = mock_device

        monitor = ResourceMonitor(device='cuda')
        stats = monitor.get_current_stats()

        # GPU stats should be present
        self.assertGreater(stats.gpu_memory_total, 0)
        self.assertGreaterEqual(stats.gpu_memory_used, 0)

    # ====== Batch Size Recommendation Tests ======

    def test_get_recommended_batch_size_cpu(self):
        """Test batch size recommendation in CPU mode returns base"""
        batch_size = self.monitor.get_recommended_batch_size(
            base_batch_size=8,
            memory_per_item=0.5
        )

        # CPU mode should return base batch size
        self.assertEqual(batch_size, 8)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.get_device_properties')
    def test_get_recommended_batch_size_gpu(self, mock_props, mock_mem_alloc, mock_cuda_avail):
        """Test batch size recommendation with GPU"""
        # Mock 16GB GPU with 2GB used
        mock_cuda_avail.return_value = True
        mock_mem_alloc.return_value = 2 * (1024**3)
        mock_device = MagicMock()
        mock_device.total_memory = 16 * (1024**3)
        mock_props.return_value = mock_device

        monitor = ResourceMonitor(device='cuda')

        # Request batch size with 1GB per item
        # Available: 16 * 0.9 - 2 = 12.4 GB
        # Max batch: 12.4 / 1 = 12
        batch_size = monitor.get_recommended_batch_size(
            base_batch_size=16,
            memory_per_item=1.0
        )

        # Should be capped by available memory
        self.assertLessEqual(batch_size, 16)
        self.assertGreaterEqual(batch_size, 1)

    def test_get_recommended_batch_size_respects_min_max(self):
        """Test batch size recommendation respects min/max bounds"""
        # Note: in CPU mode, the function returns base_batch_size directly
        # So we test this with a GPU mock
        with patch('torch.cuda.is_available') as mock_cuda, \
             patch('torch.cuda.memory_allocated') as mock_mem, \
             patch('torch.cuda.get_device_properties') as mock_props:

            # Mock GPU with limited memory
            mock_cuda.return_value = True
            mock_mem.return_value = 1 * (1024**3)  # 1GB used
            mock_device = MagicMock()
            mock_device.total_memory = 4 * (1024**3)  # 4GB total
            mock_props.return_value = mock_device

            monitor = ResourceMonitor(device='cuda')

            batch_size = monitor.get_recommended_batch_size(
                base_batch_size=100,
                memory_per_item=0.5,
                min_batch_size=4,
                max_batch_size=32
            )

            # Should be within bounds
            self.assertGreaterEqual(batch_size, 4)
            self.assertLessEqual(batch_size, 32)

    # ====== GPU Memory Check Tests ======

    def test_check_gpu_memory_available_cpu(self):
        """Test GPU memory check returns False in CPU mode"""
        available = self.monitor.check_gpu_memory_available(required_gb=2.0)
        self.assertFalse(available)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.get_device_properties')
    def test_check_gpu_memory_available_gpu(self, mock_props, mock_mem_alloc, mock_cuda_avail):
        """Test GPU memory availability check"""
        # Mock 16GB GPU with 4GB used
        mock_cuda_avail.return_value = True
        mock_mem_alloc.return_value = 4 * (1024**3)
        mock_device = MagicMock()
        mock_device.total_memory = 16 * (1024**3)
        mock_props.return_value = mock_device

        monitor = ResourceMonitor(device='cuda')

        # Should have 12GB available
        self.assertTrue(monitor.check_gpu_memory_available(8.0))
        self.assertTrue(monitor.check_gpu_memory_available(12.0))
        self.assertFalse(monitor.check_gpu_memory_available(14.0))

    # ====== Resource Warning Tests ======

    @patch.object(ResourceMonitor, 'get_current_stats')
    def test_check_resource_warnings_normal(self, mock_stats):
        """Test resource warnings under normal conditions"""
        # Mock normal resource usage
        mock_stats.return_value = ResourceStats(
            gpu_memory_used=0.0,
            gpu_memory_total=0.0,
            gpu_memory_percent=0.0,
            gpu_utilization=0.0,
            cpu_percent=30.0,
            ram_used=8.0,
            ram_total=16.0,
            ram_percent=50.0,
            timestamp=123456.0
        )

        warnings = self.monitor.check_resource_warnings()

        # Should have no warnings
        self.assertEqual(len(warnings), 0)

    @patch.object(ResourceMonitor, 'get_current_stats')
    def test_check_resource_warnings_high_ram(self, mock_stats):
        """Test resource warnings with high RAM usage"""
        # Mock high RAM usage (96%)
        mock_stats.return_value = ResourceStats(
            gpu_memory_used=0.0,
            gpu_memory_total=0.0,
            gpu_memory_percent=0.0,
            gpu_utilization=0.0,
            cpu_percent=30.0,
            ram_used=15.36,
            ram_total=16.0,
            ram_percent=96.0,
            timestamp=123456.0
        )

        warnings = self.monitor.check_resource_warnings()

        # Should warn about RAM
        self.assertIn('ram', warnings)
        self.assertIn('critical', warnings['ram'].lower())

    @patch.object(ResourceMonitor, 'get_current_stats')
    def test_check_resource_warnings_high_gpu(self, mock_stats):
        """Test resource warnings with high GPU usage"""
        # Set monitor to GPU mode
        self.monitor.gpu_available = True

        # Mock high GPU usage (96%)
        mock_stats.return_value = ResourceStats(
            gpu_memory_used=15.36,
            gpu_memory_total=16.0,
            gpu_memory_percent=96.0,
            gpu_utilization=95.0,
            cpu_percent=30.0,
            ram_used=8.0,
            ram_total=16.0,
            ram_percent=50.0,
            timestamp=123456.0
        )

        warnings = self.monitor.check_resource_warnings()

        # Should warn about GPU memory
        self.assertIn('gpu_memory', warnings)

    # ====== Memory Summary Tests ======

    def test_get_memory_summary(self):
        """Test getting formatted memory summary"""
        summary = self.monitor.get_memory_summary()

        self.assertIsInstance(summary, str)
        self.assertIn('Resource Summary', summary)
        self.assertIn('RAM:', summary)
        self.assertIn('CPU:', summary)

    # ====== GPU Cache Clearing Tests ======

    def test_clear_gpu_cache_cpu(self):
        """Test clearing GPU cache in CPU mode does nothing"""
        try:
            self.monitor.clear_gpu_cache()
        except Exception as e:
            self.fail(f"clear_gpu_cache raised exception in CPU mode: {e}")

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_clear_gpu_cache_gpu(self, mock_empty_cache, mock_cuda_avail):
        """Test clearing GPU cache"""
        mock_cuda_avail.return_value = True

        monitor = ResourceMonitor(device='cuda')
        monitor.clear_gpu_cache()

        # Should call empty_cache
        mock_empty_cache.assert_called_once()

    # ====== Wait for Memory Tests ======

    def test_wait_for_memory_cpu(self):
        """Test waiting for memory in CPU mode returns immediately"""
        result = self.monitor.wait_for_memory(required_gb=2.0, timeout=1.0)

        # CPU mode should return True immediately
        self.assertTrue(result)

    @patch.object(ResourceMonitor, 'check_gpu_memory_available')
    def test_wait_for_memory_gpu_available(self, mock_check):
        """Test waiting for memory when already available"""
        self.monitor.gpu_available = True
        mock_check.return_value = True

        result = self.monitor.wait_for_memory(required_gb=2.0, timeout=5.0)

        # Should return True immediately
        self.assertTrue(result)
        self.assertEqual(mock_check.call_count, 1)

    @patch.object(ResourceMonitor, 'check_gpu_memory_available')
    def test_wait_for_memory_gpu_timeout(self, mock_check):
        """Test waiting for memory with timeout"""
        self.monitor.gpu_available = True
        mock_check.return_value = False

        result = self.monitor.wait_for_memory(
            required_gb=2.0,
            timeout=0.5,
            check_interval=0.1
        )

        # Should timeout and return False
        self.assertFalse(result)
        # Should have checked multiple times
        self.assertGreater(mock_check.call_count, 1)


if __name__ == '__main__':
    unittest.main()
