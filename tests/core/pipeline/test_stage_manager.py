#!/usr/bin/env python3
"""
Unit Tests for StageManager

Tests stage registration, dependency resolution, and execution tracking.

Author: Claude Code
Date: 2025-01-17
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
import json
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.core.pipeline.stage_manager import (
    StageManager,
    Stage,
    StageStatus,
    StageResult
)


class TestStage(unittest.TestCase):
    """Test cases for Stage dataclass"""

    def test_stage_creation(self):
        """Test creating a Stage object"""
        func = MagicMock()
        stage = Stage(
            name="test_stage",
            function=func,
            description="Test stage",
            dependencies=[],
            enabled=True,
            optional=False
        )

        self.assertEqual(stage.name, "test_stage")
        self.assertEqual(stage.function, func)
        self.assertEqual(stage.description, "Test stage")
        self.assertEqual(stage.dependencies, [])
        self.assertTrue(stage.enabled)
        self.assertFalse(stage.optional)
        self.assertEqual(stage.status, StageStatus.PENDING)

    def test_stage_defaults(self):
        """Test Stage default values"""
        func = MagicMock()
        stage = Stage(
            name="test",
            function=func,
            description="Test"
        )

        self.assertEqual(stage.dependencies, [])
        self.assertTrue(stage.enabled)
        self.assertFalse(stage.optional)
        self.assertEqual(stage.status, StageStatus.PENDING)


class TestStageStatus(unittest.TestCase):
    """Test cases for StageStatus enum"""

    def test_stage_status_values(self):
        """Test StageStatus enum values"""
        self.assertEqual(StageStatus.PENDING.value, 'pending')
        self.assertEqual(StageStatus.RUNNING.value, 'running')
        self.assertEqual(StageStatus.COMPLETED.value, 'completed')
        self.assertEqual(StageStatus.FAILED.value, 'failed')
        self.assertEqual(StageStatus.SKIPPED.value, 'skipped')


class TestStageResult(unittest.TestCase):
    """Test cases for StageResult dataclass"""

    def test_stage_result_success(self):
        """Test creating successful StageResult"""
        result = StageResult(
            success=True,
            stage="test_stage",
            message="Success"
        )

        self.assertTrue(result.success)
        self.assertEqual(result.stage, "test_stage")
        self.assertEqual(result.message, "Success")
        self.assertIsNone(result.error)
        self.assertEqual(result.outputs, {})

    def test_stage_result_failure(self):
        """Test creating failed StageResult"""
        result = StageResult(
            success=False,
            stage="test_stage",
            message="Failed",
            error="Error message"
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Error message")


class TestStageManager(unittest.TestCase):
    """Test cases for StageManager class"""

    def setUp(self):
        """Set up test fixtures"""
        self.manager = StageManager()

    # ====== Stage Registration Tests ======

    def test_register_stage(self):
        """Test registering a stage"""
        func = MagicMock()
        self.manager.register_stage(
            name="test_stage",
            function=func,
            description="Test stage"
        )

        self.assertIn("test_stage", self.manager.stages)
        stage = self.manager.stages["test_stage"]
        self.assertEqual(stage.name, "test_stage")
        self.assertEqual(stage.function, func)

    def test_register_stage_with_dependencies(self):
        """Test registering stages with dependencies"""
        func1 = MagicMock()
        func2 = MagicMock()

        self.manager.register_stage("stage1", func1, "Stage 1")
        self.manager.register_stage(
            "stage2",
            func2,
            "Stage 2",
            dependencies=["stage1"]
        )

        stage2 = self.manager.stages["stage2"]
        self.assertEqual(stage2.dependencies, ["stage1"])

    def test_register_duplicate_stage_raises_error(self):
        """Test registering duplicate stage name raises error"""
        func = MagicMock()
        self.manager.register_stage("test", func, "Test")

        with self.assertRaises(ValueError):
            self.manager.register_stage("test", func, "Duplicate")

    def test_register_stage_with_unknown_dependency_raises_error(self):
        """Test registering stage with unknown dependency raises error"""
        func = MagicMock()

        with self.assertRaises(ValueError):
            self.manager.register_stage(
                "test",
                func,
                "Test",
                dependencies=["nonexistent"]
            )

    # ====== Dependency Resolution Tests ======

    def test_resolve_dependencies_no_deps(self):
        """Test dependency resolution with no dependencies"""
        func = MagicMock()
        self.manager.register_stage("stage1", func, "Stage 1")
        self.manager.register_stage("stage2", func, "Stage 2")

        order = self.manager.resolve_dependencies()

        # Both stages should be in order
        self.assertEqual(len(order), 2)
        self.assertIn("stage1", order)
        self.assertIn("stage2", order)

    def test_resolve_dependencies_linear_chain(self):
        """Test dependency resolution with linear chain"""
        func = MagicMock()
        self.manager.register_stage("stage1", func, "Stage 1")
        self.manager.register_stage("stage2", func, "Stage 2", dependencies=["stage1"])
        self.manager.register_stage("stage3", func, "Stage 3", dependencies=["stage2"])

        order = self.manager.resolve_dependencies()

        # Should be in correct order
        self.assertEqual(order, ["stage1", "stage2", "stage3"])

    def test_resolve_dependencies_diamond(self):
        """Test dependency resolution with diamond dependency"""
        func = MagicMock()
        #     A
        #    / \
        #   B   C
        #    \ /
        #     D
        self.manager.register_stage("A", func, "A")
        self.manager.register_stage("B", func, "B", dependencies=["A"])
        self.manager.register_stage("C", func, "C", dependencies=["A"])
        self.manager.register_stage("D", func, "D", dependencies=["B", "C"])

        order = self.manager.resolve_dependencies()

        # A must be first, D must be last, B and C between them
        self.assertEqual(order[0], "A")
        self.assertEqual(order[-1], "D")
        self.assertIn("B", order[1:3])
        self.assertIn("C", order[1:3])

    def test_resolve_dependencies_circular_raises_error(self):
        """Test circular dependency raises error"""
        func = MagicMock()
        self.manager.register_stage("stage1", func, "Stage 1", dependencies=[])

        # Temporarily bypass dependency check to create circular dependency
        stage2 = Stage("stage2", func, "Stage 2", dependencies=["stage3"])
        stage3 = Stage("stage3", func, "Stage 3", dependencies=["stage2"])

        self.manager.stages["stage2"] = stage2
        self.manager.stages["stage3"] = stage3

        with self.assertRaises(ValueError) as ctx:
            self.manager.resolve_dependencies()

        self.assertIn("Circular dependency", str(ctx.exception))

    def test_resolve_dependencies_respects_disabled(self):
        """Test dependency resolution skips disabled stages"""
        func = MagicMock()
        self.manager.register_stage("stage1", func, "Stage 1")
        self.manager.register_stage("stage2", func, "Stage 2", enabled=False)
        self.manager.register_stage("stage3", func, "Stage 3")

        order = self.manager.resolve_dependencies()

        # stage2 should not be in order
        self.assertNotIn("stage2", order)
        self.assertIn("stage1", order)
        self.assertIn("stage3", order)

    # ====== Stage Execution Tests ======

    def test_execute_stage_success(self):
        """Test executing a stage successfully"""
        func = MagicMock(return_value={
            'success': True,
            'stage': 'test_stage',
            'message': 'Success'
        })

        self.manager.register_stage("test_stage", func, "Test")
        result = self.manager.execute_stage("test_stage", config={})

        self.assertTrue(result.success)
        self.assertEqual(result.stage, "test_stage")
        func.assert_called_once()

        # Check stage status updated
        stage = self.manager.stages["test_stage"]
        self.assertEqual(stage.status, StageStatus.COMPLETED)

    def test_execute_stage_failure(self):
        """Test executing a stage that fails"""
        func = MagicMock(return_value={
            'success': False,
            'stage': 'test_stage',
            'message': 'Failed',
            'error': 'Test error'
        })

        self.manager.register_stage("test_stage", func, "Test")
        result = self.manager.execute_stage("test_stage", config={})

        self.assertFalse(result.success)
        self.assertEqual(result.error, 'Test error')

        # Check stage status updated
        stage = self.manager.stages["test_stage"]
        self.assertEqual(stage.status, StageStatus.FAILED)
        self.assertEqual(stage.error_message, 'Test error')

    def test_execute_stage_exception(self):
        """Test executing a stage that raises exception"""
        func = MagicMock(side_effect=RuntimeError("Test exception"))

        self.manager.register_stage("test_stage", func, "Test")
        result = self.manager.execute_stage("test_stage", config={})

        self.assertFalse(result.success)
        self.assertIn("Test exception", result.error)

        # Check stage status updated
        stage = self.manager.stages["test_stage"]
        self.assertEqual(stage.status, StageStatus.FAILED)

    def test_execute_nonexistent_stage(self):
        """Test executing nonexistent stage raises error"""
        with self.assertRaises(ValueError):
            self.manager.execute_stage("nonexistent", config={})

    def test_execute_disabled_stage(self):
        """Test executing disabled stage is skipped"""
        func = MagicMock()
        self.manager.register_stage("test_stage", func, "Test", enabled=False)

        result = self.manager.execute_stage("test_stage", config={})

        self.assertTrue(result.success)  # Success but skipped
        func.assert_not_called()

        # Check stage status updated
        stage = self.manager.stages["test_stage"]
        self.assertEqual(stage.status, StageStatus.SKIPPED)

    # ====== Status and Progress Tests ======

    def test_get_stage_status(self):
        """Test getting stage status"""
        func = MagicMock()
        self.manager.register_stage("test_stage", func, "Test")

        status = self.manager.get_stage_status("test_stage")
        self.assertEqual(status, StageStatus.PENDING)

        # Execute stage
        func.return_value = {'success': True, 'stage': 'test_stage'}
        self.manager.execute_stage("test_stage", {})

        status = self.manager.get_stage_status("test_stage")
        self.assertEqual(status, StageStatus.COMPLETED)

    def test_get_progress(self):
        """Test getting execution progress"""
        func = MagicMock(return_value={'success': True, 'stage': 'test'})

        self.manager.register_stage("stage1", func, "Stage 1")
        self.manager.register_stage("stage2", func, "Stage 2")
        self.manager.register_stage("stage3", func, "Stage 3")

        # Initially 0% complete
        progress = self.manager.get_progress()
        self.assertEqual(progress['completed'], 0)
        self.assertEqual(progress['total'], 3)
        self.assertEqual(progress['percent'], 0.0)

        # Execute first stage
        func.return_value['stage'] = 'stage1'
        self.manager.execute_stage("stage1", {})

        progress = self.manager.get_progress()
        self.assertEqual(progress['completed'], 1)
        self.assertAlmostEqual(progress['percent'], 33.33, places=2)

        # Execute all stages
        func.return_value['stage'] = 'stage2'
        self.manager.execute_stage("stage2", {})
        func.return_value['stage'] = 'stage3'
        self.manager.execute_stage("stage3", {})

        progress = self.manager.get_progress()
        self.assertEqual(progress['completed'], 3)
        self.assertEqual(progress['percent'], 100.0)

    def test_get_progress_with_disabled_stages(self):
        """Test progress calculation excludes disabled stages"""
        func = MagicMock(return_value={'success': True, 'stage': 'test'})

        self.manager.register_stage("stage1", func, "Stage 1")
        self.manager.register_stage("stage2", func, "Stage 2", enabled=False)
        self.manager.register_stage("stage3", func, "Stage 3")

        progress = self.manager.get_progress()
        # Should count only enabled stages (2 total)
        self.assertEqual(progress['total'], 2)

    # ====== Checkpoint Save/Load Tests ======

    def test_save_checkpoint(self):
        """Test saving checkpoint"""
        func = MagicMock(return_value={'success': True, 'stage': 'test'})

        self.manager.register_stage("stage1", func, "Stage 1")
        self.manager.register_stage("stage2", func, "Stage 2")

        # Execute first stage
        func.return_value['stage'] = 'stage1'
        self.manager.execute_stage("stage1", {})

        # Save checkpoint
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            checkpoint_path = f.name

        try:
            self.manager.save_checkpoint(checkpoint_path)

            # Check file exists and is valid JSON
            self.assertTrue(Path(checkpoint_path).exists())

            with open(checkpoint_path, 'r') as f:
                data = json.load(f)

            self.assertIn('stages', data)
            self.assertEqual(len(data['stages']), 2)
            self.assertEqual(data['stages']['stage1']['status'], 'completed')
            self.assertEqual(data['stages']['stage2']['status'], 'pending')

        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_load_checkpoint(self):
        """Test loading checkpoint"""
        func = MagicMock(return_value={'success': True, 'stage': 'test'})

        self.manager.register_stage("stage1", func, "Stage 1")
        self.manager.register_stage("stage2", func, "Stage 2")

        # Create checkpoint data
        checkpoint_data = {
            'stages': {
                'stage1': {
                    'status': 'completed',
                    'enabled': True,
                    'error_message': None
                },
                'stage2': {
                    'status': 'pending',
                    'enabled': True,
                    'error_message': None
                }
            }
        }

        # Write checkpoint
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(checkpoint_data, f)
            checkpoint_path = f.name

        try:
            # Load checkpoint
            self.manager.load_checkpoint(checkpoint_path)

            # Check states restored
            self.assertEqual(
                self.manager.stages['stage1'].status,
                StageStatus.COMPLETED
            )
            self.assertEqual(
                self.manager.stages['stage2'].status,
                StageStatus.PENDING
            )

        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_load_checkpoint_nonexistent_file(self):
        """Test loading nonexistent checkpoint raises error"""
        with self.assertRaises(FileNotFoundError):
            self.manager.load_checkpoint("/nonexistent/checkpoint.json")

    # ====== Reset Tests ======

    def test_reset_all_stages(self):
        """Test resetting all stages"""
        func = MagicMock(return_value={'success': True, 'stage': 'test'})

        self.manager.register_stage("stage1", func, "Stage 1")
        self.manager.register_stage("stage2", func, "Stage 2")

        # Execute stages
        func.return_value['stage'] = 'stage1'
        self.manager.execute_stage("stage1", {})
        func.return_value['stage'] = 'stage2'
        self.manager.execute_stage("stage2", {})

        # Reset
        self.manager.reset()

        # All stages should be pending
        for stage in self.manager.stages.values():
            self.assertEqual(stage.status, StageStatus.PENDING)
            self.assertIsNone(stage.error_message)

    def test_reset_specific_stage(self):
        """Test resetting specific stage"""
        func = MagicMock(return_value={'success': True, 'stage': 'test'})

        self.manager.register_stage("stage1", func, "Stage 1")
        self.manager.register_stage("stage2", func, "Stage 2")

        # Execute both
        func.return_value['stage'] = 'stage1'
        self.manager.execute_stage("stage1", {})
        func.return_value['stage'] = 'stage2'
        self.manager.execute_stage("stage2", {})

        # Reset only stage1
        self.manager.reset(stage_name="stage1")

        # stage1 should be pending, stage2 still completed
        self.assertEqual(
            self.manager.stages['stage1'].status,
            StageStatus.PENDING
        )
        self.assertEqual(
            self.manager.stages['stage2'].status,
            StageStatus.COMPLETED
        )


if __name__ == '__main__':
    unittest.main()
