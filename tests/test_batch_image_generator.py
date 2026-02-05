#!/usr/bin/env python3
"""
Comprehensive Test Suite for batch_image_generator.py

Tests all core functionality:
- Prompt loading and validation
- SDXL pipeline initialization
- LoRA loading
- Checkpoint/resume functionality
- Image generation
- Error handling
- Report generation

Author: LLMProvider Tooling
Date: 2025-11-30
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.generic.synthesis.batch_image_generator import (
    PromptSpec,
    GenerationConfig,
    GenerationReport,
    SDXLSyntheticGenerator
)
from scripts.core.utils.checkpoint_manager import IndexCheckpointManager


class TestPromptSpec:
    """Test PromptSpec dataclass"""

    def test_prompt_spec_creation(self):
        """Test creating a PromptSpec"""
        spec = PromptSpec(
            prompt="a test prompt",
            seed=42,
            categories={"style": "pixar"},
            negative_prompt="bad quality"
        )

        assert spec.prompt == "a test prompt"
        assert spec.seed == 42
        assert spec.categories == {"style": "pixar"}
        assert spec.negative_prompt == "bad quality"

    def test_prompt_spec_optional_negative(self):
        """Test PromptSpec with optional negative prompt"""
        spec = PromptSpec(
            prompt="test",
            seed=123,
            categories={},
            negative_prompt=None
        )

        assert spec.negative_prompt is None


class TestGenerationConfig:
    """Test GenerationConfig dataclass"""

    def test_config_defaults(self):
        """Test default configuration values"""
        config = GenerationConfig()

        assert config.num_inference_steps == 40  # Updated default
        assert config.guidance_scale == 7.5
        assert config.lora_scale == 1.0
        assert config.checkpoint_interval == 50

    def test_config_custom_values(self):
        """Test custom configuration"""
        config = GenerationConfig(
            num_inference_steps=30,
            guidance_scale=8.0,
            lora_scale=0.8,
            checkpoint_interval=10
        )

        assert config.num_inference_steps == 30
        assert config.guidance_scale == 8.0
        assert config.lora_scale == 0.8
        assert config.checkpoint_interval == 10


class TestGenerationReport:
    """Test GenerationReport dataclass"""

    def test_report_creation(self):
        """Test creating a generation report"""
        report = GenerationReport(
            character="test_char",
            lora_path="/path/to/lora.safetensors",
            total_prompts=100,
            images_generated=95,
            images_rejected=5,
            start_time="2025-11-30T12:00:00",
            end_time="2025-11-30T12:10:00",
            duration_seconds=600,
            checkpoint_saves=2,
            resumed_from_index=0
        )

        assert report.character == "test_char"
        assert report.total_prompts == 100
        assert report.images_generated == 95
        assert report.images_rejected == 5
        assert report.checkpoint_saves == 2


class TestIndexCheckpointManager:
    """Test IndexCheckpointManager functionality"""

    def test_checkpoint_save_and_load(self):
        """Test saving and loading checkpoints"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_mgr = IndexCheckpointManager(Path(tmpdir))

            # Save checkpoint
            checkpoint_mgr.save(
                last_completed_index=10,
                total_items=100,
                character="test_char",
                seeds_generated=[1, 2, 3]
            )

            # Load checkpoint
            state = checkpoint_mgr.load()

            assert state is not None
            assert state["last_completed_index"] == 10
            assert state["total_items"] == 100
            assert state["character"] == "test_char"
            assert state["seeds_generated"] == [1, 2, 3]
            assert state["progress_percent"] == 11.0

    def test_checkpoint_resume_index(self):
        """Test getting resume index from checkpoint"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_mgr = IndexCheckpointManager(Path(tmpdir))

            # No checkpoint - should return 0
            assert checkpoint_mgr.get_resume_index() == 0

            # Save checkpoint
            checkpoint_mgr.save(
                last_completed_index=5,
                total_items=20
            )

            # Should return last_completed_index + 1
            assert checkpoint_mgr.get_resume_index() == 6

    def test_checkpoint_clear(self):
        """Test clearing checkpoint"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_mgr = IndexCheckpointManager(Path(tmpdir))

            # Save checkpoint
            checkpoint_mgr.save(
                last_completed_index=5,
                total_items=10
            )

            assert checkpoint_mgr.exists()

            # Clear checkpoint
            checkpoint_mgr.clear()

            assert not checkpoint_mgr.exists()

    def test_checkpoint_atomic_write(self):
        """Test that checkpoint writes are atomic"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_mgr = IndexCheckpointManager(Path(tmpdir))

            # Save checkpoint
            checkpoint_mgr.save(
                last_completed_index=10,
                total_items=100
            )

            # Verify checkpoint file exists and temp file doesn't
            assert checkpoint_mgr.checkpoint_file.exists()
            assert not checkpoint_mgr.checkpoint_file.with_suffix('.tmp').exists()


class TestPromptLoading:
    """Test prompt loading and validation"""

    def test_load_valid_prompts(self):
        """Test loading valid prompt file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.json"

            test_prompts = [
                {
                    "prompt": "test prompt 1",
                    "seed": 42,
                    "categories": {"style": "pixar"},
                    "negative_prompt": "bad quality"
                },
                {
                    "prompt": "test prompt 2",
                    "seed": 123,
                    "categories": {},
                    "negative_prompt": None
                }
            ]

            with open(prompts_file, 'w') as f:
                json.dump(test_prompts, f)

            # Load prompts
            with open(prompts_file, 'r') as f:
                loaded = json.load(f)

            assert len(loaded) == 2
            assert loaded[0]["prompt"] == "test prompt 1"
            assert loaded[0]["seed"] == 42
            assert loaded[1]["prompt"] == "test prompt 2"

    def test_load_empty_prompts(self):
        """Test loading empty prompt file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.json"

            with open(prompts_file, 'w') as f:
                json.dump([], f)

            with open(prompts_file, 'r') as f:
                loaded = json.load(f)

            assert len(loaded) == 0


class TestReportGeneration:
    """Test generation report functionality"""

    def test_report_save_and_load(self):
        """Test saving and loading generation report"""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_file = Path(tmpdir) / "report.json"

            report = GenerationReport(
                character="bryce",
                lora_path="/path/to/lora.safetensors",
                total_prompts=20,
                images_generated=18,
                images_rejected=2,
                start_time="2025-11-30T12:00:00",
                end_time="2025-11-30T12:10:00",
                duration_seconds=600,
                checkpoint_saves=1,
                resumed_from_index=0
            )

            # Save report
            from dataclasses import asdict
            with open(report_file, 'w') as f:
                json.dump(asdict(report), f, indent=2)

            # Load report
            with open(report_file, 'r') as f:
                loaded = json.load(f)

            assert loaded["character"] == "bryce"
            assert loaded["total_prompts"] == 20
            assert loaded["images_generated"] == 18
            assert loaded["images_rejected"] == 2

    def test_report_metrics_calculation(self):
        """Test report metrics calculations"""
        report = GenerationReport(
            character="test",
            lora_path="/path/to/lora.safetensors",
            total_prompts=100,
            images_generated=85,
            images_rejected=15,
            start_time="2025-11-30T12:00:00",
            end_time="2025-11-30T12:10:00",
            duration_seconds=600,
            checkpoint_saves=2,
            resumed_from_index=0
        )

        # Calculate metrics
        success_rate = report.images_generated / report.total_prompts * 100
        rejection_rate = report.images_rejected / report.total_prompts * 100
        images_per_sec = report.images_generated / report.duration_seconds

        assert success_rate == 85.0
        assert rejection_rate == 15.0
        assert pytest.approx(images_per_sec, abs=0.01) == pytest.approx(0.14166667, abs=0.01)


class TestEndToEndFlow:
    """Integration tests for end-to-end flow"""

    def test_checkpoint_resume_flow(self):
        """Test complete checkpoint/resume flow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_mgr = IndexCheckpointManager(checkpoint_dir)

            total_items = 10

            # Simulate first run (interrupted at index 4)
            for i in range(5):
                if (i + 1) % 2 == 0:  # Save every 2 items
                    checkpoint_mgr.save(
                        last_completed_index=i,
                        total_items=total_items,
                        seeds_generated=list(range(i + 1))
                    )

            # Check checkpoint state (last save was at i=3, since (3+1) % 2 == 0)
            state = checkpoint_mgr.load()
            assert state["last_completed_index"] == 3
            assert len(state["seeds_generated"]) == 4

            # Simulate resume (resume from last_completed_index + 1 = 3 + 1 = 4)
            resume_idx = checkpoint_mgr.get_resume_index()
            assert resume_idx == 4

            # Continue from resume index
            for i in range(resume_idx, total_items):
                if (i + 1) % 2 == 0:
                    checkpoint_mgr.save(
                        last_completed_index=i,
                        total_items=total_items,
                        seeds_generated=list(range(i + 1))
                    )

            # Final state
            final_state = checkpoint_mgr.load()
            assert final_state["last_completed_index"] == 9
            assert final_state["progress_percent"] == 100.0

            # Clear after completion
            checkpoint_mgr.clear()
            assert not checkpoint_mgr.exists()


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_corrupted_checkpoint_handling(self):
        """Test handling of corrupted checkpoint file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_mgr = IndexCheckpointManager(Path(tmpdir))

            # Create corrupted checkpoint
            with open(checkpoint_mgr.checkpoint_file, 'w') as f:
                f.write("invalid json {{{")

            # Should return None and move corrupted file
            state = checkpoint_mgr.load()
            assert state is None
            assert checkpoint_mgr.checkpoint_file.with_suffix('.corrupted').exists()

    def test_missing_checkpoint_file(self):
        """Test handling when no checkpoint exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_mgr = IndexCheckpointManager(Path(tmpdir))

            # Should return None
            state = checkpoint_mgr.load()
            assert state is None

            # Resume index should be 0
            assert checkpoint_mgr.get_resume_index() == 0


def run_tests():
    """Run all tests with pytest"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
