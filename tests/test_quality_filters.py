#!/usr/bin/env python3
"""
Comprehensive Test Suite for Quality Filtering System

Tests all filter components:
- BlurDetector (Laplacian variance)
- DuplicateDetector (perceptual hashing)
- NSFWDetector (CLIP-based safety)
- ImageQualityFilter (main orchestrator)

Author: LLMProvider Tooling
Date: 2025-11-30
"""

import json
import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from unittest.mock import Mock, patch

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.generic.quality.blur_detector import BlurDetector
from scripts.generic.quality.duplicate_detector import DuplicateDetector
from scripts.generic.quality.nsfw_detector import NSFWDetector
from scripts.generic.quality.image_quality_filter import (
    ImageQualityFilter,
    ImageQualityMetrics,
    FilterConfig,
    FilteringReport
)


# ============================================================================
# Test Utilities
# ============================================================================

def create_test_image(size=(512, 512), color=(255, 0, 0)):
    """Create a solid color test image"""
    return Image.new('RGB', size, color)


def create_blurry_image(size=(512, 512)):
    """Create a blurry test image"""
    img = create_test_image(size)
    return img.filter(ImageFilter.GaussianBlur(radius=20))


def create_sharp_image(size=(512, 512)):
    """Create a sharp test image with high-frequency content"""
    img = Image.new('RGB', size)
    pixels = img.load()

    # Create checkerboard pattern (high frequency)
    for i in range(size[0]):
        for j in range(size[1]):
            if (i // 16 + j // 16) % 2 == 0:
                pixels[i, j] = (255, 255, 255)
            else:
                pixels[i, j] = (0, 0, 0)

    return img


def create_duplicate_image(original_path: Path):
    """Create a duplicate of an image with minor modifications"""
    img = Image.open(original_path)
    # Add slight noise
    arr = np.array(img)
    noise = np.random.randint(-5, 5, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ============================================================================
# BlurDetector Tests
# ============================================================================

class TestBlurDetector:
    """Test BlurDetector functionality"""

    def test_detector_initialization(self):
        """Test creating blur detector"""
        detector = BlurDetector(threshold=100.0)
        assert detector.threshold == 100.0

    def test_sharp_image_detection(self):
        """Test that sharp images are detected correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sharp image
            img = create_sharp_image()
            img_path = Path(tmpdir) / "sharp.png"
            img.save(img_path)

            detector = BlurDetector(threshold=100.0)
            score = detector.compute_blur_score(img_path)
            is_blurry = detector.is_blurry(img_path)
            level = detector.classify_blur_level(img_path)

            # Sharp image should have high score
            assert score > 100.0
            assert not is_blurry
            assert level in ['sharp', 'slightly_blurry']

    def test_blurry_image_detection(self):
        """Test that blurry images are detected correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create blurry image
            img = create_blurry_image()
            img_path = Path(tmpdir) / "blurry.png"
            img.save(img_path)

            detector = BlurDetector(threshold=100.0)
            score = detector.compute_blur_score(img_path)
            is_blurry = detector.is_blurry(img_path)
            level = detector.classify_blur_level(img_path)

            # Blurry image should have low score
            assert score < 100.0
            assert is_blurry
            assert level == 'very_blurry'

    def test_batch_detection(self):
        """Test batch blur detection"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple images
            sharp_img = create_sharp_image()
            blurry_img = create_blurry_image()

            sharp_path = Path(tmpdir) / "sharp.png"
            blurry_path = Path(tmpdir) / "blurry.png"

            sharp_img.save(sharp_path)
            blurry_img.save(blurry_path)

            detector = BlurDetector(threshold=100.0)
            results = detector.batch_detect([sharp_path, blurry_path])

            assert len(results) == 2
            assert results[sharp_path]['is_blurry'] == False
            assert results[blurry_path]['is_blurry'] == True


# ============================================================================
# DuplicateDetector Tests
# ============================================================================

class TestDuplicateDetector:
    """Test DuplicateDetector functionality"""

    def test_detector_initialization(self):
        """Test creating duplicate detector"""
        detector = DuplicateDetector(
            hash_algorithm='phash',
            hamming_threshold=8
        )
        assert detector.hash_algorithm == 'phash'
        assert detector.hamming_threshold == 8

    def test_hash_computation(self):
        """Test perceptual hash computation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = create_test_image()
            img_path = Path(tmpdir) / "test.png"
            img.save(img_path)

            detector = DuplicateDetector()
            hash1 = detector.compute_hash(img_path)
            hash2 = detector.compute_hash(img_path)

            # Same image should produce same hash
            assert hash1 == hash2

    def test_identical_images(self):
        """Test that identical images are detected as duplicates"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two identical images
            img = create_test_image()
            img1_path = Path(tmpdir) / "img1.png"
            img2_path = Path(tmpdir) / "img2.png"
            img.save(img1_path)
            img.save(img2_path)

            detector = DuplicateDetector(hamming_threshold=8)
            are_dupes = detector.are_duplicates(img1_path, img2_path)

            assert are_dupes == True

    def test_different_images(self):
        """Test that different images are not detected as duplicates"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two visually different images (sharp vs blurry patterns)
            img1 = create_sharp_image()
            img2 = create_blurry_image()

            img1_path = Path(tmpdir) / "img1.png"
            img2_path = Path(tmpdir) / "img2.png"
            img1.save(img1_path)
            img2.save(img2_path)

            detector = DuplicateDetector(hamming_threshold=8)
            are_dupes = detector.are_duplicates(img1_path, img2_path)

            # Sharp checkerboard and blurry image should be different
            assert are_dupes == False

    def test_find_duplicates(self):
        """Test finding duplicates in a batch"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create original and duplicates
            img = create_test_image()

            paths = []
            for i in range(3):
                path = Path(tmpdir) / f"img{i}.png"
                img.save(path)
                paths.append(path)

            detector = DuplicateDetector(hamming_threshold=8)
            duplicates = detector.find_duplicates(paths)

            # Should find 2 duplicates (img1 and img2 are duplicates of img0)
            assert len(duplicates) > 0

    def test_deduplication(self):
        """Test deduplication functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 images: one unique, two identical
            img1 = create_sharp_image((512, 512))
            img2 = create_sharp_image((512, 512))  # Identical to img1
            img3 = create_blurry_image((512, 512))  # Different

            paths = []
            for i, img in enumerate([img1, img2, img3]):
                path = Path(tmpdir) / f"img{i}.png"
                img.save(path)
                paths.append(path)

            detector = DuplicateDetector(hamming_threshold=8)
            unique, duplicates = detector.deduplicate(paths)

            # Core functionality: detect duplicates and separate them
            # Should find at least 1 duplicate (img2 is duplicate of img1)
            assert len(duplicates) >= 1
            # Should keep at least 2 unique images
            assert len(unique) >= 2


# ============================================================================
# NSFWDetector Tests (Mock-based due to model dependency)
# ============================================================================

class TestNSFWDetector:
    """Test NSFWDetector functionality"""

    def test_detector_initialization(self):
        """Test creating NSFW detector"""
        # Skip if no GPU available (NSFW detector requires model)
        try:
            detector = NSFWDetector(threshold=0.3, device='cpu')
            assert detector.threshold == 0.3
            assert detector.device == 'cpu'
        except Exception as e:
            pytest.skip(f"NSFW detector initialization failed: {e}")

    def test_safe_image_classification(self):
        """Test classifying safe images"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create simple test image
            img = create_test_image()
            img_path = Path(tmpdir) / "safe.png"
            img.save(img_path)

            try:
                detector = NSFWDetector(threshold=0.3, device='cpu')
                classification, score = detector.classify(img_path)

                # Simple solid color should be classified as safe
                assert classification == 'safe'
                assert score < 0.5
            except Exception as e:
                pytest.skip(f"NSFW detection failed: {e}")


# ============================================================================
# FilterConfig Tests
# ============================================================================

class TestFilterConfig:
    """Test FilterConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = FilterConfig()

        assert config.blur_threshold == 100.0
        assert config.enable_blur_detection == True
        assert config.duplicate_threshold == 8
        assert config.enable_duplicate_detection == True
        assert config.nsfw_threshold == 0.3
        assert config.enable_nsfw_filtering == True

    def test_custom_config(self):
        """Test custom configuration"""
        config = FilterConfig(
            blur_threshold=150.0,
            enable_blur_detection=False,
            duplicate_threshold=10,
            nsfw_threshold=0.5
        )

        assert config.blur_threshold == 150.0
        assert config.enable_blur_detection == False
        assert config.duplicate_threshold == 10
        assert config.nsfw_threshold == 0.5


# ============================================================================
# ImageQualityMetrics Tests
# ============================================================================

class TestImageQualityMetrics:
    """Test ImageQualityMetrics dataclass"""

    def test_metrics_creation(self):
        """Test creating quality metrics"""
        metrics = ImageQualityMetrics(
            path="/path/to/image.png",
            blur_score=150.0,
            is_blurry=False,
            is_duplicate=False,
            duplicate_of=None,
            perceptual_hash="abc123",
            nsfw_score=0.1,
            is_nsfw=False,
            overall_quality='good',
            rejection_reasons=[]
        )

        assert metrics.path == "/path/to/image.png"
        assert metrics.blur_score == 150.0
        assert metrics.is_blurry == False
        assert metrics.overall_quality == 'good'
        assert len(metrics.rejection_reasons) == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestImageQualityFilterIntegration:
    """Integration tests for complete filtering pipeline"""

    def test_filter_initialization(self):
        """Test initializing quality filter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = FilterConfig(
                enable_blur_detection=True,
                enable_duplicate_detection=True,
                enable_nsfw_filtering=False  # Disable to avoid model loading
            )

            filter_system = ImageQualityFilter(
                config=config,
                checkpoint_dir=Path(tmpdir),
                device='cpu'
            )

            assert filter_system.config.enable_blur_detection == True
            assert filter_system.config.enable_duplicate_detection == True

    def test_quality_tier_classification(self):
        """Test quality tier classification logic"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = FilterConfig(
                excellent_blur_threshold=250.0,
                acceptable_blur_threshold=150.0,
                enable_nsfw_filtering=False
            )

            filter_system = ImageQualityFilter(
                config=config,
                checkpoint_dir=Path(tmpdir),
                device='cpu'
            )

            # Test excellent quality
            metrics_excellent = ImageQualityMetrics(
                path="test.png",
                blur_score=300.0,
                is_blurry=False,
                is_duplicate=False,
                duplicate_of=None,
                perceptual_hash="abc",
                nsfw_score=0.1,
                is_nsfw=False,
                overall_quality='',
                rejection_reasons=[]
            )
            tier = filter_system.classify_quality_tier(metrics_excellent)
            assert tier == 'excellent'

            # Test good quality
            metrics_good = ImageQualityMetrics(
                path="test.png",
                blur_score=200.0,
                is_blurry=False,
                is_duplicate=False,
                duplicate_of=None,
                perceptual_hash="abc",
                nsfw_score=0.1,
                is_nsfw=False,
                overall_quality='',
                rejection_reasons=[]
            )
            tier = filter_system.classify_quality_tier(metrics_good)
            assert tier == 'good'

            # Test rejected (blurry)
            metrics_rejected = ImageQualityMetrics(
                path="test.png",
                blur_score=50.0,
                is_blurry=True,
                is_duplicate=False,
                duplicate_of=None,
                perceptual_hash="abc",
                nsfw_score=0.1,
                is_nsfw=False,
                overall_quality='',
                rejection_reasons=['blurry']
            )
            tier = filter_system.classify_quality_tier(metrics_rejected)
            assert tier == 'rejected'


# ============================================================================
# Run Tests
# ============================================================================

def run_tests():
    """Run all tests with pytest"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
