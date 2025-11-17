#!/usr/bin/env python3
"""
Prepare Pose/Action LoRA Training Data (RTM-Pose Version)

Prepares pose-specific character images for pose/action LoRA training by:
1. Loading character instances from SAM2 output
2. Detecting pose keypoints using RTM-Pose (MMPose)
3. Classifying actions/poses using geometric rules
4. Filtering by target action
5. Generating pose-specific captions
6. Organizing into kohya_ss training format

Features:
- RTM-Pose keypoint detection (COCO 17 keypoints)
- Rule-based pose classification (standing, running, walking, etc.)
- Keypoint normalization and geometric analysis
- Pose-specific caption generation

Usage:
    python prepare_pose_lora_data.py \\
        --character-instances /path/to/instances/ \\
        --output-dir /path/to/training_data/running_pose/ \\
        --action-name "running" \\
        --target-size 200 \\
        --device cuda

Author: AI Pipeline
Date: 2025-01-17
"""

import sys
import shutil
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import imagehash

# Add scripts directory to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.core.utils.logger import setup_logger
from scripts.core.utils.path_utils import ensure_dir
from scripts.core.utils.checkpoint_manager import CheckpointManager
from scripts.core.pose_estimation import RTMPoseDetector, RuleBasedPoseClassifier, PoseNormalizer, ViewClassifier


# Pose action definitions (extended descriptions)
POSE_ACTION_DESCRIPTIONS = {
    "standing": {
        "keywords": ["neutral stance", "standing pose", "upright", "standing still"],
        "caption_template": "{character} standing pose, neutral stance, upright, {style}"
    },
    "walking": {
        "keywords": ["walking", "stepping", "in motion", "casual walk"],
        "caption_template": "{character} walking pose, stepping forward, in motion, {style}"
    },
    "running": {
        "keywords": ["running", "sprinting", "dynamic motion", "forward lean"],
        "caption_template": "{character} running pose, dynamic motion, sprinting, {style}"
    },
    "jumping": {
        "keywords": ["jumping", "mid-air", "airborne", "leap"],
        "caption_template": "{character} jumping pose, mid-air, airborne, {style}"
    },
    "sitting": {
        "keywords": ["sitting", "seated", "relaxed pose", "sitting down"],
        "caption_template": "{character} sitting pose, seated position, relaxed, {style}"
    },
    "crouching": {
        "keywords": ["crouching", "squatting", "bent knees", "low position"],
        "caption_template": "{character} crouching pose, low position, bent knees, {style}"
    },
    "reaching": {
        "keywords": ["reaching out", "extended arm", "gesture", "reaching for"],
        "caption_template": "{character} reaching pose, extended arm, gesture, {style}"
    },
}


class PoseQualityFilter:
    """
    Quality filtering for pose images.

    Filters:
    - Blur detection (Laplacian variance)
    - Size validation
    - Aspect ratio checks
    """

    def __init__(self,
                 min_blur_score: float = 100.0,
                 min_width: int = 128,
                 min_height: int = 128,
                 max_aspect_ratio: float = 3.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize quality filter.

        Args:
            min_blur_score: Minimum Laplacian variance (lower = more blur)
            min_width: Minimum image width
            min_height: Minimum image height
            max_aspect_ratio: Maximum width/height ratio
            logger: Logger instance
        """
        self.min_blur_score = min_blur_score
        self.min_width = min_width
        self.min_height = min_height
        self.max_aspect_ratio = max_aspect_ratio
        self.logger = logger or logging.getLogger(__name__)

    def check_blur(self, image_path: Path) -> Tuple[bool, float]:
        """
        Check if image is too blurry.

        Args:
            image_path: Path to image

        Returns:
            (is_acceptable, blur_score)
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            return False, 0.0

        # Compute Laplacian variance
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        blur_score = laplacian.var()

        is_acceptable = blur_score >= self.min_blur_score

        return is_acceptable, blur_score

    def check_size(self, image_path: Path) -> Tuple[bool, int, int]:
        """
        Check if image meets size requirements.

        Args:
            image_path: Path to image

        Returns:
            (is_acceptable, width, height)
        """
        img = Image.open(image_path)
        width, height = img.size

        is_acceptable = (width >= self.min_width and height >= self.min_height)

        return is_acceptable, width, height

    def check_aspect_ratio(self, width: int, height: int) -> bool:
        """Check if aspect ratio is acceptable."""
        aspect_ratio = max(width, height) / (min(width, height) + 1e-8)
        return aspect_ratio <= self.max_aspect_ratio

    def filter_image(self, image_path: Path) -> Tuple[bool, Dict]:
        """
        Run all quality checks.

        Args:
            image_path: Path to image

        Returns:
            (passes_all_checks, check_results_dict)
        """
        results = {}

        # Size check
        size_ok, width, height = self.check_size(image_path)
        results['size_ok'] = size_ok
        results['width'] = width
        results['height'] = height

        # Aspect ratio check
        aspect_ok = self.check_aspect_ratio(width, height)
        results['aspect_ok'] = aspect_ok

        # Blur check
        blur_ok, blur_score = self.check_blur(image_path)
        results['blur_ok'] = blur_ok
        results['blur_score'] = blur_score

        # Overall pass/fail
        passes = size_ok and aspect_ok and blur_ok

        return passes, results


class PerceptualHashDeduplicator:
    """
    Deduplication using perceptual hashing.

    Uses imagehash library (pHash/aHash) to detect near-duplicate images.
    """

    def __init__(self,
                 hash_size: int = 16,
                 threshold: int = 8,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize deduplicator.

        Args:
            hash_size: Hash size (larger = more sensitive)
            threshold: Hamming distance threshold (lower = stricter)
            logger: Logger instance
        """
        self.hash_size = hash_size
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)

        # Track seen hashes
        self.seen_hashes = {}  # hash -> image_path

    def compute_hash(self, image_path: Path) -> imagehash.ImageHash:
        """
        Compute perceptual hash.

        Args:
            image_path: Path to image

        Returns:
            ImageHash object
        """
        img = Image.open(image_path)
        phash = imagehash.average_hash(img, hash_size=self.hash_size)
        return phash

    def is_duplicate(self, image_path: Path) -> Tuple[bool, Optional[Path]]:
        """
        Check if image is duplicate of previously seen image.

        Args:
            image_path: Path to image

        Returns:
            (is_duplicate, original_image_path)
        """
        phash = self.compute_hash(image_path)

        # Check against seen hashes
        for seen_hash, seen_path in self.seen_hashes.items():
            distance = phash - seen_hash  # Hamming distance

            if distance <= self.threshold:
                # Duplicate found
                return True, seen_path

        # Not a duplicate, add to seen hashes
        self.seen_hashes[phash] = image_path

        return False, None

    def reset(self):
        """Clear seen hashes."""
        self.seen_hashes.clear()


class PoseLoRADataPreparer:
    """
    Prepare pose/action LoRA training data using RTM-Pose.
    """

    def __init__(self,
                 character_instances_dir: Path,
                 output_dir: Path,
                 action_name: str,
                 character_name: Optional[str] = None,
                 style_description: str = "pixar style, 3d animation, smooth shading",
                 device: str = 'cuda',
                 logger=None):
        """
        Initialize pose LoRA data preparer.

        Args:
            character_instances_dir: Directory containing character instances.
            output_dir: Output directory for training data.
            action_name: Action/pose name (e.g., "running", "jumping").
            character_name: Character name (for captions).
            style_description: Style description for captions.
            device: Device ('cuda' or 'cpu').
            logger: Logger instance.
        """
        self.character_instances_dir = Path(character_instances_dir)
        self.output_dir = Path(output_dir)
        self.action_name = action_name.lower()
        self.character_name = character_name or "character"
        self.style_description = style_description
        self.device = device
        self.logger = logger or setup_logger(__name__)

        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.captions_dir = self.output_dir / "captions"
        self.vis_dir = self.output_dir / "visualizations"
        self.logs_dir = self.output_dir / "logs"

        for dir_path in [self.images_dir, self.captions_dir, self.vis_dir, self.logs_dir]:
            ensure_dir(dir_path)

        # Initialize pose estimation components
        self.logger.info("Initializing RTM-Pose detector...")
        self.pose_detector = RTMPoseDetector(
            model_size='m',  # Medium size (balance of speed/accuracy)
            device=self.device,
            conf_threshold=0.3,  # 3D characters: lower threshold
            logger=self.logger
        )

        self.logger.info("Initializing pose classifier...")
        self.pose_classifier = RuleBasedPoseClassifier(logger=self.logger)

        self.logger.info("Initializing view classifier...")
        self.view_classifier = ViewClassifier(logger=self.logger)

        self.logger.info("Initializing pose normalizer...")
        self.pose_normalizer = PoseNormalizer()

        # Quality filtering and deduplication
        self.logger.info("Initializing quality filter...")
        self.quality_filter = PoseQualityFilter(
            min_blur_score=100.0,
            min_width=128,
            min_height=128,
            logger=self.logger
        )

        self.logger.info("Initializing deduplicator...")
        self.deduplicator = PerceptualHashDeduplicator(
            hash_size=16,
            threshold=8,
            logger=self.logger
        )

        # Initialize checkpoint manager for resume capability
        self.logger.info("Initializing checkpoint manager...")
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_path=self.output_dir / ".pose_detection_checkpoint.json",
            save_interval=50,  # Save every 50 images (pose detection is medium speed)
            logger=self.logger
        )

        self.logger.info(f"Pose LoRA Data Preparer initialized")
        self.logger.info(f"  Character Instances: {self.character_instances_dir}")
        self.logger.info(f"  Output: {self.output_dir}")
        self.logger.info(f"  Target Action: {self.action_name}")
        self.logger.info(f"  Device: {self.device}")

    def prepare_dataset(self, target_size: int = 200, visualize: bool = False):
        """
        Full pipeline for preparing pose LoRA dataset.

        Args:
            target_size: Target number of images.
            visualize: Save pose visualizations.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Preparing {self.action_name} Pose LoRA Dataset")
        self.logger.info("=" * 60)

        # Step 1: Scan instances
        self.logger.info("Step 1: Scanning character instances...")
        instance_files = self._scan_instances()

        if len(instance_files) == 0:
            self.logger.error("No instance files found!")
            return

        # Step 1.5: Quality filtering
        self.logger.info("Step 1.5: Quality filtering (blur, size, aspect ratio)...")
        filtered_files, filter_stats = self._filter_quality(instance_files)

        self.logger.info(f"Quality filtering results:")
        self.logger.info(f"  Passed: {filter_stats['passed']}")
        self.logger.info(f"  Failed (size): {filter_stats['size_failed']}")
        self.logger.info(f"  Failed (aspect): {filter_stats['aspect_failed']}")
        self.logger.info(f"  Failed (blur): {filter_stats['blur_failed']}")

        if len(filtered_files) == 0:
            self.logger.error("No files passed quality filtering!")
            return

        instance_files = filtered_files

        # Step 2: Detect poses
        self.logger.info("Step 2: Detecting poses with RTM-Pose...")
        pose_results = self._detect_poses(instance_files)

        # Step 3: Classify poses
        self.logger.info("Step 3: Classifying poses...")
        classified_results = self._classify_poses(pose_results)

        # Step 4: Filter by target action
        self.logger.info(f"Step 4: Filtering for '{self.action_name}' action...")
        target_results = self._filter_by_action(classified_results)

        if len(target_results) == 0:
            self.logger.error(f"No instances found for action '{self.action_name}'!")
            return

        # Step 4.5: Deduplication
        self.logger.info("Step 4.5: Deduplicating images...")
        deduplicated_results, dup_count = self._deduplicate_poses(target_results)

        self.logger.info(f"Deduplication: removed {dup_count} duplicates, kept {len(deduplicated_results)}")

        if len(deduplicated_results) == 0:
            self.logger.error("No images remaining after deduplication!")
            return

        target_results = deduplicated_results

        # Step 5: Limit to target size
        if len(target_results) > target_size:
            self.logger.info(f"Step 5: Sampling {target_size} images from {len(target_results)}...")
            target_results = self._sample_diverse(target_results, target_size)
        else:
            self.logger.info(f"Step 5: Using all {len(target_results)} matching images...")

        # Step 6: Generate captions and copy images
        self.logger.info("Step 6: Generating captions and organizing dataset...")
        self._assemble_dataset(target_results, visualize)

        # Step 7: Save metadata
        self.logger.info("Step 7: Saving metadata...")
        self._save_metadata(target_results)

        # Clean up checkpoint on successful completion
        self.checkpoint_mgr.cleanup()

        self.logger.info("=" * 60)
        self.logger.info(f"✅ Pose LoRA dataset preparation complete!")
        self.logger.info("=" * 60)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Total images: {len(target_results)}")
        self.logger.info(f"Action: {self.action_name}")

    def _scan_instances(self) -> List[Path]:
        """Scan character instance files."""
        patterns = ['*.png', '*.jpg', '*.jpeg']
        instance_files = []

        for pattern in patterns:
            instance_files.extend(self.character_instances_dir.glob(pattern))

        instance_files = sorted(instance_files)
        self.logger.info(f"Found {len(instance_files)} character instances")

        return instance_files

    def _detect_poses(self, instance_files: List[Path]) -> List[Dict]:
        """
        Detect poses using RTM-Pose with checkpoint/resume support.

        Returns:
            List of dicts with 'image_path', 'keypoints', 'scores', 'success'
        """
        # Load checkpoint if exists
        if self.checkpoint_mgr.exists():
            self.checkpoint_mgr.load()
            self.logger.info(f"Resuming pose detection...")

        # Get unprocessed files
        unprocessed = self.checkpoint_mgr.get_unprocessed_items(instance_files)
        self.logger.info(f"Processing {len(unprocessed)} remaining images (already processed: {len(self.checkpoint_mgr)})...")

        results = []

        for img_path in tqdm(unprocessed, desc="Detecting poses"):
            # Detect pose
            result = self.pose_detector.detect_with_person_bbox(str(img_path))

            results.append({
                'image_path': img_path,
                'keypoints': result['keypoints'],
                'scores': result['scores'],
                'success': result['success'],
                'num_valid': result['num_valid']
            })

            # Mark as processed (auto-saves every 50 items)
            self.checkpoint_mgr.mark_processed(img_path)

        # Force save final checkpoint
        self.checkpoint_mgr.save(force=True)

        successful = sum(1 for r in results if r['success'])
        self.logger.info(f"Successful pose detections: {successful}/{len(results)}")

        return results

    def _classify_poses(self, pose_results: List[Dict]) -> List[Dict]:
        """
        Classify poses using rule-based classifier.

        Returns:
            List of dicts with added 'action' and 'confidence' fields
        """
        classified = []

        for result in pose_results:
            if not result['success']:
                result['action'] = 'unknown'
                result['confidence'] = 0.0
                classified.append(result)
                continue

            # Classify pose action
            action, confidence = self.pose_classifier.classify(
                result['keypoints'],
                result['scores']
            )

            result['action'] = action
            result['confidence'] = confidence

            # Classify view angle
            view, view_conf = self.view_classifier.classify(
                result['keypoints'],
                result['scores']
            )

            result['view'] = view
            result['view_confidence'] = view_conf
            classified.append(result)

        # Count actions
        action_counts = defaultdict(int)
        for r in classified:
            action_counts[r['action']] += 1

        self.logger.info("Action distribution:")
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            self.logger.info(f"  {action}: {count}")

        return classified

    def _filter_by_action(self, classified_results: List[Dict]) -> List[Dict]:
        """Filter results by target action."""
        filtered = [
            r for r in classified_results
            if r['action'] == self.action_name and r['confidence'] > 0.5
        ]

        self.logger.info(f"Filtered {len(filtered)} images for action '{self.action_name}'")

        return filtered

    def _sample_diverse(self, results: List[Dict], target_size: int) -> List[Dict]:
        """
        Sample diverse poses from results using normalized keypoint features.

        Strategy:
        1. Normalize all keypoints (center, scale)
        2. Compute pairwise distances in pose space
        3. Greedy farthest-point sampling for maximum diversity
        4. Ensure balanced view angle distribution
        """
        # Extract and normalize keypoints
        normalized_poses = []
        valid_results = []

        for result in results:
            if not result['success']:
                continue

            # Normalize keypoints
            norm_kpts, success = self.pose_normalizer.normalize(
                result['keypoints'],
                result['scores']
            )

            if success:
                # Flatten to 1D feature vector
                feature = norm_kpts.flatten()
                normalized_poses.append(feature)
                valid_results.append(result)

        if len(valid_results) <= target_size:
            return valid_results

        # Convert to numpy array
        features = np.array(normalized_poses)  # (N, D)

        # Greedy farthest-point sampling
        sampled_indices = []
        remaining_indices = list(range(len(features)))

        # Start with random seed
        np.random.seed(42)
        first_idx = np.random.choice(remaining_indices)
        sampled_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Iteratively select farthest points
        for _ in tqdm(range(target_size - 1), desc="Diversity sampling"):
            if not remaining_indices:
                break

            # Compute min distance to selected points
            selected_features = features[sampled_indices]  # (M, D)
            remaining_features = features[remaining_indices]  # (N_rem, D)

            # Euclidean distance
            distances = np.sqrt(
                ((remaining_features[:, None, :] - selected_features[None, :, :]) ** 2).sum(axis=2)
            )  # (N_rem, M)

            min_distances = distances.min(axis=1)  # (N_rem,)

            # Select farthest point
            farthest_local_idx = min_distances.argmax()
            farthest_idx = remaining_indices[farthest_local_idx]

            sampled_indices.append(farthest_idx)
            remaining_indices.remove(farthest_idx)

        # Return sampled results
        sampled = [valid_results[i] for i in sampled_indices]

        # Log view distribution
        view_counts = defaultdict(int)
        for r in sampled:
            view_counts[r.get('view', 'unknown')] += 1

        self.logger.info(f"Diversity sampling complete. View distribution:")
        for view, count in sorted(view_counts.items(), key=lambda x: -x[1]):
            self.logger.info(f"  {view}: {count}")

        return sampled

    def _assemble_dataset(self, results: List[Dict], visualize: bool):
        """
        Generate captions and copy images to output directory.
        """
        for i, result in enumerate(tqdm(results, desc="Assembling dataset")):
            img_path = result['image_path']
            img_name = f"{self.action_name}_{i:05d}.png"

            # Copy image
            dest_path = self.images_dir / img_name
            shutil.copy2(img_path, dest_path)

            # Generate caption
            caption = self._generate_caption(result)

            # Save caption
            caption_path = self.captions_dir / f"{img_name.rsplit('.', 1)[0]}.txt"
            with open(caption_path, 'w') as f:
                f.write(caption)

            # Visualize if requested
            if visualize:
                vis_path = self.vis_dir / img_name
                self.pose_detector.visualize(
                    str(img_path),
                    result['keypoints'],
                    result['scores'],
                    str(vis_path)
                )

    def _generate_caption(self, result: Dict) -> str:
        """
        Generate pose-specific caption with view angle information.

        Args:
            result: Detection/classification result dict.

        Returns:
            Caption string.
        """
        action = result['action']
        view = result.get('view', 'unknown')

        # Get template
        if action in POSE_ACTION_DESCRIPTIONS:
            template = POSE_ACTION_DESCRIPTIONS[action]['caption_template']
        else:
            template = "{character} {action} pose, {style}"

        # Format base caption
        caption = template.format(
            character=self.character_name,
            action=action,
            style=self.style_description
        )

        # Add view information if available and not unknown
        if view != 'unknown' and view != '':
            # Convert view name to natural language
            view_descriptions = {
                'front': 'front view',
                'three_quarter_right': 'three-quarter view from right',
                'three_quarter_left': 'three-quarter view from left',
                'side_right': 'right side profile',
                'side_left': 'left side profile',
                'back': 'back view',
                'back_right': 'back three-quarter view from right',
                'back_left': 'back three-quarter view from left',
            }

            view_desc = view_descriptions.get(view, view.replace('_', ' '))
            caption = f"{caption}, {view_desc}"

        return caption

    def _save_metadata(self, results: List[Dict]):
        """Save dataset metadata."""
        metadata = {
            'action': self.action_name,
            'character': self.character_name,
            'style_description': self.style_description,
            'num_images': len(results),
            'device': self.device,
            'pose_detector': 'RTM-Pose-M',
            'pose_classifier': 'Rule-based geometric',
            'created_at': datetime.now().isoformat(),

            'action_statistics': {
                'target_action': self.action_name,
                'total_detected': len(results),
                'avg_confidence': float(np.mean([r['confidence'] for r in results])),
                'avg_valid_keypoints': float(np.mean([r['num_valid'] for r in results]))
            }
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Metadata saved to {metadata_path}")

    def _filter_quality(self, instance_files: List[Path]) -> Tuple[List[Path], Dict]:
        """
        Filter instances by quality.

        Args:
            instance_files: List of image paths

        Returns:
            (filtered_files, filter_statistics)
        """
        filtered_files = []
        filter_stats = defaultdict(int)

        for img_path in tqdm(instance_files, desc="Quality filtering"):
            passes, results = self.quality_filter.filter_image(img_path)

            if passes:
                filtered_files.append(img_path)
                filter_stats['passed'] += 1
            else:
                if not results['size_ok']:
                    filter_stats['size_failed'] += 1
                if not results['aspect_ok']:
                    filter_stats['aspect_failed'] += 1
                if not results['blur_ok']:
                    filter_stats['blur_failed'] += 1

        return filtered_files, dict(filter_stats)

    def _deduplicate_poses(self, pose_results: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Deduplicate pose results using perceptual hashing.

        Args:
            pose_results: List of pose detection results

        Returns:
            (deduplicated_results, duplicate_count)
        """
        deduplicated_results = []
        dup_count = 0

        self.deduplicator.reset()

        for result in tqdm(pose_results, desc="Deduplicating"):
            img_path = result['image_path']

            is_dup, original_path = self.deduplicator.is_duplicate(img_path)

            if not is_dup:
                deduplicated_results.append(result)
            else:
                dup_count += 1

        return deduplicated_results, dup_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Pose LoRA training data using RTM-Pose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare running pose dataset
  python prepare_pose_lora_data.py \\
      --character-instances /path/to/luca_instances/ \\
      --output-dir /path/to/luca_running_pose/ \\
      --action-name "running" \\
      --character-name "luca" \\
      --target-size 200 \\
      --device cuda

  # Prepare sitting pose dataset with visualizations
  python prepare_pose_lora_data.py \\
      --character-instances /path/to/instances/ \\
      --output-dir /path/to/sitting_pose/ \\
      --action-name "sitting" \\
      --target-size 150 \\
      --visualize \\
      --device cuda
        """
    )

    # Required arguments
    parser.add_argument('--character-instances', type=str, required=True,
                       help='Directory containing character instance images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for training dataset')
    parser.add_argument('--action-name', type=str, required=True,
                       choices=list(POSE_ACTION_DESCRIPTIONS.keys()),
                       help='Target action/pose name')

    # Optional arguments
    parser.add_argument('--character-name', type=str, default='character',
                       help='Character name (for captions)')
    parser.add_argument('--style-description', type=str,
                       default='pixar style, 3d animation, smooth shading',
                       help='Style description for captions')
    parser.add_argument('--target-size', type=int, default=200,
                       help='Target number of images (default: 200)')
    parser.add_argument('--visualize', action='store_true',
                       help='Save pose keypoint visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for processing (default: cuda)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logger
    log_dir = Path(args.output_dir) / "logs"
    ensure_dir(log_dir)

    logger = setup_logger(
        name="pose_lora_prep",
        log_file=log_dir / f"prepare_pose_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO
    )

    # Run pipeline
    preparer = PoseLoRADataPreparer(
        character_instances_dir=args.character_instances,
        output_dir=args.output_dir,
        action_name=args.action_name,
        character_name=args.character_name,
        style_description=args.style_description,
        device=args.device,
        logger=logger
    )

    preparer.prepare_dataset(
        target_size=args.target_size,
        visualize=args.visualize
    )


if __name__ == '__main__':
    main()
