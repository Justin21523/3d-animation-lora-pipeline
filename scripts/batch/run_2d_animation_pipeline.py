#!/usr/bin/env python3
"""
2D Animation Character Pipeline (Complete End-to-End)
=====================================================

Complete pipeline for 2D animation character extraction:
1. Frame Extraction (ffmpeg) - preserves episode structure
2. YOLO Character Detection - per episode
3. ToonOut/BiRefNet Segmentation - per episode

Features:
- **Episode-aware structure**: Each video/episode gets its own subdirectory
- Retry mechanism with exponential backoff
- GPU memory management (automatic cleanup)
- Progress tracking and resumption
- Detailed logging

Directory Structure (Episode Mode):
    project/
    ├── videos/
    │   ├── episode_01.mp4
    │   └── episode_02.mp4
    ├── frames/
    │   ├── ep01/           # frames from episode_01
    │   └── ep02/           # frames from episode_02
    ├── yolo_crops/
    │   ├── ep01/           # crops from ep01 frames
    │   └── ep02/           # crops from ep02 frames
    └── segmented/
        ├── ep01/
        │   ├── characters/
        │   └── masks/
        └── ep02/
            ├── characters/
            └── masks/

Usage:
    python run_2d_animation_pipeline.py \
        --project gumbell \
        --base-dir /mnt/data/datasets/general \
        --all-stages

    # Flat mode (no episode structure, all in one directory)
    python run_2d_animation_pipeline.py --project gumbell --all-stages --flat

Author: LLMProvider Tooling
Date: 2025-12-08
"""

import argparse
import gc
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # Paths
    base_dir: Path = Path("/mnt/data/datasets/general")
    models_dir: Path = Path("/mnt/c/ai_models")
    scripts_dir: Path = Path("/mnt/c/ai_projects/2d-animation-lora-pipeline/scripts")

    # YOLO settings
    yolo_model: str = "yolov8x.pt"
    yolo_confidence: float = 0.25  # Lower threshold for 2D animation characters
    yolo_bbox_padding: float = 0.15
    yolo_min_size: int = 64
    yolo_classes: List[int] = field(default_factory=lambda: [0])  # person class

    # ToonOut/BiRefNet settings
    toonout_model: str = "toonout/birefnet_finetuned_toonout.pth"
    toonout_input_size: int = 1024
    toonout_threshold: float = 0.5

    # Frame extraction settings
    frame_interval: int = 10  # Extract every N frames
    frame_quality: int = 95

    # Processing settings
    batch_size: int = 16
    max_retries: int = 3
    retry_delay: float = 5.0
    memory_threshold_gb: float = 2.0  # Clear GPU if free memory below this

    # Structure settings
    flat_mode: bool = False  # If True, don't create episode subdirectories

    # Device
    device: str = "cuda"


# ============================================================================
# Utility Functions
# ============================================================================

def sanitize_episode_name(video_name: str) -> str:
    """
    Create a clean, short episode identifier from video filename.

    Examples:
        "Watch The Wonderfully Weird World of Gumball 2025 full movie on Gomovies hd_2.ts" -> "ep02"
        "episode_01.mp4" -> "ep01"
        "S01E05_The_Adventure.mkv" -> "s01e05"
    """
    stem = Path(video_name).stem.lower()

    # Try to find episode/season patterns
    # Pattern: S01E05, s1e5, etc.
    match = re.search(r's(\d+)e(\d+)', stem, re.IGNORECASE)
    if match:
        return f"s{int(match.group(1)):02d}e{int(match.group(2)):02d}"

    # Pattern: Episode 5, ep5, episode_05
    match = re.search(r'(?:episode|ep)[_\s]*(\d+)', stem, re.IGNORECASE)
    if match:
        return f"ep{int(match.group(1)):02d}"

    # Pattern: ending with _2, _3 etc (like "...hd_2.ts")
    match = re.search(r'[_\-](\d+)$', stem)
    if match:
        return f"ep{int(match.group(1)):02d}"

    # Pattern: just a number at the end
    match = re.search(r'(\d+)$', stem)
    if match:
        return f"ep{int(match.group(1)):02d}"

    # Fallback: use first 20 chars of sanitized name
    clean = re.sub(r'[^a-z0-9]', '_', stem)
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean[:20] if clean else "ep00"


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(project_name: str, log_dir: Path) -> logging.Logger:
    """Setup logging with file and console handlers"""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{project_name}_{timestamp}.log"

    logger = logging.getLogger(f"pipeline_{project_name}")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ============================================================================
# Memory Management
# ============================================================================

def get_gpu_memory_info() -> Tuple[float, float]:
    """Get GPU memory info (used, free) in GB"""
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            free = total - used
            return used, free
    except:
        pass
    return 0.0, 0.0


def clear_gpu_memory():
    """Clear GPU memory"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass
    gc.collect()


def check_and_clear_memory(threshold_gb: float = 2.0, logger: Optional[logging.Logger] = None):
    """Check GPU memory and clear if below threshold"""
    used, free = get_gpu_memory_info()
    if free < threshold_gb:
        if logger:
            logger.warning(f"Low GPU memory ({free:.2f}GB free), clearing cache...")
        clear_gpu_memory()
        used, free = get_gpu_memory_info()
        if logger:
            logger.info(f"After cleanup: {free:.2f}GB free")


# ============================================================================
# Retry Decorator
# ============================================================================

def retry_with_backoff(max_retries: int = 3, base_delay: float = 5.0, logger: Optional[logging.Logger] = None):
    """Decorator for retry with exponential backoff"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    delay = base_delay * (2 ** attempt)
                    if logger:
                        logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                        logger.info(f"Retrying in {delay:.1f}s...")
                    else:
                        print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                        print(f"Retrying in {delay:.1f}s...")

                    # Clear memory before retry
                    clear_gpu_memory()
                    time.sleep(delay)

            raise last_exception
        return wrapper
    return decorator


# ============================================================================
# Stage 1: Frame Extraction (Episode-Aware)
# ============================================================================

class FrameExtractor:
    """Extract frames from videos using ffmpeg - preserves episode structure"""

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def extract_frames(self, video_path: Path, output_dir: Path) -> int:
        """Extract frames from a single video to output_dir"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command
        output_pattern = output_dir / "frame_%06d.jpg"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"select=not(mod(n\\,{self.config.frame_interval}))",
            "-vsync", "vfr",
            "-q:v", str(max(1, (100 - self.config.frame_quality) // 5 + 1)),  # Quality 1-5
            str(output_pattern)
        ]

        self.logger.info(f"Extracting frames from: {video_path.name}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode != 0:
                self.logger.error(f"ffmpeg error: {result.stderr}")
                return 0

            # Count extracted frames
            frames = list(output_dir.glob("frame_*.jpg"))
            self.logger.info(f"Extracted {len(frames)} frames from {video_path.name}")
            return len(frames)

        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout extracting frames from {video_path.name}")
            return 0
        except Exception as e:
            self.logger.error(f"Error extracting frames: {e}")
            return 0

    def process_project(self, project_dir: Path) -> Dict:
        """Extract frames from all videos in a project"""
        videos_dir = project_dir / "videos"
        frames_base_dir = project_dir / "frames"

        if not videos_dir.exists():
            self.logger.error(f"Videos directory not found: {videos_dir}")
            return {"success": False, "error": "Videos directory not found"}

        # Find all video files
        video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".ts", ".m2ts", ".webm"}
        videos = sorted([
            v for v in videos_dir.iterdir()
            if v.suffix.lower() in video_extensions
        ])

        if not videos:
            self.logger.error(f"No videos found in: {videos_dir}")
            return {"success": False, "error": "No videos found"}

        self.logger.info(f"Found {len(videos)} videos to process")
        self.logger.info(f"Episode mode: {'OFF (flat)' if self.config.flat_mode else 'ON (per-episode directories)'}")

        stats = {
            "total_videos": len(videos),
            "processed_videos": 0,
            "total_frames": 0,
            "failed_videos": [],
            "episodes": {}  # episode_id -> frame_count
        }

        for i, video in enumerate(videos, 1):
            self.logger.info(f"[{i}/{len(videos)}] Processing: {video.name}")

            if self.config.flat_mode:
                # Flat mode: all frames in one directory with video prefix
                output_dir = frames_base_dir
                episode_id = "all"
            else:
                # Episode mode: each video gets its own subdirectory
                episode_id = sanitize_episode_name(video.name)
                output_dir = frames_base_dir / episode_id
                self.logger.info(f"  -> Episode ID: {episode_id}")

            try:
                frame_count = self.extract_frames(video, output_dir)
                if frame_count > 0:
                    stats["processed_videos"] += 1
                    stats["total_frames"] += frame_count
                    stats["episodes"][episode_id] = stats["episodes"].get(episode_id, 0) + frame_count
                else:
                    stats["failed_videos"].append(video.name)
            except Exception as e:
                self.logger.error(f"Failed to process {video.name}: {e}")
                stats["failed_videos"].append(video.name)

        # Save episode mapping
        episode_map = {
            "created": datetime.now().isoformat(),
            "flat_mode": self.config.flat_mode,
            "videos": [
                {
                    "video": v.name,
                    "episode_id": sanitize_episode_name(v.name) if not self.config.flat_mode else "all"
                }
                for v in videos
            ],
            "episode_frame_counts": stats["episodes"]
        }

        with open(frames_base_dir / "episode_mapping.json", "w", encoding="utf-8") as f:
            json.dump(episode_map, f, indent=2)

        stats["success"] = len(stats["failed_videos"]) == 0
        return stats


# ============================================================================
# Stage 2: YOLO Detection (Episode-Aware)
# ============================================================================

class YOLODetector:
    """YOLO-based character detection - preserves episode structure"""

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._model = None

    @property
    def model(self):
        """Lazy load YOLO model"""
        if self._model is None:
            from ultralytics import YOLO
            model_path = self.config.models_dir / "detection" / self.config.yolo_model
            self.logger.info(f"Loading YOLO model: {model_path}")
            self._model = YOLO(str(model_path))
            self._model.to(self.config.device)
        return self._model

    def detect_and_crop(self, image: np.ndarray, frame_name: str) -> List[Tuple[np.ndarray, Dict]]:
        """Detect characters and return cropped images with metadata"""
        h, w = image.shape[:2]

        results = self.model(
            image,
            conf=self.config.yolo_confidence,
            classes=self.config.yolo_classes,
            verbose=False
        )

        if len(results) == 0 or len(results[0].boxes) == 0:
            return []

        crops = []

        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            # Calculate padding
            box_w = x2 - x1
            box_h = y2 - y1
            pad = int(self.config.yolo_bbox_padding * max(box_w, box_h))

            # Expand bbox with padding
            crop_x1 = max(0, int(x1) - pad)
            crop_y1 = max(0, int(y1) - pad)
            crop_x2 = min(w, int(x2) + pad)
            crop_y2 = min(h, int(y2) + pad)

            # Check minimum size
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1

            if crop_w < self.config.yolo_min_size or crop_h < self.config.yolo_min_size:
                continue

            # Crop image
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            metadata = {
                "crop_id": idx,
                "confidence": round(conf, 3),
                "original_bbox": [int(x1), int(y1), int(x2), int(y2)],
                "crop_region": [crop_x1, crop_y1, crop_x2, crop_y2],
                "crop_size": [crop_w, crop_h],
                "frame_name": frame_name
            }

            crops.append((crop, metadata))

        return crops

    @retry_with_backoff(max_retries=3, base_delay=5.0)
    def process_batch(self, frames: List[Path], output_dir: Path) -> List[Dict]:
        """Process a batch of frames"""
        results = []

        for frame_path in frames:
            image = cv2.imread(str(frame_path))
            if image is None:
                continue

            crops = self.detect_and_crop(image, frame_path.stem)

            for crop, metadata in crops:
                crop_filename = f"{frame_path.stem}_crop{metadata['crop_id']}.jpg"
                crop_path = output_dir / crop_filename

                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                metadata["crop_file"] = crop_filename
                results.append(metadata)

        return results

    def process_episode(self, frames_dir: Path, crops_dir: Path, episode_id: str) -> Dict:
        """Process a single episode's frames"""
        crops_dir.mkdir(parents=True, exist_ok=True)

        # Get all frames
        frame_paths = sorted([
            p for p in frames_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        if not frame_paths:
            return {"frames": 0, "crops": 0, "results": []}

        stats = {
            "frames": len(frame_paths),
            "frames_with_detections": 0,
            "crops": 0,
            "results": []
        }

        # Process in batches
        batch_size = self.config.batch_size

        for i in range(0, len(frame_paths), batch_size):
            batch = frame_paths[i:i + batch_size]

            try:
                check_and_clear_memory(self.config.memory_threshold_gb, self.logger)
                batch_results = self.process_batch(batch, crops_dir)
                stats["results"].extend(batch_results)
                stats["crops"] += len(batch_results)

                frames_in_batch = set(r["frame_name"] for r in batch_results)
                stats["frames_with_detections"] += len(frames_in_batch)

            except Exception as e:
                self.logger.error(f"Batch failed in {episode_id}: {e}")

        return stats

    def process_project(self, project_dir: Path) -> Dict:
        """Process all frames in a project (episode-aware)"""
        frames_base_dir = project_dir / "frames"
        crops_base_dir = project_dir / "yolo_crops"

        if not frames_base_dir.exists():
            self.logger.error(f"Frames directory not found: {frames_base_dir}")
            return {"success": False, "error": "Frames directory not found"}

        # Check if episode mode (has subdirectories) or flat mode
        subdirs = [d for d in frames_base_dir.iterdir() if d.is_dir()]

        if subdirs and not self.config.flat_mode:
            # Episode mode: process each episode subdirectory
            self.logger.info(f"Processing {len(subdirs)} episodes")
            episodes = sorted(subdirs, key=lambda x: x.name)
        else:
            # Flat mode: treat frames_base_dir as single "episode"
            self.logger.info("Processing in flat mode (single directory)")
            episodes = [frames_base_dir]

        stats = {
            "total_frames": 0,
            "frames_with_detections": 0,
            "total_crops": 0,
            "episodes": {},
            "all_results": []
        }

        for episode_dir in tqdm(episodes, desc="YOLO Detection (episodes)"):
            episode_id = episode_dir.name if episode_dir != frames_base_dir else "all"

            if episode_dir != frames_base_dir:
                crops_dir = crops_base_dir / episode_id
            else:
                crops_dir = crops_base_dir

            self.logger.info(f"Processing episode: {episode_id}")

            ep_stats = self.process_episode(episode_dir, crops_dir, episode_id)

            stats["total_frames"] += ep_stats["frames"]
            stats["frames_with_detections"] += ep_stats.get("frames_with_detections", 0)
            stats["total_crops"] += ep_stats["crops"]
            stats["episodes"][episode_id] = {
                "frames": ep_stats["frames"],
                "crops": ep_stats["crops"]
            }
            stats["all_results"].extend(ep_stats["results"])

            self.logger.info(f"  {episode_id}: {ep_stats['frames']} frames -> {ep_stats['crops']} crops")

        # Save detection results
        metadata = {
            "created": datetime.now().isoformat(),
            "config": {
                "model": self.config.yolo_model,
                "confidence": self.config.yolo_confidence,
                "bbox_padding": self.config.yolo_bbox_padding,
                "min_size": self.config.yolo_min_size
            },
            "stats": {
                "total_frames": stats["total_frames"],
                "frames_with_detections": stats["frames_with_detections"],
                "total_crops": stats["total_crops"]
            },
            "episodes": stats["episodes"],
            "detections": stats["all_results"]
        }

        with open(crops_base_dir / "detection_results.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        stats["success"] = True

        # Unload model to free memory
        self._model = None
        clear_gpu_memory()

        return stats


# ============================================================================
# Stage 3: ToonOut Segmentation (Episode-Aware)
# ============================================================================

class ToonOutSegmenter:
    """ToonOut/BiRefNet-based segmentation for 2D animation - preserves episode structure"""

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._model = None

        # Add BiRefNet to path
        birefnet_path = config.models_dir / "segmentation" / "BiRefNet"
        sys.path.insert(0, str(birefnet_path))

    @property
    def model(self):
        """Lazy load ToonOut/BiRefNet model"""
        if self._model is None:
            import torch
            from models.birefnet import BiRefNet
            from utils import check_state_dict

            model_path = self.config.models_dir / "segmentation" / self.config.toonout_model
            self.logger.info(f"Loading ToonOut/BiRefNet: {model_path}")

            model = BiRefNet(bb_pretrained=False)
            state_dict = torch.load(str(model_path), map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)

            model.to(self.config.device)
            model.eval()

            self._model = model
            self.logger.info(f"ToonOut/BiRefNet loaded on {self.config.device}")

        return self._model

    def segment_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Segment a single image"""
        import torch
        from torch.cuda import amp

        h, w = image.shape[:2]
        input_size = self.config.toonout_input_size

        # Preprocess: resize maintaining aspect ratio
        scale = input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_h = (input_size - new_h) // 2
        pad_w = (input_size - new_w) // 2

        img_padded = np.zeros((input_size, input_size, 3), dtype=np.float32)
        img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized

        # Normalize and convert to tensor
        img_tensor = img_padded / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))  # HWC -> CHW
        img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0).to(self.config.device)

        # Inference
        with torch.no_grad(), amp.autocast(device_type='cuda'):
            pred = self.model(img_tensor)[-1].sigmoid()

        # Postprocess
        pred = pred.squeeze().cpu().numpy()

        # Remove padding
        pred = pred[pad_h:pad_h+new_h, pad_w:pad_w+new_w]

        # Resize back to original size
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

        # Binarize
        confidence = float(pred.mean())
        binary_mask = (pred > self.config.toonout_threshold).astype(np.uint8) * 255

        return binary_mask, confidence

    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to create RGBA image with transparent background"""
        rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = mask
        return rgba

    @retry_with_backoff(max_retries=3, base_delay=5.0)
    def process_single(self, image_path: Path, output_dir: Path, masks_dir: Path) -> Optional[Dict]:
        """Process a single image"""
        from PIL import Image

        image = cv2.imread(str(image_path))
        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Segment
        mask, confidence = self.segment_image(image_rgb)

        # Calculate mask ratio
        mask_area = np.sum(mask > 127)
        mask_ratio = mask_area / (h * w)

        # Apply mask
        result = self.apply_mask(image_rgb, mask)

        # Save outputs
        basename = image_path.stem

        # Save character image (RGBA PNG)
        char_filename = f"{basename}.png"
        char_path = output_dir / char_filename
        Image.fromarray(result).save(char_path)

        # Save mask
        mask_filename = f"{basename}_mask.png"
        mask_path = masks_dir / mask_filename
        cv2.imwrite(str(mask_path), mask)

        return {
            "input_file": image_path.name,
            "output_file": char_filename,
            "mask_file": mask_filename,
            "mask_area": int(mask_area),
            "mask_ratio": round(mask_ratio, 4),
            "confidence": round(confidence, 4)
        }

    def process_episode(self, crops_dir: Path, output_dir: Path, masks_dir: Path, episode_id: str) -> Dict:
        """Process a single episode's crops"""
        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Get all crop images
        image_paths = sorted([
            p for p in crops_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith(".")
        ])

        if not image_paths:
            return {"total": 0, "successful": 0, "failed": 0, "results": []}

        stats = {
            "total": len(image_paths),
            "successful": 0,
            "failed": 0,
            "results": []
        }

        for image_path in image_paths:
            try:
                if stats["successful"] % 50 == 0:
                    check_and_clear_memory(self.config.memory_threshold_gb, self.logger)

                result = self.process_single(image_path, output_dir, masks_dir)

                if result:
                    stats["results"].append(result)
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1

            except Exception as e:
                self.logger.error(f"Failed to process {image_path.name}: {e}")
                stats["failed"] += 1

        return stats

    def process_project(self, project_dir: Path) -> Dict:
        """Process all YOLO crops in a project (episode-aware)"""
        crops_base_dir = project_dir / "yolo_crops"
        segmented_base_dir = project_dir / "segmented"

        if not crops_base_dir.exists():
            self.logger.error(f"YOLO crops directory not found: {crops_base_dir}")
            return {"success": False, "error": "YOLO crops directory not found"}

        # Check if episode mode (has subdirectories) or flat mode
        subdirs = [d for d in crops_base_dir.iterdir() if d.is_dir()]

        if subdirs and not self.config.flat_mode:
            # Episode mode
            self.logger.info(f"Processing {len(subdirs)} episodes")
            episodes = sorted(subdirs, key=lambda x: x.name)
        else:
            # Flat mode
            self.logger.info("Processing in flat mode")
            episodes = [crops_base_dir]

        stats = {
            "total_images": 0,
            "successful": 0,
            "failed": 0,
            "episodes": {},
            "all_results": []
        }

        for episode_dir in tqdm(episodes, desc="ToonOut Segmentation (episodes)"):
            episode_id = episode_dir.name if episode_dir != crops_base_dir else "all"

            if episode_dir != crops_base_dir:
                output_dir = segmented_base_dir / episode_id / "characters"
                masks_dir = segmented_base_dir / episode_id / "masks"
            else:
                output_dir = segmented_base_dir / "characters"
                masks_dir = segmented_base_dir / "masks"

            self.logger.info(f"Processing episode: {episode_id}")

            ep_stats = self.process_episode(episode_dir, output_dir, masks_dir, episode_id)

            stats["total_images"] += ep_stats["total"]
            stats["successful"] += ep_stats["successful"]
            stats["failed"] += ep_stats["failed"]
            stats["episodes"][episode_id] = {
                "total": ep_stats["total"],
                "successful": ep_stats["successful"],
                "failed": ep_stats["failed"]
            }
            stats["all_results"].extend(ep_stats["results"])

            self.logger.info(f"  {episode_id}: {ep_stats['successful']}/{ep_stats['total']} successful")

        # Save metadata
        metadata = {
            "created": datetime.now().isoformat(),
            "source_dir": str(crops_base_dir),
            "config": {
                "model": self.config.toonout_model,
                "input_size": self.config.toonout_input_size,
                "threshold": self.config.toonout_threshold
            },
            "stats": {
                "total_images": stats["total_images"],
                "successful": stats["successful"],
                "failed": stats["failed"]
            },
            "episodes": stats["episodes"],
            "results": stats["all_results"]
        }

        with open(segmented_base_dir / "segmentation_results.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        stats["success"] = stats["failed"] == 0

        # Unload model to free memory
        self._model = None
        clear_gpu_memory()

        return stats


# ============================================================================
# Pipeline Orchestrator
# ============================================================================

class Pipeline2DAnimation:
    """Complete 2D animation pipeline orchestrator"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = None

    def run(
        self,
        project_name: str,
        extract_frames: bool = False,
        yolo_detect: bool = False,
        toonout_segment: bool = False,
        all_stages: bool = False
    ) -> Dict:
        """Run the pipeline"""
        project_dir = self.config.base_dir / project_name

        if not project_dir.exists():
            raise ValueError(f"Project directory not found: {project_dir}")

        # Setup logging
        log_dir = project_dir / "logs"
        self.logger = setup_logging(project_name, log_dir)

        self.logger.info("=" * 60)
        self.logger.info(f"2D ANIMATION PIPELINE - {project_name.upper()}")
        self.logger.info("=" * 60)
        self.logger.info(f"Project: {project_dir}")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"Episode Mode: {'OFF (flat)' if self.config.flat_mode else 'ON'}")
        self.logger.info("=" * 60)

        results = {
            "project": project_name,
            "started": datetime.now().isoformat(),
            "flat_mode": self.config.flat_mode,
            "stages": {}
        }

        try:
            # Stage 1: Frame Extraction
            if all_stages or extract_frames:
                self.logger.info("\n" + "=" * 60)
                self.logger.info("STAGE 1: FRAME EXTRACTION")
                self.logger.info("=" * 60)

                extractor = FrameExtractor(self.config, self.logger)
                stage_result = extractor.process_project(project_dir)
                results["stages"]["frame_extraction"] = stage_result

                self.logger.info(f"Frame extraction: {stage_result.get('total_frames', 0)} frames extracted")
                if "episodes" in stage_result:
                    for ep_id, count in stage_result["episodes"].items():
                        self.logger.info(f"  {ep_id}: {count} frames")

                if not stage_result.get("success"):
                    self.logger.warning("Frame extraction had issues, continuing...")

            # Stage 2: YOLO Detection
            if all_stages or yolo_detect:
                self.logger.info("\n" + "=" * 60)
                self.logger.info("STAGE 2: YOLO CHARACTER DETECTION")
                self.logger.info("=" * 60)

                detector = YOLODetector(self.config, self.logger)
                stage_result = detector.process_project(project_dir)
                results["stages"]["yolo_detection"] = stage_result

                self.logger.info(f"YOLO detection: {stage_result.get('total_crops', 0)} crops saved")

            # Stage 3: ToonOut Segmentation
            if all_stages or toonout_segment:
                self.logger.info("\n" + "=" * 60)
                self.logger.info("STAGE 3: TOONOUT SEGMENTATION")
                self.logger.info("=" * 60)

                segmenter = ToonOutSegmenter(self.config, self.logger)
                stage_result = segmenter.process_project(project_dir)
                results["stages"]["toonout_segmentation"] = stage_result

                self.logger.info(f"ToonOut: {stage_result.get('successful', 0)} characters segmented")

            results["success"] = True
            results["completed"] = datetime.now().isoformat()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.error(traceback.format_exc())
            results["success"] = False
            results["error"] = str(e)

        # Save pipeline results
        results_path = project_dir / "pipeline_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Results saved to: {results_path}")

        return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="2D Animation Character Pipeline (Episode-Aware)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete pipeline (episode mode - default)
    python run_2d_animation_pipeline.py --project gumbell --all-stages

    # Run in flat mode (no episode subdirectories)
    python run_2d_animation_pipeline.py --project gumbell --all-stages --flat

    # Run individual stages
    python run_2d_animation_pipeline.py --project gumbell --extract-frames
    python run_2d_animation_pipeline.py --project gumbell --yolo-detect
    python run_2d_animation_pipeline.py --project gumbell --toonout-segment

Directory Structure (Episode Mode):
    project/
    ├── videos/
    │   ├── episode_01.mp4
    │   └── episode_02.mp4
    ├── frames/
    │   ├── ep01/           # frames from episode_01
    │   └── ep02/           # frames from episode_02
    ├── yolo_crops/
    │   ├── ep01/           # crops from ep01 frames
    │   └── ep02/           # crops from ep02 frames
    └── segmented/
        ├── ep01/
        │   ├── characters/
        │   └── masks/
        └── ep02/
            ├── characters/
            └── masks/
        """
    )

    parser.add_argument("--project", "-p", required=True,
                        help="Project name (subdirectory under base-dir)")
    parser.add_argument("--base-dir", "-b", type=Path,
                        default=Path("/mnt/data/datasets/general"),
                        help="Base directory containing projects")

    # Stage selection
    parser.add_argument("--all-stages", "-a", action="store_true",
                        help="Run all pipeline stages")
    parser.add_argument("--extract-frames", action="store_true",
                        help="Run frame extraction only")
    parser.add_argument("--yolo-detect", action="store_true",
                        help="Run YOLO detection only")
    parser.add_argument("--toonout-segment", action="store_true",
                        help="Run ToonOut segmentation only")

    # Structure options
    parser.add_argument("--flat", action="store_true",
                        help="Flat mode: don't create episode subdirectories")

    # Processing options
    parser.add_argument("--frame-interval", type=int, default=10,
                        help="Extract every N frames (default: 10)")
    parser.add_argument("--yolo-confidence", type=float, default=0.25,
                        help="YOLO confidence threshold for 2D animation (default: 0.25)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for processing (default: 16)")
    parser.add_argument("--device", "-d", default="cuda",
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Check that at least one stage is selected
    if not (args.all_stages or args.extract_frames or args.yolo_detect or args.toonout_segment):
        parser.error("Must specify --all-stages or at least one stage flag")

    # Create config
    config = PipelineConfig(
        base_dir=args.base_dir,
        device=args.device,
        frame_interval=args.frame_interval,
        yolo_confidence=args.yolo_confidence,
        batch_size=args.batch_size,
        flat_mode=args.flat
    )

    # Run pipeline
    pipeline = Pipeline2DAnimation(config)
    results = pipeline.run(
        project_name=args.project,
        extract_frames=args.extract_frames,
        yolo_detect=args.yolo_detect,
        toonout_segment=args.toonout_segment,
        all_stages=args.all_stages
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Episode Mode: {'OFF (flat)' if config.flat_mode else 'ON'}")
    print()

    for stage_name, stage_result in results.get("stages", {}).items():
        status = "✓" if stage_result.get("success") else "✗"
        print(f"{status} {stage_name}")

        if stage_name == "frame_extraction":
            print(f"  Total frames: {stage_result.get('total_frames', 0)}")
            if "episodes" in stage_result:
                for ep_id, count in stage_result["episodes"].items():
                    print(f"    {ep_id}: {count} frames")
        elif stage_name == "yolo_detection":
            print(f"  Total crops: {stage_result.get('total_crops', 0)}")
            if "episodes" in stage_result:
                for ep_id, ep_stats in stage_result["episodes"].items():
                    print(f"    {ep_id}: {ep_stats['crops']} crops")
        elif stage_name == "toonout_segmentation":
            print(f"  Successful: {stage_result.get('successful', 0)}")
            print(f"  Failed: {stage_result.get('failed', 0)}")
            if "episodes" in stage_result:
                for ep_id, ep_stats in stage_result["episodes"].items():
                    print(f"    {ep_id}: {ep_stats['successful']}/{ep_stats['total']}")

    print("=" * 60)

    return 0 if results.get("success") else 1


if __name__ == "__main__":
    exit(main())
