#!/usr/bin/env python3
"""
Fast Character LoRA Pipeline
=============================

Unified pipeline for preparing character LoRA training data from frames.
Uses YOLO + MobileSAM for fast, high-quality character segmentation.

Pipeline Flow:
    Input Frames → YOLO Detection → Bbox + Padding → MobileSAM Segmentation
    → Background Processing → Resize to 1024×1024 → Quality Filtering
    → Caption Generation (OpenAI GPT-4o-mini) → Kohya Format Output

Usage:
    python fast_character_lora_pipeline.py \
        --input-dir /path/to/character_frames \
        --output-dir /path/to/training_data \
        --character-name "luca" \
        --caption-engine openai \
        --target-size 1024 \
        --background solid-gray

Author: LLMProvider Tooling
Date: 2025-12-07
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    input_dir: Path
    output_dir: Path
    character_name: str
    target_size: int = 1024
    background_mode: str = "solid-gray"  # transparent, solid-gray, solid-white, solid-black, blur
    background_color: Tuple[int, int, int] = (128, 128, 128)
    bbox_padding: float = 0.15
    yolo_model: str = "yolov11n.pt"
    yolo_confidence: float = 0.5
    caption_engine: str = "openai"  # openai, llm_provider, template
    caption_model: str = "gpt-4o-mini"
    blur_threshold: float = 80.0
    min_size: int = 128
    dedup_threshold: int = 8
    num_workers: int = 4
    device: str = "cuda"
    repeats: int = 10  # Kohya repeats


class FastCharacterLoraPipeline:
    """
    Fast pipeline for character LoRA training data preparation.

    Integrates:
    - YOLOv11 for character detection
    - MobileSAM for fine segmentation
    - Quality filters (blur, size, dedup)
    - OpenAI/LLMProvider API for captioning
    - Kohya format output
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stats = {
            "total_frames": 0,
            "detected_characters": 0,
            "passed_quality": 0,
            "captioned": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None
        }

        # Lazy-loaded components
        self._yolo_detector = None
        self._sam_segmenter = None
        self._caption_engine = None
        self._blur_filter = None
        self._deduplicator = None

        # Create output directories
        self._setup_output_dirs()

    def _setup_output_dirs(self):
        """Create output directory structure"""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Kohya format: {repeats}_{character_name}
        self.kohya_dir = self.config.output_dir / f"{self.config.repeats}_{self.config.character_name}"
        self.kohya_dir.mkdir(parents=True, exist_ok=True)

        # Intermediate directories
        self.segments_dir = self.config.output_dir / "segments"
        self.segments_dir.mkdir(parents=True, exist_ok=True)

    @property
    def yolo_detector(self):
        """Lazy load YOLO detector"""
        if self._yolo_detector is None:
            from generic.segmentation.yolo_sam_segmentation import YOLODetector
            self._yolo_detector = YOLODetector(
                model_path=self.config.yolo_model,
                device=self.config.device,
                confidence_threshold=self.config.yolo_confidence,
                classes=[0]  # person class only
            )
        return self._yolo_detector

    @property
    def sam_segmenter(self):
        """Lazy load MobileSAM segmenter"""
        if self._sam_segmenter is None:
            from generic.segmentation.yolo_sam_segmentation import MobileSAMSegmenter
            self._sam_segmenter = MobileSAMSegmenter(device=self.config.device)
        return self._sam_segmenter

    @property
    def caption_engine(self):
        """Lazy load caption engine"""
        if self._caption_engine is None:
            if self.config.caption_engine == "openai":
                from generic.training.caption_engines.openai_api_engine import OpenAIAPICaptionEngine
                self._caption_engine = OpenAIAPICaptionEngine({
                    "model_name": self.config.caption_model,
                    "max_tokens": 300,
                    "schema_mode": False
                })
            elif self.config.caption_engine == "llm_provider":
                from generic.training.caption_engines.llm_provider_api_engine import LLMProviderAPICaptionEngine
                self._caption_engine = LLMProviderAPICaptionEngine({
                    "model_name": "sonnet-3.5",
                    "max_tokens": 512,
                    "schema_mode": False
                })
            else:
                from generic.training.caption_engines.template_engine import TemplateCaptionEngine
                self._caption_engine = TemplateCaptionEngine({
                    "prefix": "a 3d animated character, pixar style"
                })
        return self._caption_engine

    @property
    def blur_filter(self):
        """Lazy load blur filter"""
        if self._blur_filter is None:
            from generic.training.quality_filters.blur_filter import BlurFilter
            self._blur_filter = BlurFilter({
                "threshold": self.config.blur_threshold
            })
        return self._blur_filter

    @property
    def deduplicator(self):
        """Lazy load perceptual hash deduplicator"""
        if self._deduplicator is None:
            from generic.training.quality_filters.perceptual_hash_deduplicator import PerceptualHashDeduplicator
            self._deduplicator = PerceptualHashDeduplicator({
                "threshold": self.config.dedup_threshold
            })
        return self._deduplicator

    def process_frame(self, frame_path: Path) -> List[Dict]:
        """
        Process a single frame through the pipeline.

        Args:
            frame_path: Path to input frame

        Returns:
            List of processed character data dicts
        """
        results = []

        try:
            # Load frame
            image = cv2.imread(str(frame_path))
            if image is None:
                return results

            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # YOLO detection
            bboxes = self.yolo_detector.detect(image)

            if not bboxes:
                return results

            # Process each detected character
            for idx, bbox in enumerate(bboxes):
                # Expand bbox with padding
                expanded_bbox = bbox.expand(self.config.bbox_padding, w, h)

                # MobileSAM segmentation
                mask = self.sam_segmenter.segment(image_rgb, expanded_bbox)

                # Apply background processing
                processed_image = self._apply_background(image_rgb, mask)

                # Crop to bbox region
                cropped = self._crop_to_content(processed_image, mask)

                if cropped is None:
                    continue

                # Resize to target size
                resized = self._resize_with_letterbox(cropped, self.config.target_size)

                # Quality check - blur
                if not self._check_blur(resized):
                    continue

                # Quality check - size
                if not self._check_size(cropped):
                    continue

                results.append({
                    "image": resized,
                    "source_frame": frame_path.name,
                    "bbox_idx": idx,
                    "bbox": (expanded_bbox.x1, expanded_bbox.y1, expanded_bbox.x2, expanded_bbox.y2),
                    "confidence": bbox.confidence
                })

        except Exception as e:
            print(f"Error processing {frame_path}: {e}")

        return results

    def _apply_background(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply background processing based on mode"""
        if self.config.background_mode == "transparent":
            # Return RGBA with alpha channel
            rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = image
            rgba[:, :, 3] = mask
            return rgba

        elif self.config.background_mode.startswith("solid"):
            # Solid color background
            if self.config.background_mode == "solid-gray":
                bg_color = (128, 128, 128)
            elif self.config.background_mode == "solid-white":
                bg_color = (255, 255, 255)
            elif self.config.background_mode == "solid-black":
                bg_color = (0, 0, 0)
            else:
                bg_color = self.config.background_color

            # Create background
            background = np.full_like(image, bg_color, dtype=np.uint8)

            # Composite
            mask_3ch = np.stack([mask / 255.0] * 3, axis=-1)
            result = (image * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)
            return result

        elif self.config.background_mode == "blur":
            # Blurred background
            blurred = cv2.GaussianBlur(image, (51, 51), 0)
            mask_3ch = np.stack([mask / 255.0] * 3, axis=-1)
            result = (image * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
            return result

        return image

    def _crop_to_content(self, image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """Crop image to mask content bounds with padding"""
        # Find bounding box of mask
        coords = np.column_stack(np.where(mask > 127))
        if len(coords) == 0:
            return None

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Add small padding
        pad = 10
        h, w = mask.shape[:2]
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        y_max = min(h, y_max + pad)
        x_max = min(w, x_max + pad)

        return image[y_min:y_max, x_min:x_max]

    def _resize_with_letterbox(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image with letterboxing to target size"""
        h, w = image.shape[:2]
        scale = target_size / max(h, w)

        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Create canvas
        if len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA
            canvas = np.zeros((target_size, target_size, 4), dtype=np.uint8)
        else:
            # RGB with gray background
            canvas = np.full((target_size, target_size, 3), 128, dtype=np.uint8)

        # Center paste
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas

    def _check_blur(self, image: np.ndarray) -> bool:
        """Check if image passes blur filter"""
        try:
            # Convert to grayscale for blur detection
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
                else:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()

            return variance >= self.config.blur_threshold
        except:
            return True

    def _check_size(self, image: np.ndarray) -> bool:
        """Check if image meets minimum size requirements"""
        h, w = image.shape[:2]
        return min(h, w) >= self.config.min_size

    def run(self):
        """Run the full pipeline"""
        print("=" * 70)
        print("🚀 FAST CHARACTER LORA PIPELINE")
        print("=" * 70)
        print(f"Input:        {self.config.input_dir}")
        print(f"Output:       {self.config.output_dir}")
        print(f"Character:    {self.config.character_name}")
        print(f"Target size:  {self.config.target_size}×{self.config.target_size}")
        print(f"Background:   {self.config.background_mode}")
        print(f"Caption:      {self.config.caption_engine} ({self.config.caption_model})")
        print("=" * 70)

        self.stats["start_time"] = datetime.now()

        # Get all input frames
        frame_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        frame_paths = sorted([
            p for p in self.config.input_dir.iterdir()
            if p.suffix.lower() in frame_extensions
        ])

        self.stats["total_frames"] = len(frame_paths)
        print(f"\n📁 Found {len(frame_paths)} input frames")

        # Process frames
        all_results = []
        for frame_path in tqdm(frame_paths, desc="Processing frames"):
            results = self.process_frame(frame_path)
            all_results.extend(results)
            self.stats["detected_characters"] += len(results)

        print(f"\n✓ Detected {len(all_results)} character instances")

        # Deduplication
        print("\n🔍 Deduplicating...")
        unique_results = self._deduplicate(all_results)
        self.stats["passed_quality"] = len(unique_results)
        print(f"✓ {len(unique_results)} unique images after deduplication")

        # Save images
        print("\n💾 Saving images...")
        saved_paths = []
        for idx, result in enumerate(tqdm(unique_results, desc="Saving")):
            filename = f"{self.config.character_name}_{idx:05d}.png"
            save_path = self.kohya_dir / filename

            # Convert to PIL and save
            if result["image"].shape[2] == 4:
                pil_img = Image.fromarray(result["image"], mode="RGBA")
            else:
                pil_img = Image.fromarray(result["image"], mode="RGB")

            pil_img.save(save_path)
            saved_paths.append(save_path)
            result["saved_path"] = save_path

        # Generate captions
        if self.config.caption_engine != "skip":
            print(f"\n📝 Generating captions with {self.config.caption_engine}...")
            self._generate_captions(unique_results)

        # Save metadata
        self._save_metadata(unique_results)

        self.stats["end_time"] = datetime.now()

        # Print summary
        self._print_summary()

        return unique_results

    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate images using perceptual hashing"""
        if len(results) <= 1:
            return results

        try:
            import imagehash

            seen_hashes = {}
            unique_results = []

            for result in results:
                # Convert to PIL for hashing
                if result["image"].shape[2] == 4:
                    pil_img = Image.fromarray(result["image"][:, :, :3])
                else:
                    pil_img = Image.fromarray(result["image"])

                # Compute perceptual hash
                phash = imagehash.phash(pil_img)

                # Check for duplicates
                is_dup = False
                for existing_hash in seen_hashes.keys():
                    if phash - existing_hash < self.config.dedup_threshold:
                        is_dup = True
                        break

                if not is_dup:
                    seen_hashes[phash] = result
                    unique_results.append(result)

            return unique_results

        except ImportError:
            print("⚠️ imagehash not installed, skipping deduplication")
            return results

    def _generate_captions(self, results: List[Dict]):
        """Generate captions for all images"""
        for result in tqdm(results, desc="Captioning"):
            try:
                # Generate caption
                caption = self.caption_engine.generate_single(result["saved_path"])

                # Clean up caption
                caption = self._clean_caption(caption)

                # Save caption file
                caption_path = result["saved_path"].with_suffix(".txt")
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(caption)

                result["caption"] = caption
                self.stats["captioned"] += 1

            except Exception as e:
                print(f"⚠️ Caption failed for {result['saved_path']}: {e}")
                # Use default caption
                default_caption = f"a 3d animated character, pixar style, smooth shading, studio lighting"
                caption_path = result["saved_path"].with_suffix(".txt")
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(default_caption)
                result["caption"] = default_caption

    def _clean_caption(self, caption: str) -> str:
        """Clean and format caption for training"""
        # Remove JSON if present
        if caption.strip().startswith("{"):
            try:
                data = json.loads(caption)
                if "final_caption" in data:
                    caption = data["final_caption"]
            except:
                pass

        # Clean whitespace
        caption = " ".join(caption.split())

        # Ensure reasonable length
        words = caption.split()
        if len(words) > 100:
            caption = " ".join(words[:100])

        return caption

    def _save_metadata(self, results: List[Dict]):
        """Save pipeline metadata"""
        metadata = {
            "pipeline": "fast_character_lora_pipeline",
            "version": "1.0.0",
            "character_name": self.config.character_name,
            "config": {
                "target_size": self.config.target_size,
                "background_mode": self.config.background_mode,
                "bbox_padding": self.config.bbox_padding,
                "caption_engine": self.config.caption_engine,
                "caption_model": self.config.caption_model,
                "blur_threshold": self.config.blur_threshold,
                "repeats": self.config.repeats
            },
            "stats": {
                "total_frames": self.stats["total_frames"],
                "detected_characters": self.stats["detected_characters"],
                "output_images": len(results),
                "captioned": self.stats["captioned"],
                "start_time": self.stats["start_time"].isoformat() if self.stats["start_time"] else None,
                "end_time": self.stats["end_time"].isoformat() if self.stats["end_time"] else None
            },
            "images": [
                {
                    "filename": r["saved_path"].name,
                    "source_frame": r["source_frame"],
                    "caption": r.get("caption", "")
                }
                for r in results
            ]
        }

        metadata_path = self.config.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _print_summary(self):
        """Print pipeline summary"""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        print("\n" + "=" * 70)
        print("📊 PIPELINE SUMMARY")
        print("=" * 70)
        print(f"Total frames processed:     {self.stats['total_frames']}")
        print(f"Characters detected:        {self.stats['detected_characters']}")
        print(f"Unique images (after dedup): {self.stats['passed_quality']}")
        print(f"Captions generated:         {self.stats['captioned']}")
        print(f"Duration:                   {duration:.1f} seconds")
        print(f"\nOutput directory: {self.kohya_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Fast Character LoRA Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python fast_character_lora_pipeline.py \\
        --input-dir /path/to/frames \\
        --output-dir /path/to/output \\
        --character-name luca

    # With OpenAI captioning
    python fast_character_lora_pipeline.py \\
        --input-dir /path/to/frames \\
        --output-dir /path/to/output \\
        --character-name luca \\
        --caption-engine openai \\
        --caption-model gpt-4o-mini

    # Skip captioning (use template)
    python fast_character_lora_pipeline.py \\
        --input-dir /path/to/frames \\
        --output-dir /path/to/output \\
        --character-name luca \\
        --caption-engine template
        """
    )

    # Required arguments
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory containing input frames")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Output directory for training data")
    parser.add_argument("--character-name", required=True,
                        help="Character name for output files")

    # Processing options
    parser.add_argument("--target-size", type=int, default=1024,
                        help="Target image size (default: 1024)")
    parser.add_argument("--background", default="solid-gray",
                        choices=["transparent", "solid-gray", "solid-white", "solid-black", "blur"],
                        help="Background processing mode (default: solid-gray)")
    parser.add_argument("--bbox-padding", type=float, default=0.15,
                        help="Bounding box padding ratio (default: 0.15)")

    # YOLO options
    parser.add_argument("--yolo-model", default="yolov11n.pt",
                        help="YOLO model path (default: yolov11n.pt)")
    parser.add_argument("--yolo-confidence", type=float, default=0.5,
                        help="YOLO confidence threshold (default: 0.5)")

    # Caption options
    parser.add_argument("--caption-engine", default="openai",
                        choices=["openai", "llm_provider", "template", "skip"],
                        help="Caption engine (default: openai)")
    parser.add_argument("--caption-model", default="gpt-4o-mini",
                        help="Caption model (default: gpt-4o-mini)")

    # Quality options
    parser.add_argument("--blur-threshold", type=float, default=80.0,
                        help="Blur detection threshold (default: 80.0)")
    parser.add_argument("--min-size", type=int, default=128,
                        help="Minimum character size (default: 128)")
    parser.add_argument("--dedup-threshold", type=int, default=8,
                        help="Perceptual hash dedup threshold (default: 8)")

    # Training options
    parser.add_argument("--repeats", type=int, default=10,
                        help="Kohya repeats (default: 10)")

    # Hardware options
    parser.add_argument("--device", default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers (default: 4)")

    args = parser.parse_args()

    # Create config
    config = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        character_name=args.character_name,
        target_size=args.target_size,
        background_mode=args.background,
        bbox_padding=args.bbox_padding,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        caption_engine=args.caption_engine,
        caption_model=args.caption_model,
        blur_threshold=args.blur_threshold,
        min_size=args.min_size,
        dedup_threshold=args.dedup_threshold,
        repeats=args.repeats,
        device=args.device,
        num_workers=args.workers
    )

    # Run pipeline
    pipeline = FastCharacterLoraPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
