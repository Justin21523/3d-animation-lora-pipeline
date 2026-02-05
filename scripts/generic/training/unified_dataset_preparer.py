#!/usr/bin/env python3
"""
Unified Dataset Preparer for LoRA Training.

Prepares training datasets from clustered character images with:
- Quality filtering
- Caption generation/validation
- Kohya-compatible dataset structure
- Dataset augmentation options
- Multi-format export (SD1.5, SDXL)

AI_WAREHOUSE 3.0 compliant paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random

logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    """Information about a single training image."""
    path: Path
    width: int
    height: int
    file_size: int
    caption: Optional[str] = None
    quality_score: float = 1.0
    hash: Optional[str] = None


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    # Source
    source_dirs: List[Path]
    character_name: str

    # Output
    output_dir: Path
    base_model: str = "sdxl"  # "sd15" or "sdxl"

    # Quality filtering
    min_resolution: int = 512
    max_resolution: int = 2048
    min_quality_score: float = 0.5
    remove_duplicates: bool = True
    duplicate_threshold: float = 0.95

    # Caption
    generate_captions: bool = True
    caption_prefix: str = ""
    caption_suffix: str = ""
    max_caption_length: int = 77
    use_trigger_word: bool = True
    trigger_word: Optional[str] = None

    # Dataset size
    target_size: Optional[int] = None  # None = use all
    min_size: int = 20
    max_size: int = 1000

    # Augmentation
    enable_augmentation: bool = False
    flip_horizontal: bool = False  # Usually False for 2D animation
    color_jitter: bool = False  # Usually False for 2D animation

    # Format
    caption_extension: str = ".txt"
    copy_images: bool = True  # False = symlinks

    @property
    def effective_trigger(self) -> str:
        if self.trigger_word:
            return self.trigger_word
        # Auto-generate from character name
        return self.character_name.lower().replace(" ", "_")


@dataclass
class DatasetStats:
    """Statistics about prepared dataset."""
    character_name: str
    total_source_images: int
    images_after_quality_filter: int
    images_after_dedup: int
    final_image_count: int
    avg_resolution: Tuple[int, int]
    avg_quality_score: float
    caption_stats: Dict
    output_dir: str
    base_model: str

    def to_dict(self) -> Dict:
        return asdict(self)


class UnifiedDatasetPreparer:
    """
    Complete dataset preparation for LoRA training.

    Workflow:
    1. Collect images from source directories
    2. Filter by quality (resolution, blur, etc.)
    3. Remove duplicates (perceptual hash)
    4. Generate/validate captions
    5. Sample to target size
    6. Export in Kohya-compatible format
    """

    # AI_WAREHOUSE 3.0 paths
    DEFAULT_OUTPUT_ROOT = Path("/mnt/data/training/lora")
    CAPTION_MODEL_DEFAULT = "qwen2_vl"

    def __init__(
        self,
        vlm_captioner=None,  # Optional VLMCaptioner instance
        quality_scorer=None,  # Optional ImageQualityScorer instance
        num_workers: int = 8,
    ):
        self.vlm_captioner = vlm_captioner
        self.quality_scorer = quality_scorer
        self.num_workers = num_workers

    def prepare_character_dataset(
        self,
        config: DatasetConfig,
    ) -> DatasetStats:
        """
        Prepare a complete character dataset for LoRA training.

        Args:
            config: DatasetConfig with all settings

        Returns:
            DatasetStats with preparation results
        """
        logger.info(f"Preparing dataset for character: {config.character_name}")

        # Step 1: Collect images
        images = self._collect_images(config.source_dirs)
        total_source = len(images)
        logger.info(f"Collected {total_source} source images")

        if total_source < config.min_size:
            logger.warning(
                f"Only {total_source} images found, minimum is {config.min_size}"
            )

        # Step 2: Quality filtering
        images = self._filter_by_quality(images, config)
        after_quality = len(images)
        logger.info(f"After quality filter: {after_quality} images")

        # Step 3: Duplicate removal
        if config.remove_duplicates:
            images = self._remove_duplicates(images, config.duplicate_threshold)
            after_dedup = len(images)
            logger.info(f"After dedup: {after_dedup} images")
        else:
            after_dedup = after_quality

        # Step 4: Sample to target size
        if config.target_size and len(images) > config.target_size:
            images = self._sample_diverse(images, config.target_size)
            logger.info(f"Sampled to {len(images)} images")

        # Enforce max size
        if len(images) > config.max_size:
            images = images[:config.max_size]

        # Step 5: Generate/validate captions
        if config.generate_captions:
            images = self._generate_captions(images, config)

        # Step 6: Export dataset
        self._export_dataset(images, config)

        # Calculate stats
        widths = [img.width for img in images]
        heights = [img.height for img in images]
        avg_res = (
            sum(widths) // len(widths) if widths else 0,
            sum(heights) // len(heights) if heights else 0,
        )

        caption_lengths = [
            len(img.caption.split()) if img.caption else 0
            for img in images
        ]

        stats = DatasetStats(
            character_name=config.character_name,
            total_source_images=total_source,
            images_after_quality_filter=after_quality,
            images_after_dedup=after_dedup,
            final_image_count=len(images),
            avg_resolution=avg_res,
            avg_quality_score=sum(img.quality_score for img in images) / len(images) if images else 0,
            caption_stats={
                "avg_length": sum(caption_lengths) / len(caption_lengths) if caption_lengths else 0,
                "min_length": min(caption_lengths) if caption_lengths else 0,
                "max_length": max(caption_lengths) if caption_lengths else 0,
                "has_trigger": sum(1 for img in images if img.caption and config.effective_trigger in img.caption.lower()),
            },
            output_dir=str(config.output_dir),
            base_model=config.base_model,
        )

        # Save metadata
        self._save_metadata(stats, config)

        logger.info(f"Dataset prepared: {stats.final_image_count} images at {config.output_dir}")
        return stats

    def _collect_images(
        self,
        source_dirs: List[Path],
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    ) -> List[ImageInfo]:
        """Collect all images from source directories."""
        from PIL import Image

        images = []
        seen_paths = set()

        for source_dir in source_dirs:
            source_dir = Path(source_dir)
            if not source_dir.exists():
                logger.warning(f"Source directory not found: {source_dir}")
                continue

            for ext in extensions:
                for img_path in source_dir.glob(f"*{ext}"):
                    if img_path in seen_paths:
                        continue
                    seen_paths.add(img_path)

                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                        images.append(ImageInfo(
                            path=img_path,
                            width=w,
                            height=h,
                            file_size=img_path.stat().st_size,
                        ))
                    except Exception as e:
                        logger.warning(f"Could not read {img_path}: {e}")

        return images

    def _filter_by_quality(
        self,
        images: List[ImageInfo],
        config: DatasetConfig,
    ) -> List[ImageInfo]:
        """Filter images by quality criteria."""
        filtered = []

        for img in images:
            # Resolution check
            min_dim = min(img.width, img.height)
            max_dim = max(img.width, img.height)

            if min_dim < config.min_resolution:
                continue
            if max_dim > config.max_resolution:
                continue

            # Quality score (use scorer if available)
            if self.quality_scorer:
                try:
                    score = self.quality_scorer.score_image(img.path)
                    img.quality_score = score
                except Exception as e:
                    logger.warning(f"Quality scoring failed for {img.path}: {e}")
                    img.quality_score = 0.7  # Default

            if img.quality_score < config.min_quality_score:
                continue

            filtered.append(img)

        return filtered

    def _remove_duplicates(
        self,
        images: List[ImageInfo],
        threshold: float = 0.95,
    ) -> List[ImageInfo]:
        """Remove duplicate images using perceptual hash."""
        try:
            import imagehash
            from PIL import Image
        except ImportError:
            logger.warning("imagehash not available, skipping dedup")
            return images

        # Compute hashes
        hashes = {}
        for img in images:
            try:
                with Image.open(img.path) as pil_img:
                    h = imagehash.phash(pil_img)
                    img.hash = str(h)
                    hashes[img.path] = h
            except Exception as e:
                logger.warning(f"Hash failed for {img.path}: {e}")

        # Find duplicates (greedy)
        unique = []
        seen_hashes = []

        for img in images:
            if img.path not in hashes:
                unique.append(img)
                continue

            h = hashes[img.path]
            is_dup = False

            for seen_h in seen_hashes:
                similarity = 1 - (h - seen_h) / 64  # phash is 64 bits
                if similarity >= threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique.append(img)
                seen_hashes.append(h)

        return unique

    def _sample_diverse(
        self,
        images: List[ImageInfo],
        target_size: int,
    ) -> List[ImageInfo]:
        """
        Sample diverse subset of images.

        Uses quality-weighted random sampling to prefer higher quality images
        while maintaining diversity.
        """
        if len(images) <= target_size:
            return images

        # Weight by quality score
        weights = [img.quality_score for img in images]
        total = sum(weights)
        weights = [w / total for w in weights]

        # Weighted sampling without replacement
        indices = list(range(len(images)))
        selected_indices = []

        for _ in range(target_size):
            if not indices:
                break

            # Weighted random choice
            r = random.random()
            cumsum = 0
            for i, idx in enumerate(indices):
                cumsum += weights[idx]
                if r <= cumsum:
                    selected_indices.append(idx)
                    indices.remove(idx)
                    break

        return [images[i] for i in sorted(selected_indices)]

    def _generate_captions(
        self,
        images: List[ImageInfo],
        config: DatasetConfig,
    ) -> List[ImageInfo]:
        """Generate or load captions for images."""
        trigger = config.effective_trigger

        for img in images:
            # Check for existing caption file
            caption_path = img.path.with_suffix(config.caption_extension)
            if caption_path.exists():
                try:
                    img.caption = caption_path.read_text(encoding="utf-8").strip()
                    # Ensure trigger word
                    if config.use_trigger_word and trigger not in img.caption.lower():
                        img.caption = f"{trigger}, {img.caption}"
                    continue
                except Exception:
                    pass

            # Generate caption
            if self.vlm_captioner:
                try:
                    result = self.vlm_captioner.generate_caption(
                        img.path,
                        prefix=config.caption_prefix,
                    )
                    base_caption = result.caption if hasattr(result, 'caption') else str(result)
                except Exception as e:
                    logger.warning(f"VLM caption failed for {img.path}: {e}")
                    base_caption = "a character"
            else:
                # Default caption
                base_caption = "a 2d animated character"

            # Build final caption
            parts = []
            if config.use_trigger_word:
                parts.append(trigger)
            if config.caption_prefix:
                parts.append(config.caption_prefix)
            parts.append(base_caption)
            if config.caption_suffix:
                parts.append(config.caption_suffix)

            img.caption = ", ".join(parts)

            # Truncate if too long
            tokens = img.caption.split()
            if len(tokens) > config.max_caption_length:
                img.caption = " ".join(tokens[:config.max_caption_length])

        return images

    def _export_dataset(
        self,
        images: List[ImageInfo],
        config: DatasetConfig,
    ) -> None:
        """Export dataset in Kohya-compatible format."""
        output_dir = Path(config.output_dir)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(images):
            # Determine output filename
            stem = img.path.stem
            suffix = img.path.suffix
            dest_image = images_dir / f"{stem}{suffix}"

            # Handle name collisions
            if dest_image.exists():
                dest_image = images_dir / f"{stem}_{i}{suffix}"

            # Copy or symlink image
            if config.copy_images:
                shutil.copy2(img.path, dest_image)
            else:
                dest_image.symlink_to(img.path.resolve())

            # Write caption
            if img.caption:
                caption_path = dest_image.with_suffix(config.caption_extension)
                caption_path.write_text(img.caption, encoding="utf-8")

    def _save_metadata(
        self,
        stats: DatasetStats,
        config: DatasetConfig,
    ) -> None:
        """Save dataset metadata."""
        output_dir = Path(config.output_dir)
        metadata = {
            "stats": stats.to_dict(),
            "config": {
                "character_name": config.character_name,
                "base_model": config.base_model,
                "trigger_word": config.effective_trigger,
                "source_dirs": [str(d) for d in config.source_dirs],
                "target_size": config.target_size,
                "min_resolution": config.min_resolution,
                "generate_captions": config.generate_captions,
            },
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def generate_kohya_config(
        self,
        dataset_dir: Union[str, Path],
        output_path: Union[str, Path],
        base_model: str = "sdxl",
        network_dim: int = 32,
        learning_rate: float = 1e-4,
        epochs: int = 10,
    ) -> Path:
        """
        Generate Kohya training config (TOML format).

        Args:
            dataset_dir: Path to prepared dataset
            output_path: Path for output config file
            base_model: "sd15" or "sdxl"
            network_dim: LoRA network dimension
            learning_rate: Training learning rate
            epochs: Number of training epochs

        Returns:
            Path to generated config file
        """
        dataset_dir = Path(dataset_dir)
        output_path = Path(output_path)

        # Load metadata
        metadata_path = dataset_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            character_name = metadata.get("config", {}).get("character_name", "character")
            trigger_word = metadata.get("config", {}).get("trigger_word", character_name)
            image_count = metadata.get("stats", {}).get("final_image_count", 100)
        else:
            character_name = dataset_dir.name
            trigger_word = character_name.lower().replace(" ", "_")
            image_count = len(list((dataset_dir / "images").glob("*.png")))

        # Calculate training steps
        repeats = max(1, 500 // image_count)  # Target ~500 images per epoch
        steps_per_epoch = (image_count * repeats)
        total_steps = steps_per_epoch * epochs

        # Base model paths
        if base_model == "sdxl":
            model_path = "/mnt/c/ai_models/stable-diffusion/sd_xl_base_1.0.safetensors"
            resolution = 1024
        else:
            model_path = "/mnt/c/ai_models/stable-diffusion/v1-5-pruned-emaonly.safetensors"
            resolution = 512

        # Output path
        lora_output_dir = Path("/mnt/c/ai_models") / f"lora_{base_model}" / character_name

        config_content = f'''# Kohya LoRA Training Config
# Character: {character_name}
# Generated by UnifiedDatasetPreparer

[model]
pretrained_model_name_or_path = "{model_path}"
v2 = false
v_parameterization = false

[train]
output_dir = "{lora_output_dir}"
output_name = "{trigger_word}_lora_{base_model}"
save_model_as = "safetensors"
save_precision = "fp16"
save_every_n_epochs = 2
max_train_epochs = {epochs}
train_batch_size = 1
gradient_accumulation_steps = 4
learning_rate = {learning_rate}
lr_scheduler = "cosine"
lr_warmup_steps = 100
mixed_precision = "bf16"
seed = 42

[network]
network_module = "networks.lora"
network_dim = {network_dim}
network_alpha = {network_dim // 2}

[dataset]
train_data_dir = "{dataset_dir / 'images'}"
resolution = {resolution}
enable_bucket = true
min_bucket_reso = {resolution // 2}
max_bucket_reso = {int(resolution * 1.5)}
bucket_reso_steps = 64
caption_extension = ".txt"
shuffle_caption = true
keep_tokens = 1

[optimizer]
optimizer_type = "AdamW8bit"
optimizer_args = []

[logging]
logging_dir = "{lora_output_dir / 'logs'}"
log_prefix = "{trigger_word}"
'''

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(config_content, encoding="utf-8")
        logger.info(f"Generated Kohya config: {output_path}")

        return output_path


def prepare_from_clusters(
    cluster_dir: Union[str, Path],
    output_dir: Union[str, Path],
    character_names: Optional[Dict[int, str]] = None,
    base_model: str = "sdxl",
    target_size: int = 400,
) -> List[DatasetStats]:
    """
    Convenience function to prepare datasets from clustering output.

    Args:
        cluster_dir: Directory containing character_* subdirectories
        output_dir: Base output directory for datasets
        character_names: Optional mapping of cluster_id to character name
        base_model: "sd15" or "sdxl"
        target_size: Target images per character

    Returns:
        List of DatasetStats for each character
    """
    cluster_dir = Path(cluster_dir)
    output_dir = Path(output_dir)

    preparer = UnifiedDatasetPreparer()
    results = []

    # Find character directories
    char_dirs = sorted(cluster_dir.glob("character_*"))

    for char_dir in char_dirs:
        # Extract cluster ID
        try:
            cluster_id = int(char_dir.name.split("_")[1])
        except (IndexError, ValueError):
            cluster_id = 0

        # Get character name
        if character_names and cluster_id in character_names:
            char_name = character_names[cluster_id]
        else:
            char_name = char_dir.name

        config = DatasetConfig(
            source_dirs=[char_dir],
            character_name=char_name,
            output_dir=output_dir / char_name,
            base_model=base_model,
            target_size=target_size,
            generate_captions=True,
            caption_prefix="a 2d animated character",
        )

        stats = preparer.prepare_character_dataset(config)
        results.append(stats)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare training dataset for LoRA"
    )
    parser.add_argument(
        "source_dirs",
        nargs="+",
        help="Source directories containing character images"
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--character-name", "-n",
        required=True,
        help="Character name"
    )
    parser.add_argument(
        "--base-model",
        choices=["sd15", "sdxl"],
        default="sdxl",
        help="Base model type (default: sdxl)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=400,
        help="Target dataset size (default: 400)"
    )
    parser.add_argument(
        "--trigger-word",
        help="Trigger word (default: auto from character name)"
    )
    parser.add_argument(
        "--caption-prefix",
        default="a 2d animated character",
        help="Caption prefix"
    )
    parser.add_argument(
        "--no-captions",
        action="store_true",
        help="Skip caption generation"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip duplicate removal"
    )
    parser.add_argument(
        "--min-resolution",
        type=int,
        default=512,
        help="Minimum image resolution"
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Also generate Kohya training config"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = DatasetConfig(
        source_dirs=[Path(d) for d in args.source_dirs],
        character_name=args.character_name,
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        target_size=args.target_size,
        trigger_word=args.trigger_word,
        caption_prefix=args.caption_prefix,
        generate_captions=not args.no_captions,
        remove_duplicates=not args.no_dedup,
        min_resolution=args.min_resolution,
    )

    preparer = UnifiedDatasetPreparer()
    stats = preparer.prepare_character_dataset(config)

    print(f"\nDataset Prepared:")
    print(f"  Character: {stats.character_name}")
    print(f"  Source images: {stats.total_source_images}")
    print(f"  After quality: {stats.images_after_quality_filter}")
    print(f"  After dedup: {stats.images_after_dedup}")
    print(f"  Final count: {stats.final_image_count}")
    print(f"  Avg resolution: {stats.avg_resolution}")
    print(f"  Output: {stats.output_dir}")

    if args.generate_config:
        config_path = Path(args.output_dir) / f"{args.character_name}_train.toml"
        preparer.generate_kohya_config(
            args.output_dir,
            config_path,
            base_model=args.base_model,
        )
        print(f"  Config: {config_path}")
