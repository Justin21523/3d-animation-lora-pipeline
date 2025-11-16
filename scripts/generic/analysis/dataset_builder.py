#!/usr/bin/env python3
"""
Multi-Modal Dataset Builder

Purpose: Build training datasets from analyzed content
Features: Multiple formats (HuggingFace, PyTorch, JSON), train/val/test splits
Use Cases: Character LoRA, action recognition, audio-visual learning

Usage:
    python dataset_builder.py \
        --input-dir /path/to/processed_data \
        --output-dir /path/to/dataset \
        --dataset-type character \
        --format huggingface \
        --split-ratio 0.8 0.1 0.1 \
        --project luca
"""

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from collections import defaultdict
import random

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class DatasetConfig:
    """Configuration for dataset building"""
    dataset_type: str = "character"  # character, multimodal, temporal, action
    format: str = "pytorch"  # pytorch, huggingface, json, webdataset
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # train/val/test
    min_samples_per_class: int = 10
    max_samples_per_class: Optional[int] = None
    image_size: Tuple[int, int] = (512, 512)
    quality_threshold: float = 0.5
    include_audio: bool = False
    include_temporal: bool = False
    seed: int = 42


class DatasetBuilder:
    """Build multi-modal training datasets"""

    def __init__(self, config: DatasetConfig):
        """
        Initialize dataset builder

        Args:
            config: Dataset configuration
        """
        self.config = config
        random.seed(config.seed)
        np.random.seed(config.seed)

    def parse_frame_number(self, frame_name: str) -> Optional[int]:
        """Parse frame number from filename"""
        import re
        numbers = re.findall(r'\d+', frame_name)
        if numbers:
            return int(numbers[0])
        return None

    def load_character_clusters(self, clusters_dir: Path) -> Dict[str, List[Path]]:
        """
        Load character identity clusters

        Args:
            clusters_dir: Directory with character clusters

        Returns:
            Dictionary mapping character to image paths
        """
        clusters_dir = Path(clusters_dir)
        clusters = {}

        for char_dir in clusters_dir.iterdir():
            if char_dir.is_dir() and not char_dir.name.startswith('.') and char_dir.name != 'noise':
                character = char_dir.name
                images = list(char_dir.glob("*.png")) + list(char_dir.glob("*.jpg"))

                if len(images) >= self.config.min_samples_per_class:
                    # Apply max samples limit if specified
                    if self.config.max_samples_per_class:
                        images = random.sample(images, min(len(images), self.config.max_samples_per_class))
                    clusters[character] = images

        print(f"   Loaded {len(clusters)} character clusters")
        for char, imgs in clusters.items():
            print(f"      {char}: {len(imgs)} images")

        return clusters

    def load_quality_scores(self, quality_json: Path) -> Dict[str, float]:
        """
        Load quality scores from preprocessing

        Args:
            quality_json: Path to quality filter results

        Returns:
            Dictionary mapping image path to quality score
        """
        if not quality_json.exists():
            return {}

        with open(quality_json, 'r') as f:
            data = json.load(f)

        scores = {}
        for result in data.get('filtered_images', []):
            scores[result['path']] = result.get('overall_score', 1.0)

        return scores

    def load_captions(self, captions_dir: Path) -> Dict[str, str]:
        """
        Load captions for images

        Args:
            captions_dir: Directory with caption text files

        Returns:
            Dictionary mapping image path to caption
        """
        captions = {}

        if not captions_dir.exists():
            return captions

        for caption_file in captions_dir.glob("*.txt"):
            with open(caption_file, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            # Match with corresponding image
            img_name = caption_file.stem
            captions[img_name] = caption

        return captions

    def load_audio_segments(self, audio_json: Path) -> Dict[int, Dict]:
        """
        Load audio segments mapped to frames

        Args:
            audio_json: Path to audio analysis JSON

        Returns:
            Dictionary mapping frame to audio info
        """
        if not audio_json.exists():
            return {}

        with open(audio_json, 'r') as f:
            data = json.load(f)

        # Map frames to audio segments
        frame_audio = {}
        for segment in data.get('segments', []):
            start_frame = int(segment.get('start', 0) * 24)  # Assume 24 fps
            end_frame = int(segment.get('end', 0) * 24)

            for frame in range(start_frame, end_frame + 1):
                frame_audio[frame] = {
                    'speaker': segment.get('speaker', 'unknown'),
                    'start': segment['start'],
                    'end': segment['end']
                }

        return frame_audio

    def load_motion_data(self, motion_json: Path) -> Dict[int, Dict]:
        """
        Load motion analysis data

        Args:
            motion_json: Path to motion analysis JSON

        Returns:
            Dictionary mapping frame to motion info
        """
        if not motion_json.exists():
            return {}

        with open(motion_json, 'r') as f:
            data = json.load(f)

        motion_data = {}
        for frame_data in data.get('motion_data', []):
            frame_idx = frame_data.get('frame_idx')
            if frame_idx is not None:
                motion_data[frame_idx] = {
                    'motion_type': frame_data.get('motion_type', 'unknown'),
                    'magnitude_mean': frame_data.get('magnitude_mean', 0.0),
                    'action_type': frame_data.get('action_type', 'idle')
                }

        return motion_data

    def split_dataset(
        self,
        samples: List[Any],
        by_class: bool = False,
        class_labels: Optional[List[str]] = None
    ) -> Tuple[List, List, List]:
        """
        Split dataset into train/val/test

        Args:
            samples: List of samples
            by_class: Whether to stratify by class
            class_labels: Class labels for stratification

        Returns:
            Train, validation, test splits
        """
        train_ratio, val_ratio, test_ratio = self.config.split_ratio

        if by_class and class_labels:
            # Stratified split
            unique_classes = sorted(set(class_labels))
            train_samples, val_samples, test_samples = [], [], []

            for cls in unique_classes:
                cls_indices = [i for i, label in enumerate(class_labels) if label == cls]
                cls_samples = [samples[i] for i in cls_indices]

                n_samples = len(cls_samples)
                n_train = int(n_samples * train_ratio)
                n_val = int(n_samples * val_ratio)

                random.shuffle(cls_samples)

                train_samples.extend(cls_samples[:n_train])
                val_samples.extend(cls_samples[n_train:n_train + n_val])
                test_samples.extend(cls_samples[n_train + n_val:])

        else:
            # Random split
            n_samples = len(samples)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)

            shuffled = samples.copy()
            random.shuffle(shuffled)

            train_samples = shuffled[:n_train]
            val_samples = shuffled[n_train:n_train + n_val]
            test_samples = shuffled[n_train + n_val:]

        print(f"   Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")

        return train_samples, val_samples, test_samples

    def resize_image(self, image_path: Path, target_size: Tuple[int, int]) -> Image.Image:
        """Resize image to target size"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img

    def build_character_dataset(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Build character identification/generation dataset

        Args:
            input_dir: Input directory with clusters
            output_dir: Output directory

        Returns:
            Dataset metadata
        """
        print(f"\n👤 Building character dataset...")

        # Load character clusters
        clusters_dir = input_dir / "clusters" if (input_dir / "clusters").exists() else input_dir
        characters = self.load_character_clusters(clusters_dir)

        if not characters:
            raise ValueError("No character clusters found")

        # Load optional data
        quality_scores = self.load_quality_scores(input_dir / "quality_filter.json")
        captions_dir = input_dir / "captions"
        captions = self.load_captions(captions_dir) if captions_dir.exists() else {}

        # Prepare samples
        samples = []
        class_labels = []

        for character, images in characters.items():
            for img_path in images:
                # Apply quality filter
                quality = quality_scores.get(str(img_path), 1.0)
                if quality < self.config.quality_threshold:
                    continue

                caption = captions.get(img_path.stem, f"a {character}")

                samples.append({
                    'image_path': img_path,
                    'character': character,
                    'caption': caption,
                    'quality': quality
                })
                class_labels.append(character)

        print(f"   Total samples: {len(samples)}")

        # Split dataset
        train_samples, val_samples, test_samples = self.split_dataset(
            samples,
            by_class=True,
            class_labels=class_labels
        )

        # Build dataset in specified format
        if self.config.format == "pytorch":
            metadata = self.build_pytorch_dataset(
                train_samples, val_samples, test_samples, output_dir
            )
        elif self.config.format == "huggingface":
            metadata = self.build_huggingface_dataset(
                train_samples, val_samples, test_samples, output_dir
            )
        elif self.config.format == "json":
            metadata = self.build_json_dataset(
                train_samples, val_samples, test_samples, output_dir
            )
        else:
            raise ValueError(f"Unsupported format: {self.config.format}")

        metadata['dataset_type'] = 'character'
        metadata['num_classes'] = len(characters)
        metadata['classes'] = list(characters.keys())

        return metadata

    def build_multimodal_dataset(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Build multi-modal dataset with audio and visual features

        Args:
            input_dir: Input directory
            output_dir: Output directory

        Returns:
            Dataset metadata
        """
        print(f"\n🎬 Building multi-modal dataset...")

        # Load visual data
        clusters_dir = input_dir / "clusters"
        characters = self.load_character_clusters(clusters_dir) if clusters_dir.exists() else {}

        # Load audio data
        audio_json = input_dir / "speaker_diarization.json"
        audio_segments = self.load_audio_segments(audio_json) if audio_json.exists() else {}

        # Load sync data
        sync_json = input_dir / "multimodal_sync.json"
        sync_data = {}
        if sync_json.exists():
            with open(sync_json, 'r') as f:
                sync_data = json.load(f)

        # Prepare multi-modal samples
        samples = []

        for character, images in characters.items():
            for img_path in images:
                frame_num = self.parse_frame_number(img_path.name)
                if frame_num is None:
                    continue

                # Get audio segment
                audio_info = audio_segments.get(frame_num, {})

                sample = {
                    'image_path': img_path,
                    'character': character,
                    'frame': frame_num,
                    'has_audio': bool(audio_info),
                    'speaker': audio_info.get('speaker', 'none')
                }

                samples.append(sample)

        print(f"   Total multi-modal samples: {len(samples)}")

        # Split and build
        train_samples, val_samples, test_samples = self.split_dataset(samples, by_class=False)

        if self.config.format == "json":
            metadata = self.build_json_dataset(
                train_samples, val_samples, test_samples, output_dir
            )
        else:
            metadata = self.build_pytorch_dataset(
                train_samples, val_samples, test_samples, output_dir
            )

        metadata['dataset_type'] = 'multimodal'
        metadata['has_audio'] = len(audio_segments) > 0

        return metadata

    def build_temporal_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        sequence_length: int = 8
    ) -> Dict:
        """
        Build temporal sequence dataset for action recognition

        Args:
            input_dir: Input directory
            output_dir: Output directory
            sequence_length: Number of frames per sequence

        Returns:
            Dataset metadata
        """
        print(f"\n⏱️ Building temporal sequence dataset...")

        # Load frames
        frames_dir = input_dir / "frames" if (input_dir / "frames").exists() else input_dir
        frames = sorted(
            list(frames_dir.glob("frame_*.jpg")) +
            list(frames_dir.glob("frame_*.png"))
        )

        # Load motion data
        motion_json = input_dir / "motion_analysis.json"
        motion_data = self.load_motion_data(motion_json) if motion_json.exists() else {}

        # Create sequences
        sequences = []
        for i in range(0, len(frames) - sequence_length + 1, sequence_length // 2):
            sequence_frames = frames[i:i + sequence_length]

            # Get dominant motion type for sequence
            motion_types = []
            for frame_path in sequence_frames:
                frame_num = self.parse_frame_number(frame_path.name)
                if frame_num and frame_num in motion_data:
                    motion_types.append(motion_data[frame_num].get('action_type', 'idle'))

            dominant_motion = max(set(motion_types), key=motion_types.count) if motion_types else 'idle'

            sequences.append({
                'frames': [str(f) for f in sequence_frames],
                'action': dominant_motion,
                'start_frame': self.parse_frame_number(sequence_frames[0].name),
                'length': len(sequence_frames)
            })

        print(f"   Total sequences: {len(sequences)}")

        # Split
        train_seq, val_seq, test_seq = self.split_dataset(sequences, by_class=False)

        metadata = self.build_json_dataset(train_seq, val_seq, test_seq, output_dir)
        metadata['dataset_type'] = 'temporal'
        metadata['sequence_length'] = sequence_length

        return metadata

    def build_pytorch_dataset(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        test_samples: List[Dict],
        output_dir: Path
    ) -> Dict:
        """Build PyTorch-style dataset"""
        output_dir = Path(output_dir)

        for split_name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            # Group by class if available
            if samples and 'character' in samples[0]:
                classes = sorted(set(s['character'] for s in samples))
                for cls in classes:
                    (split_dir / cls).mkdir(exist_ok=True)

                for idx, sample in enumerate(tqdm(samples, desc=f"Building {split_name}")):
                    src_img = sample['image_path']
                    cls = sample['character']
                    dst_img = split_dir / cls / f"{idx:06d}.png"

                    # Resize and save
                    img = self.resize_image(src_img, self.config.image_size)
                    img.save(dst_img)

                    # Save caption if exists
                    if 'caption' in sample:
                        caption_file = dst_img.with_suffix('.txt')
                        with open(caption_file, 'w', encoding='utf-8') as f:
                            f.write(sample['caption'])
            else:
                # Flat structure
                for idx, sample in enumerate(tqdm(samples, desc=f"Building {split_name}")):
                    src_img = sample['image_path']
                    dst_img = split_dir / f"{idx:06d}.png"

                    img = self.resize_image(src_img, self.config.image_size)
                    img.save(dst_img)

        metadata = {
            'format': 'pytorch',
            'splits': {
                'train': len(train_samples),
                'val': len(val_samples),
                'test': len(test_samples)
            },
            'image_size': self.config.image_size
        }

        return metadata

    def build_huggingface_dataset(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        test_samples: List[Dict],
        output_dir: Path
    ) -> Dict:
        """Build HuggingFace datasets format"""
        try:
            from datasets import Dataset, DatasetDict, Features, ClassLabel, Image as HFImage, Value
        except ImportError:
            print("   ⚠️ datasets library not available, falling back to PyTorch format")
            return self.build_pytorch_dataset(train_samples, val_samples, test_samples, output_dir)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def prepare_hf_samples(samples):
            hf_samples = []
            for sample in samples:
                img = self.resize_image(sample['image_path'], self.config.image_size)
                hf_sample = {
                    'image': img,
                    'character': sample.get('character', 'unknown'),
                    'caption': sample.get('caption', ''),
                    'quality': sample.get('quality', 1.0)
                }
                hf_samples.append(hf_sample)
            return hf_samples

        # Create datasets
        datasets = {}
        if train_samples:
            datasets['train'] = Dataset.from_list(prepare_hf_samples(train_samples))
        if val_samples:
            datasets['validation'] = Dataset.from_list(prepare_hf_samples(val_samples))
        if test_samples:
            datasets['test'] = Dataset.from_list(prepare_hf_samples(test_samples))

        dataset_dict = DatasetDict(datasets)

        # Save to disk
        dataset_dict.save_to_disk(str(output_dir / "hf_dataset"))

        print(f"   HuggingFace dataset saved to {output_dir / 'hf_dataset'}")

        metadata = {
            'format': 'huggingface',
            'splits': {k: len(v) for k, v in datasets.items()},
            'image_size': self.config.image_size
        }

        return metadata

    def build_json_dataset(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        test_samples: List[Dict],
        output_dir: Path
    ) -> Dict:
        """Build JSON metadata dataset"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy images and create metadata
        for split_name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
            split_dir = output_dir / split_name / "images"
            split_dir.mkdir(parents=True, exist_ok=True)

            metadata_list = []

            for idx, sample in enumerate(tqdm(samples, desc=f"Building {split_name}")):
                # Copy or resize image
                if 'image_path' in sample:
                    src_img = sample['image_path']
                    dst_img = split_dir / f"{idx:06d}.png"

                    img = self.resize_image(src_img, self.config.image_size)
                    img.save(dst_img)

                    # Create metadata entry
                    meta = {
                        'image_id': idx,
                        'image_path': f"images/{dst_img.name}",
                        **{k: v for k, v in sample.items() if k != 'image_path'}
                    }
                else:
                    meta = {
                        'image_id': idx,
                        **sample
                    }

                metadata_list.append(meta)

            # Save metadata JSON
            metadata_file = output_dir / split_name / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata_list, f, indent=2)

        metadata = {
            'format': 'json',
            'splits': {
                'train': len(train_samples),
                'val': len(val_samples),
                'test': len(test_samples)
            },
            'image_size': self.config.image_size
        }

        return metadata

    def build_dataset(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main dataset building pipeline

        Args:
            input_dir: Input directory with processed data
            output_dir: Output directory for dataset

        Returns:
            Dataset metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📦 Dataset Builder")
        print(f"   Input: {input_dir}")
        print(f"   Output: {output_dir}")
        print(f"   Type: {self.config.dataset_type}")
        print(f"   Format: {self.config.format}")

        # Build dataset based on type
        if self.config.dataset_type == "character":
            metadata = self.build_character_dataset(input_dir, output_dir)
        elif self.config.dataset_type == "multimodal":
            metadata = self.build_multimodal_dataset(input_dir, output_dir)
        elif self.config.dataset_type == "temporal":
            metadata = self.build_temporal_dataset(input_dir, output_dir)
        else:
            raise ValueError(f"Unsupported dataset type: {self.config.dataset_type}")

        # Add common metadata
        metadata.update({
            'created': datetime.now().isoformat(),
            'config': asdict(self.config),
            'input_dir': str(input_dir),
            'output_dir': str(output_dir)
        })

        # Save metadata
        metadata_path = output_dir / "dataset_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Dataset built successfully")
        print(f"   Total samples: {sum(metadata['splits'].values())}")
        print(f"   Metadata saved: {metadata_path}")

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Modal Dataset Builder (Film-Agnostic)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory with processed data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="character",
        choices=["character", "multimodal", "temporal"],
        help="Type of dataset to build"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pytorch",
        choices=["pytorch", "huggingface", "json"],
        help="Dataset format"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help="Train/val/test split ratio"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Target image size (width height)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples per class"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples per class"
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.5,
        help="Minimum quality score"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name"
    )

    args = parser.parse_args()

    config = DatasetConfig(
        dataset_type=args.dataset_type,
        format=args.format,
        split_ratio=tuple(args.split_ratio),
        min_samples_per_class=args.min_samples,
        max_samples_per_class=args.max_samples,
        image_size=tuple(args.image_size),
        quality_threshold=args.quality_threshold,
        seed=args.seed
    )

    builder = DatasetBuilder(config)
    metadata = builder.build_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")

    # Print summary
    print(f"\n📊 Dataset Summary:")
    for split, count in metadata['splits'].items():
        print(f"   {split}: {count} samples")


if __name__ == "__main__":
    main()
