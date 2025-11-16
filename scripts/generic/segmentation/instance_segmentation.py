#!/usr/bin/env python3
"""
Instance-level Segmentation for Multiple Characters

Extracts EACH character instance separately from frames containing multiple characters.
Uses SAM2 for precise instance segmentation.

Key Features:
- Detects and segments multiple character instances per frame
- Each character becomes a separate image (not grouped by frame)
- Preserves character identity information
- Optimized for 3D animation with clean edges
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional
import cv2
from tqdm import tqdm
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class SAM2InstanceSegmenter:
    """
    SAM2-based instance segmentation for multiple characters
    """

    def __init__(
        self,
        model_type: str = "sam2_hiera_large",
        device: str = "cuda",
        min_mask_size: int = 64 * 64,  # Minimum character size (4096 pixels, captures distant characters)
        max_instances: int = 15  # Maximum characters per frame (increased for scenes with multiple people)
    ):
        """
        Initialize SAM2 model

        Args:
            model_type: SAM2 model variant
            device: cuda or cpu
            min_mask_size: Minimum mask area to consider
            max_instances: Maximum number of instances to extract per frame
        """
        self.device = device
        self.model_type = model_type
        self.min_mask_size = min_mask_size
        self.max_instances = max_instances

        print(f"🔧 Initializing SAM2 instance segmenter ({model_type})...")
        self._init_sam2()

    def _init_sam2(self):
        """Initialize SAM2 model"""
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            # Model checkpoint paths
            checkpoint_path = self._get_checkpoint_path()
            config_file = self._get_config_file()

            # Build SAM2 model
            sam2_model = build_sam2(
                config_file,
                checkpoint_path,
                device=self.device
            )

            # Create automatic mask generator - OPTIMIZED FOR COMPLETE CHARACTER DETECTION
            # Updated 2025-11-16: Increased sensitivity to capture ALL character instances
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=32,  # Increased from 20: better character boundary detection
                pred_iou_thresh=0.70,  # Lowered from 0.76: capture more instances (prevents missing characters)
                stability_score_thresh=0.80,  # Lowered from 0.86: include partial occlusions
                crop_n_layers=0,  # Keep disabled for stability
                crop_n_points_downscale_factor=2,
                min_mask_region_area=self.min_mask_size,
                points_per_batch=192  # Balanced batch size
            )

            print("✓ SAM2 model loaded successfully")

        except ImportError:
            print("⚠️ SAM2 not installed. Falling back to alternative method...")
            self._init_fallback()

    def _get_checkpoint_path(self) -> str:
        """Get SAM2 checkpoint path"""
        # Check global config first
        from pathlib import Path
        model_dir = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/segmentation")

        checkpoint_mapping = {
            "sam2_hiera_large": "sam2_hiera_large.pt",
            "sam2_hiera_base": "sam2_hiera_base_plus.pt",
            "sam2_hiera_small": "sam2_hiera_small.pt",
            "sam2_hiera_tiny": "sam2_hiera_tiny.pt"
        }

        checkpoint_file = checkpoint_mapping.get(self.model_type, "sam2_hiera_large.pt")
        checkpoint_path = model_dir / checkpoint_file

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {checkpoint_path}\n"
                f"Please download from: https://github.com/facebookresearch/sam2"
            )

        return str(checkpoint_path)

    def _get_config_file(self) -> str:
        """Get SAM2 config file"""
        config_mapping = {
            "sam2_hiera_large": "sam2_hiera_l.yaml",
            "sam2_hiera_base": "sam2_hiera_b+.yaml",
            "sam2_hiera_small": "sam2_hiera_s.yaml",
            "sam2_hiera_tiny": "sam2_hiera_t.yaml"
        }
        return config_mapping.get(self.model_type, "sam2_hiera_l.yaml")

    def _init_fallback(self):
        """Fallback to simpler instance detection"""
        print("Using fallback: Connected Components + Contour Detection")
        self.mask_generator = None  # Use fallback method

    def segment_instances(
        self,
        image: Image.Image,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Segment all character instances in an image

        Args:
            image: PIL Image
            return_scores: Return confidence scores

        Returns:
            List of instance dictionaries with masks and metadata
        """
        if self.mask_generator is not None:
            return self._segment_with_sam2(image, return_scores)
        else:
            return self._segment_with_fallback(image)

    def _segment_with_sam2(
        self,
        image: Image.Image,
        return_scores: bool
    ) -> List[Dict]:
        """Segment using SAM2"""
        # Convert to numpy
        image_np = np.array(image)

        # Generate masks
        masks = self.mask_generator.generate(image_np)

        # Sort by area (largest first) and filter
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        masks = masks[:self.max_instances]

        instances = []
        for idx, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']  # [x, y, w, h]
            area = mask_data['area']
            stability_score = mask_data.get('stability_score', 0.0)
            predicted_iou = mask_data.get('predicted_iou', 0.0)

            # Quality checks
            if area < self.min_mask_size:
                continue

            instance = {
                'instance_id': idx,
                'mask': mask,
                'bbox': bbox,
                'area': area,
                'stability_score': stability_score,
                'predicted_iou': predicted_iou
            }

            instances.append(instance)

        return instances

    def _segment_with_fallback(self, image: Image.Image) -> List[Dict]:
        """Fallback: Use background removal + connected components"""
        from rembg import remove, new_session

        # Remove background
        session = new_session("isnet-general-use")
        output = remove(image, session=session, only_mask=True)
        mask = np.array(output)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        instances = []
        for idx in range(1, num_labels):  # Skip background (0)
            area = stats[idx, cv2.CC_STAT_AREA]

            if area < self.min_mask_size:
                continue

            # Create instance mask
            instance_mask = (labels == idx).astype(np.uint8) * 255

            # Get bounding box
            x = stats[idx, cv2.CC_STAT_LEFT]
            y = stats[idx, cv2.CC_STAT_TOP]
            w = stats[idx, cv2.CC_STAT_WIDTH]
            h = stats[idx, cv2.CC_STAT_HEIGHT]

            instance = {
                'instance_id': idx - 1,
                'mask': instance_mask,
                'bbox': [x, y, w, h],
                'area': area,
                'stability_score': 0.5,  # Placeholder
                'predicted_iou': 0.5
            }

            instances.append(instance)

        # Sort by area
        instances = sorted(instances, key=lambda x: x['area'], reverse=True)
        return instances[:self.max_instances]

    def extract_instance_image(
        self,
        image: Image.Image,
        instance: Dict,
        mode: str = "transparent",
        padding: int = 10,
        blur_strength: int = 15
    ) -> Image.Image:
        """
        Extract a single character instance with different background modes

        Args:
            image: Original image
            instance: Instance dictionary with mask and bbox
            mode: Background mode - "transparent", "context", or "blurred"
            padding: Padding around bounding box
            blur_strength: Blur kernel size for blurred mode (must be odd)

        Returns:
            Cropped character image with specified background mode
        """
        image_np = np.array(image)
        mask = instance['mask']
        bbox = instance['bbox']

        # Expand bbox with padding
        x, y, w, h = bbox
        # Convert to integers (SAM2 bbox values may be floats)
        x = int(max(0, x - padding))
        y = int(max(0, y - padding))
        w = int(min(image_np.shape[1] - x, w + 2 * padding))
        h = int(min(image_np.shape[0] - y, h + 2 * padding))

        # Crop image and mask
        cropped_image = image_np[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]

        if mode == "transparent":
            # RGBA with transparent background (best for compositing)
            rgba = np.zeros((*cropped_image.shape[:2], 4), dtype=np.uint8)
            rgba[:, :, :3] = cropped_image
            rgba[:, :, 3] = (cropped_mask * 255).astype(np.uint8)
            return Image.fromarray(rgba)

        elif mode == "context":
            # Keep original background (best for training with real context)
            # Just return the cropped region - no transparency needed
            return Image.fromarray(cropped_image)

        elif mode == "blurred":
            # Blur background, keep character sharp (emphasize character)
            # Ensure blur_strength is odd
            if blur_strength % 2 == 0:
                blur_strength += 1

            # Blur the entire cropped image
            blurred = cv2.GaussianBlur(cropped_image, (blur_strength, blur_strength), 0)

            # Composite: sharp where mask=True, blurred where mask=False
            mask_3ch = cropped_mask[:, :, np.newaxis].astype(bool)
            result = np.where(mask_3ch, cropped_image, blurred)
            return Image.fromarray(result)

        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'transparent', 'context', or 'blurred'")


def process_frames_to_instances(
    input_dir: Path,
    output_dir: Path,
    model_type: str = "sam2_hiera_large",
    device: str = "cuda",
    min_instance_size: int = 128 * 128,
    save_metadata: bool = True,
    save_visualization: bool = False,
    save_masks: bool = False,
    save_backgrounds: bool = False,
    context_mode: str = "all",
    context_padding: int = 20,
    cache_clear_interval: int = 10  # Reduced from 50 to clear GPU cache more frequently
) -> Dict:
    """
    Process all frames and extract character instances

    Args:
        input_dir: Directory with input frames
        output_dir: Directory to save character instances
        model_type: SAM2 model type
        device: cuda or cpu
        min_instance_size: Minimum instance area
        save_metadata: Save instance metadata JSON
        save_visualization: Save visualization of detected instances

    Returns:
        Processing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output structure based on context_mode
    # Determine which modes to generate
    modes_to_generate = []
    if context_mode == "all":
        modes_to_generate = ["transparent", "context", "blurred"]
    else:
        modes_to_generate = [context_mode]

    # Create directories for each mode
    mode_dirs = {}
    for mode in modes_to_generate:
        if mode == "transparent":
            mode_dirs[mode] = output_dir / "instances"
        elif mode == "context":
            mode_dirs[mode] = output_dir / "instances_context"
        elif mode == "blurred":
            mode_dirs[mode] = output_dir / "instances_blurred"
        mode_dirs[mode].mkdir(parents=True, exist_ok=True)

    if save_visualization:
        viz_dir = output_dir / "visualization"
        viz_dir.mkdir(exist_ok=True)

    if save_masks:
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)

    if save_backgrounds:
        backgrounds_dir = output_dir / "backgrounds"
        backgrounds_dir.mkdir(exist_ok=True)

    # Initialize segmenter
    segmenter = SAM2InstanceSegmenter(
        model_type=model_type,
        device=device,
        min_mask_size=min_instance_size
    )

    # Find all frames
    image_files = sorted(
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.png"))
    )

    print(f"\n📊 Processing {len(image_files)} frames...")

    stats = {
        'total_frames': len(image_files),
        'total_instances': 0,
        'instances_per_frame': [],
        'frames_with_multiple': 0,
        'processing_time': 0,
        'failed_frames': 0,
        'skipped_frames': 0
    }

    metadata = []
    failed_frames_log = []

    # Check for already processed frames (resume capability)
    processed_frames = set()
    # Check the first mode directory for processed frames
    first_mode_dir = mode_dirs[modes_to_generate[0]]
    if first_mode_dir.exists():
        for inst_file in first_mode_dir.glob("*_inst*.png"):
            # Extract source frame name from instance filename
            # Format: scene0090_pos5_frame000905_t456.16s_inst0.png
            frame_name = '_'.join(inst_file.stem.split('_')[:-1])  # Remove _inst0 part
            processed_frames.add(frame_name)

    print(f"📊 Found {len(processed_frames)} already processed frames, will skip them...")

    # Load existing failed frames log if exists
    failed_log_path = output_dir / "failed_frames.json"
    if failed_log_path.exists():
        with open(failed_log_path, 'r') as f:
            failed_frames_log = json.load(f).get('failed_frames', [])

    for frame_idx, image_path in enumerate(tqdm(image_files, desc="Extracting instances")):
        # Skip already processed frames
        if image_path.stem in processed_frames:
            stats['skipped_frames'] += 1
            continue

        # Clear GPU cache periodically to prevent memory leaks
        if device == "cuda" and frame_idx > 0 and frame_idx % cache_clear_interval == 0:
            torch.cuda.empty_cache()

        # Process frame with error handling and retry logic
        max_retries = 3
        retry_count = 0
        frame_processed = False

        while retry_count < max_retries and not frame_processed:
            try:
                import time
                start_time = time.time()

                # Load image
                image = Image.open(image_path).convert("RGB")

                # Segment instances with timeout protection
                # Note: Python doesn't have built-in timeout for function calls
                # We rely on SAM2's internal processing to complete or fail
                instances = segmenter.segment_instances(image)

                # Track stats
                num_instances = len(instances)
                stats['total_instances'] += num_instances
                stats['instances_per_frame'].append(num_instances)

                if num_instances > 1:
                    stats['frames_with_multiple'] += 1

                # Extract and save each instance
                for inst_idx, instance in enumerate(instances):
                    # Generate filename base
                    frame_name = image_path.stem

                    # Save instance in each requested mode
                    for mode in modes_to_generate:
                        # Extract instance with specific mode
                        instance_image = segmenter.extract_instance_image(
                            image, instance, mode=mode, padding=context_padding
                        )

                        # Generate filename based on mode
                        if mode == "transparent":
                            instance_filename = f"{frame_name}_inst{inst_idx}.png"
                        elif mode == "context":
                            instance_filename = f"{frame_name}_inst{inst_idx}_ctx.png"
                        elif mode == "blurred":
                            instance_filename = f"{frame_name}_inst{inst_idx}_blur.png"

                        instance_path = mode_dirs[mode] / instance_filename

                        # Save instance
                        instance_image.save(instance_path)

                    # Save mask if requested (only once, not per mode)
                    if save_masks:
                        mask = instance['mask']
                        # Convert boolean mask to uint8 grayscale (0-255)
                        mask_uint8 = (mask * 255).astype(np.uint8)
                        mask_image = Image.fromarray(mask_uint8, mode='L')

                        mask_filename = f"{frame_name}_inst{inst_idx}_mask.png"
                        mask_path = masks_dir / mask_filename
                        mask_image.save(mask_path)

                    # Save metadata
                    if save_metadata:
                        meta = {
                            'instance_filename': instance_filename,
                            'source_frame': image_path.name,
                            'source_frame_index': frame_idx,
                            'instance_index': inst_idx,
                            'bbox': instance['bbox'],
                            'area': instance['area'],
                            'stability_score': instance['stability_score'],
                            'predicted_iou': instance['predicted_iou']
                        }
                        metadata.append(meta)

                # Save background (inpainted) if requested
                if save_backgrounds and num_instances > 0:
                    # Create combined mask for all instances
                    combined_mask = np.zeros(image.size[::-1], dtype=np.uint8)
                    for instance in instances:
                        mask = instance['mask']
                        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)

                    # Dilate mask slightly to ensure full coverage
                    kernel = np.ones((5, 5), np.uint8)
                    combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)

                    # Convert to format needed by inpainting
                    mask_uint8 = (combined_mask * 255).astype(np.uint8)

                    # Simple inpainting using OpenCV (fast, good enough for backgrounds)
                    image_np = np.array(image)
                    inpainted = cv2.inpaint(image_np, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

                    # Save inpainted background
                    background_image = Image.fromarray(inpainted)
                    background_filename = f"{frame_name}_background.jpg"
                    background_path = backgrounds_dir / background_filename
                    background_image.save(background_path, quality=95)

                # Save visualization
                if save_visualization and num_instances > 0:
                    viz_image = visualize_instances(image, instances)
                    viz_path = viz_dir / f"{frame_name}_instances.jpg"
                    viz_image.save(viz_path, quality=90)

                # Successfully processed
                frame_processed = True

                elapsed = time.time() - start_time
                if elapsed > 30:  # Warn if frame takes too long
                    print(f"\n⚠️  Frame {image_path.name} took {elapsed:.1f}s to process")

            except KeyboardInterrupt:
                # Allow user to interrupt
                print("\n🛑 Processing interrupted by user")
                raise

            except Exception as e:
                retry_count += 1
                error_msg = f"{type(e).__name__}: {str(e)}"

                if retry_count < max_retries:
                    print(f"\n⚠️  Error processing {image_path.name} (attempt {retry_count}/{max_retries}): {error_msg}")
                    print(f"   Retrying in 2 seconds...")
                    time.sleep(2)

                    # Clear GPU cache before retry
                    if device == "cuda":
                        torch.cuda.empty_cache()
                else:
                    # Max retries reached, log and skip
                    print(f"\n❌ Failed to process {image_path.name} after {max_retries} attempts: {error_msg}")
                    print(f"   Skipping this frame...")

                    stats['failed_frames'] += 1
                    failed_frames_log.append({
                        'frame_path': str(image_path),
                        'frame_name': image_path.name,
                        'frame_index': frame_idx,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat(),
                        'retries': retry_count
                    })

                    # Save failed frames log immediately
                    with open(failed_log_path, 'w') as f:
                        json.dump({
                            'failed_frames': failed_frames_log,
                            'total_failed': len(failed_frames_log)
                        }, f, indent=2)

                    # Clear GPU cache after failure
                    if device == "cuda":
                        torch.cuda.empty_cache()

                    break  # Skip to next frame

    # Save metadata JSON
    if save_metadata:
        metadata_path = output_dir / "instances_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'metadata': metadata,
                'statistics': stats,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

    # Compute final stats
    if stats['instances_per_frame']:
        stats['avg_instances_per_frame'] = np.mean(stats['instances_per_frame'])
        stats['max_instances_per_frame'] = max(stats['instances_per_frame'])
    else:
        stats['avg_instances_per_frame'] = 0
        stats['max_instances_per_frame'] = 0

    print(f"\n✅ Instance extraction complete!")
    print(f"   Total frames: {stats['total_frames']}")
    print(f"   Processed: {stats['total_frames'] - stats['skipped_frames'] - stats['failed_frames']}")
    print(f"   Skipped (already done): {stats['skipped_frames']}")
    print(f"   Failed: {stats['failed_frames']}")
    print(f"   Total instances: {stats['total_instances']}")
    print(f"   Avg per frame: {stats['avg_instances_per_frame']:.2f}")
    print(f"   Frames with multiple characters: {stats['frames_with_multiple']}")

    if stats['failed_frames'] > 0:
        print(f"\n⚠️  {stats['failed_frames']} frames failed after retries")
        print(f"   See {failed_log_path} for details")

    return stats


def visualize_instances(image: Image.Image, instances: List[Dict]) -> Image.Image:
    """Visualize detected instances with bounding boxes"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(instances)))

    for idx, instance in enumerate(instances):
        bbox = instance['bbox']
        x, y, w, h = bbox

        # Draw bounding box
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=3,
            edgecolor=colors[idx],
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add label
        label = f"Inst {idx}\nIoU: {instance['predicted_iou']:.2f}"
        ax.text(
            x, y - 10,
            label,
            color='white',
            fontsize=10,
            bbox=dict(facecolor=colors[idx], alpha=0.7)
        )

    ax.axis('off')
    fig.tight_layout()

    # Convert to image
    fig.canvas.draw()
    viz_image = Image.frombytes(
        'RGB',
        fig.canvas.get_width_height(),
        fig.canvas.tostring_rgb()
    )
    plt.close(fig)

    return viz_image


def main():
    parser = argparse.ArgumentParser(
        description="Extract character instances from frames with multiple characters (Film-Agnostic)"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory with input frames"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for character instances. If --project is specified, this can be auto-constructed."
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name (e.g., 'luca', 'toy_story', 'finding_nemo'). "
             "Automatically constructs output paths: /mnt/data/ai_data/datasets/3d-anime/{project}/instances"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sam2_hiera_large",
        choices=["sam2_hiera_large", "sam2_hiera_base", "sam2_hiera_small", "sam2_hiera_tiny"],
        help="SAM2 model type"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=128 * 128,
        help="Minimum instance area (pixels)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization of detected instances"
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save instance masks as separate PNG files (grayscale)"
    )
    parser.add_argument(
        "--save-backgrounds",
        action="store_true",
        help="Save inpainted backgrounds (characters removed)"
    )
    parser.add_argument(
        "--context-mode",
        type=str,
        choices=["transparent", "context", "blurred", "all"],
        default="all",
        help="Background handling mode: transparent (alpha channel), "
             "context (keep original background), blurred (blur background), "
             "all (generate all three versions for mixed training)"
    )
    parser.add_argument(
        "--context-padding",
        type=int,
        default=20,
        help="Padding around instance bounding box for context versions"
    )

    args = parser.parse_args()

    # Determine output directory
    output_dir = args.output_dir
    if args.project:
        if not output_dir:
            # Auto-construct path based on project name
            base_dir = Path("/mnt/data/ai_data/datasets/3d-anime")
            output_dir = str(base_dir / args.project / "instances")
            print(f"✓ Using project: {args.project}")
            print(f"   Auto output: {output_dir}")
        else:
            print(f"✓ Project: {args.project} (output path manually specified)")
    elif not output_dir:
        parser.error("Either --output-dir or --project must be specified")

    # Process frames
    stats = process_frames_to_instances(
        input_dir=Path(args.input_dir),
        output_dir=Path(output_dir),
        model_type=args.model,
        device=args.device,
        min_instance_size=args.min_size,
        save_visualization=args.visualize,
        save_masks=args.save_masks,
        save_backgrounds=args.save_backgrounds,
        context_mode=args.context_mode,
        context_padding=args.context_padding
    )

    print(f"\n📁 Output saved to: {output_dir}")

    # Show instance directories based on context_mode
    if args.context_mode == "all":
        print(f"   Instances (transparent): {output_dir}/instances/")
        print(f"   Instances (context): {output_dir}/instances_context/")
        print(f"   Instances (blurred): {output_dir}/instances_blurred/")
    elif args.context_mode == "transparent":
        print(f"   Instances: {output_dir}/instances/")
    elif args.context_mode == "context":
        print(f"   Instances (context): {output_dir}/instances_context/")
    elif args.context_mode == "blurred":
        print(f"   Instances (blurred): {output_dir}/instances_blurred/")

    if args.save_masks:
        print(f"   Masks: {output_dir}/masks/")
    if args.save_backgrounds:
        print(f"   Backgrounds: {output_dir}/backgrounds/")
    if args.visualize:
        print(f"   Visualizations: {output_dir}/visualization/")

    if args.project:
        print(f"\n💡 Next steps for project '{args.project}':")
        print(f"   1. (Optional) Inpaint occluded instances:")
        print(f"      bash scripts/batch/run_inpainting_generic.sh")
        print(f"   2. Cluster by identity:")
        print(f"      python scripts/generic/clustering/face_identity_clustering.py \\")
        print(f"        {output_dir}/instances --output-dir .../clustered --project {args.project}")


if __name__ == "__main__":
    main()
