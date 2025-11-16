#!/usr/bin/env python3
"""
Compare inpainting quality between LaMa and OpenCV
Creates side-by-side comparison grids
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

def create_comparison_grid(opencv_dir: Path, lama_dir: Path, original_dir: Path, output_dir: Path, num_samples: int = 5):
    """
    Create side-by-side comparison grids
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get matching files
    opencv_files = sorted(list(opencv_dir.glob("*.jpg")))[:num_samples]

    if not opencv_files:
        print("❌ No OpenCV files found!")
        return

    print(f"📊 Creating comparison for {len(opencv_files)} images...")

    comparisons = []

    for opencv_path in opencv_files:
        filename = opencv_path.name
        lama_path = lama_dir / filename

        # Extract base name to find original and mask
        base_name = filename.replace("_background.jpg", "")
        original_path = original_dir / "backgrounds" / filename

        if not lama_path.exists() or not original_path.exists():
            print(f"⚠️  Skipping {filename} - missing files")
            continue

        # Load images
        original = cv2.imread(str(original_path))
        opencv_result = cv2.imread(str(opencv_path))
        lama_result = cv2.imread(str(lama_path))

        if original is None or opencv_result is None or lama_result is None:
            print(f"⚠️  Skipping {filename} - failed to load")
            continue

        # Find matching masks
        masks_dir = original_dir / "masks"
        mask_pattern = f"{base_name}_inst*_mask.png"
        mask_files = sorted(masks_dir.glob(mask_pattern))

        # Merge masks for visualization
        if mask_files:
            merged_mask = np.zeros(original.shape[:2], dtype=np.uint8)
            for mask_file in mask_files:
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    merged_mask = np.maximum(merged_mask, mask)

            # Create visualization overlay
            mask_viz = cv2.cvtColor(merged_mask, cv2.COLOR_GRAY2BGR)
            mask_viz[:, :, 1] = 0  # Remove green channel for red overlay
            original_with_mask = cv2.addWeighted(original, 0.7, mask_viz, 0.3, 0)
        else:
            original_with_mask = original.copy()
            merged_mask = np.zeros(original.shape[:2], dtype=np.uint8)

        # Resize to consistent height
        target_height = 512
        scale = target_height / original.shape[0]
        target_width = int(original.shape[1] * scale)

        original_resized = cv2.resize(original_with_mask, (target_width, target_height))
        opencv_resized = cv2.resize(opencv_result, (target_width, target_height))
        lama_resized = cv2.resize(lama_result, (target_width, target_height))

        # Create side-by-side comparison
        padding = 10
        label_height = 40

        # Create canvas
        canvas_width = target_width * 3 + padding * 4
        canvas_height = target_height + label_height + padding * 2
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        # Add labels
        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()

        # Add labels
        labels = ["Original + Mask", "OpenCV Inpaint", "LaMa Inpaint"]
        for i, label in enumerate(labels):
            x = padding + i * (target_width + padding) + target_width // 2
            bbox = draw.textbbox((x, label_height // 2), label, font=font, anchor="mm")
            draw.text((x, label_height // 2), label, fill=(0, 0, 0), font=font, anchor="mm")

        canvas = np.array(pil_canvas)

        # Place images
        y_offset = label_height + padding
        canvas[y_offset:y_offset+target_height, padding:padding+target_width] = original_resized
        canvas[y_offset:y_offset+target_height, padding*2+target_width:padding*2+target_width*2] = opencv_resized
        canvas[y_offset:y_offset+target_height, padding*3+target_width*2:padding*3+target_width*3] = lama_resized

        # Calculate quality metrics
        mask_coverage = (merged_mask > 127).sum() / merged_mask.size * 100

        # MSE between original and results (in inpainted regions)
        mask_binary = (merged_mask > 127)
        if mask_binary.sum() > 0:
            opencv_mse = np.mean((original[mask_binary] - opencv_result[mask_binary]) ** 2)
            lama_mse = np.mean((original[mask_binary] - lama_result[mask_binary]) ** 2)
        else:
            opencv_mse = 0
            lama_mse = 0

        # Add metrics text
        metrics_text = f"Mask Coverage: {mask_coverage:.1f}% | OpenCV MSE: {opencv_mse:.1f} | LaMa MSE: {lama_mse:.1f}"
        draw = ImageDraw.Draw(Image.fromarray(canvas))
        draw.text((canvas_width // 2, canvas_height - 15), metrics_text, fill=(100, 100, 100), font=font, anchor="mm")
        canvas = np.array(Image.fromarray(canvas))

        # Save individual comparison
        output_path = output_dir / f"comparison_{base_name}.jpg"
        cv2.imwrite(str(output_path), canvas)

        comparisons.append({
            "filename": filename,
            "mask_coverage": mask_coverage,
            "opencv_mse": float(opencv_mse),
            "lama_mse": float(lama_mse),
            "num_masks": len(mask_files)
        })

        print(f"✅ Created comparison: {output_path.name}")

    # Save metrics
    metrics_path = output_dir / "comparison_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            "comparisons": comparisons,
            "summary": {
                "avg_mask_coverage": np.mean([c["mask_coverage"] for c in comparisons]),
                "avg_opencv_mse": np.mean([c["opencv_mse"] for c in comparisons]),
                "avg_lama_mse": np.mean([c["lama_mse"] for c in comparisons]),
                "lama_improvement": (np.mean([c["opencv_mse"] for c in comparisons]) -
                                    np.mean([c["lama_mse"] for c in comparisons])) /
                                   np.mean([c["opencv_mse"] for c in comparisons]) * 100
            }
        }, f, indent=2)

    print(f"\n📄 Metrics saved: {metrics_path}")

    # Print summary
    print("\n" + "="*70)
    print("QUALITY COMPARISON SUMMARY")
    print("="*70)
    avg_opencv_mse = np.mean([c["opencv_mse"] for c in comparisons])
    avg_lama_mse = np.mean([c["lama_mse"] for c in comparisons])
    improvement = (avg_opencv_mse - avg_lama_mse) / avg_opencv_mse * 100

    print(f"Average Mask Coverage: {np.mean([c['mask_coverage'] for c in comparisons]):.1f}%")
    print(f"Average OpenCV MSE: {avg_opencv_mse:.1f}")
    print(f"Average LaMa MSE: {avg_lama_mse:.1f}")
    print(f"Quality Improvement: {improvement:.1f}% (lower MSE is better)")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Compare inpainting quality")
    parser.add_argument("--opencv-dir", type=str, required=True,
                       help="Directory with OpenCV results")
    parser.add_argument("--lama-dir", type=str, required=True,
                       help="Directory with LaMa results")
    parser.add_argument("--sam2-dir", type=str, required=True,
                       help="Original SAM2 directory (with backgrounds/ and masks/)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for comparisons")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to compare")

    args = parser.parse_args()

    create_comparison_grid(
        Path(args.opencv_dir),
        Path(args.lama_dir),
        Path(args.sam2_dir),
        Path(args.output_dir),
        args.num_samples
    )

if __name__ == "__main__":
    main()
