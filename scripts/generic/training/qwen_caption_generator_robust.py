#!/usr/bin/env python3
"""
Robust Qwen2-VL Caption Generator for Luca (2021) 3D Animation Characters

Enhanced version with CUDA error recovery and memory management.
Optimized for RTX 5080 + PyTorch 2.7.1 + CUDA 12.8
"""

import argparse
import json
import torch
import gc
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import warnings
import os
import traceback
warnings.filterwarnings('ignore')

# CUDA optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Set to '1' for debugging


# Luca film character information with detailed descriptions
CHARACTER_INFO = {
    'luca_human': {
        'name': 'Luca Paguro',
        'context': 'young sea monster boy in human form from Pixar\'s Luca (2021)',
        'appearance': 'brown curly hair, green eyes, light skin tone, striped shirt',
        'traits': 'curious, cautious, kind-hearted',
    },
    'alberto_human': {
        'name': 'Alberto Scorfano',
        'context': 'confident sea monster boy in human form from Pixar\'s Luca (2021)',
        'appearance': 'messy brown hair, tan skin, often shirtless or simple clothes',
        'traits': 'brave, carefree, loyal friend',
    },
    'guilia': {
        'name': 'Giulia Marcovaldo',
        'context': 'spirited Italian girl from Portorosso in Pixar\'s Luca (2021)',
        'appearance': 'red curly hair, freckles, round glasses, energetic',
        'traits': 'passionate, determined, friendly',
    },
    'alberto_seamonster': {
        'name': 'Alberto Scorfano',
        'context': 'sea monster form from Pixar\'s Luca (2021)',
        'appearance': 'red-orange scales, fish-like features, expressive eyes',
        'traits': 'confident sea creature',
    },
    'luca_seamonster': {
        'name': 'Luca Paguro',
        'context': 'sea monster form from Pixar\'s Luca (2021)',
        'appearance': 'teal and purple scales, large eyes, fish-like fins',
        'traits': 'gentle sea creature',
    },
    'massimo': {
        'name': 'Massimo Marcovaldo',
        'context': 'Giulia\'s father, fisherman from Pixar\'s Luca (2021)',
        'appearance': 'large build, black beard, tattoos, missing right arm',
        'traits': 'kind, protective, skilled fisherman',
    },
    'ercole': {
        'name': 'Ercole Visconti',
        'context': 'town bully from Portorosso in Pixar\'s Luca (2021)',
        'appearance': 'slicked dark hair, athletic, stylish 1960s Italian clothes',
        'traits': 'arrogant, competitive',
    },
}

# Default context for other characters
DEFAULT_CONTEXT = "character from Pixar's Luca (2021), set in 1960s Italian Riviera"


def get_character_prompt(cluster_name: str) -> str:
    """Generate Qwen2-VL prompt for character"""
    info = CHARACTER_INFO.get(cluster_name, {})

    if info:
        name = info['name']
        context = info['context']
        appearance = info.get('appearance', '')

        prompt = f"""Describe this 3D animated character image from {context}.

This is {name}. {appearance}

Provide a detailed caption focusing on:
1. Character's pose and expression
2. Visible clothing and details
3. Lighting and rendering style (Pixar 3D animation)
4. Background elements if present

Keep the caption concise (40-60 words) and suitable for image training."""
    else:
        prompt = f"""Describe this 3D animated character from Pixar's Luca (2021).

Provide a detailed caption focusing on:
1. Character's appearance and pose
2. Expression and mood
3. Clothing and visual details
4. Pixar's 3D rendering style

Keep the caption concise (40-60 words) and suitable for image training."""

    return prompt


class Qwen2VLCaptionGenerator:
    """Generate captions using Qwen2-VL for Luca characters with robust error handling"""

    def __init__(self, device: str = 'cuda', model_size: str = '7B'):
        """Initialize Qwen2-VL model"""
        self.device = device
        self.model_size = model_size
        self.processed_count = 0
        self.cache_clear_interval = 50  # Clear cache every N images

        # Use local model path from AI warehouse
        MODEL_PATH = f"/mnt/c/AI_LLM_projects/ai_warehouse/models/vlm/Qwen2-VL-{model_size}-Instruct"

        print(f"Loading Qwen2-VL-{model_size}-Instruct model from {MODEL_PATH}...")

        # Load model with 8-bit quantization (CUDA 12.8 compatible)
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True) if device == 'cuda' else None

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            quantization_config=quantization_config,
            device_map="auto" if device == 'cuda' else None,  # Auto device placement
            local_files_only=True,
        )

        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)

        # NOTE: torch.compile disabled due to RTX 5080 compatibility issues
        # if device == 'cuda':
        #     self.model = torch.compile(self.model, mode="reduce-overhead")

        print(f"✓ Qwen2-VL-{model_size} loaded on {device}")
        self._clear_cuda_cache()

    def _clear_cuda_cache(self):
        """Clear CUDA cache and synchronize"""
        if self.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

    def generate_caption(self, image_path: Path, cluster_name: str, retry_count: int = 3) -> str:
        """Generate caption for a single image with retry logic"""

        for attempt in range(retry_count):
            try:
                # Get character-specific prompt
                prompt = get_character_prompt(cluster_name)

                # Prepare messages for Qwen2-VL
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": str(image_path),
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                # Process inputs
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.device)

                # Generate caption
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,  # Deterministic for stability
                    )

                # Synchronize before processing output
                if self.device == 'cuda':
                    torch.cuda.synchronize()

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                caption = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                # Post-process
                caption = self._post_process(caption, cluster_name)

                # Clear cache periodically
                self.processed_count += 1
                if self.processed_count % self.cache_clear_interval == 0:
                    self._clear_cuda_cache()

                return caption

            except RuntimeError as e:
                error_msg = str(e)
                if "CUDA" in error_msg or "out of memory" in error_msg:
                    print(f"\n⚠️  CUDA error on attempt {attempt + 1}/{retry_count}: {error_msg}")

                    # Aggressive cleanup
                    self._clear_cuda_cache()

                    if attempt < retry_count - 1:
                        print(f"   Retrying...")
                        continue
                    else:
                        raise Exception(f"Failed after {retry_count} attempts: {error_msg}")
                else:
                    raise
            except Exception as e:
                print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")
                if attempt < retry_count - 1:
                    self._clear_cuda_cache()
                    continue
                else:
                    raise

    def _post_process(self, caption: str, cluster_name: str) -> str:
        """Post-process caption for training"""

        # Add Pixar/Luca prefix
        info = CHARACTER_INFO.get(cluster_name, {})
        if info:
            char_name = info['name']
            prefix = f"a 3d animated character, {char_name} from Pixar Luca, "
        else:
            prefix = "a 3d animated character from Pixar Luca, "

        # Clean and format
        caption = caption.strip()
        if not caption.endswith('.'):
            caption += '.'

        final_caption = prefix + caption

        # Limit length (SD 1.5 limit ~77 tokens)
        words = final_caption.split()
        if len(words) > 75:
            final_caption = ' '.join(words[:75]) + '.'

        return final_caption

    def process_cluster(self, cluster_dir: Path, output_dir: Path, cluster_name: str) -> dict:
        """Process all images in a cluster"""

        # Find all images (check both cluster_dir and cluster_dir/images)
        image_paths = []
        images_dir = cluster_dir / "images"
        search_dir = images_dir if images_dir.exists() else cluster_dir

        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(search_dir.glob(ext))

        if len(image_paths) == 0:
            return {
                "cluster": cluster_name,
                "total": 0,
                "processed": 0,
                "failed": 0,
            }

        # Create output directories
        images_dir = output_dir / cluster_name / "images"
        captions_dir = output_dir / cluster_name / "captions"
        images_dir.mkdir(parents=True, exist_ok=True)
        captions_dir.mkdir(parents=True, exist_ok=True)

        # Process images
        processed = 0
        failed = 0
        failed_images = []
        captions_list = []

        for img_path in tqdm(image_paths, desc=f"Captioning {cluster_name}"):
            try:
                # Prepare output paths
                output_img_path = images_dir / img_path.name
                output_txt_path = captions_dir / f"{img_path.stem}.txt"

                # Skip if both image and caption already exist (resume support)
                if output_img_path.exists() and output_txt_path.exists():
                    # Load existing caption for manifest
                    with open(output_txt_path, 'r', encoding='utf-8') as f:
                        caption = f.read()
                    processed += 1
                    captions_list.append({
                        "filename": img_path.name,
                        "caption": caption,
                    })
                    continue

                # Generate caption
                caption = self.generate_caption(img_path, cluster_name)

                # Copy image
                if not output_img_path.exists():
                    import shutil
                    shutil.copy2(img_path, output_img_path)

                # Write caption
                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write(caption)

                captions_list.append({
                    "filename": img_path.name,
                    "caption": caption,
                })

                processed += 1

            except Exception as e:
                failed += 1
                failed_images.append(img_path.name)
                print(f"✗ Error processing {img_path.name}: {e}")

                # Log detailed error for debugging
                with open(output_dir / f"errors_{cluster_name}.log", 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Image: {img_path.name}\n")
                    f.write(f"Error: {e}\n")
                    f.write(f"Traceback:\n{traceback.format_exc()}\n")

                continue

        # Save manifest
        manifest_path = output_dir / cluster_name / "captions_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump({
                "cluster": cluster_name,
                "character": CHARACTER_INFO.get(cluster_name, {}).get('name', cluster_name),
                "total_images": len(image_paths),
                "captions": captions_list[:10],  # Sample
                "failed_images": failed_images[:20],  # Sample
            }, f, indent=2, ensure_ascii=False)

        return {
            "cluster": cluster_name,
            "total": len(image_paths),
            "processed": processed,
            "failed": failed,
        }


def main():
    parser = argparse.ArgumentParser(description="Robust Qwen2-VL caption generator with CUDA error recovery")
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--model-size', type=str, default='7B', choices=['2B', '7B'])
    parser.add_argument('--clusters', type=str, nargs='+', help='Specific clusters to process')

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"LUCA CAPTION GENERATION (Qwen2-VL-{args.model_size} ROBUST)")
    print("=" * 70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Model:  Qwen2-VL-{args.model_size}-Instruct")
    print("=" * 70)
    print()

    # Initialize generator
    generator = Qwen2VLCaptionGenerator(device=args.device, model_size=args.model_size)

    # Find clusters
    SKIP_DIRS = {'noise', '__pycache__', '.git', '.DS_Store', 'enhancement_report.json'}
    cluster_dirs = [
        d for d in args.input_dir.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith('.')
    ]

    if args.clusters:
        cluster_dirs = [d for d in cluster_dirs if d.name in args.clusters]

    cluster_dirs = sorted(cluster_dirs, key=lambda x: x.name)

    print(f"Found {len(cluster_dirs)} clusters:")
    for d in cluster_dirs:
        # Check both direct and images/ subdirectory
        images_dir = d / "images"
        search_dir = images_dir if images_dir.exists() else d
        count = len(list(search_dir.glob("*.png")))
        print(f"   - {d.name}: {count} images")
    print()

    # Process each cluster
    all_stats = []
    total_processed = 0
    total_failed = 0

    for cluster_dir in cluster_dirs:
        print(f"\n{'='*60}")
        print(f"Processing {cluster_dir.name}")
        print(f"{'='*60}")

        stats = generator.process_cluster(cluster_dir, args.output_dir, cluster_dir.name)

        all_stats.append(stats)
        total_processed += stats["processed"]
        total_failed += stats["failed"]

        print(f"   Processed: {stats['processed']}/{stats['total']}")
        if stats['failed'] > 0:
            print(f"   ⚠️  Failed: {stats['failed']}")

    # Save report
    report = {
        "model": f"Qwen2-VL-{args.model_size}-Instruct",
        "device": args.device,
        "film": "Luca (2021) Pixar",
        "summary": {
            "total_clusters": len(cluster_dirs),
            "total_captions": total_processed,
            "total_failed": total_failed,
        },
        "clusters": all_stats,
    }

    report_path = args.output_dir / "caption_generation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print("CAPTION GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total captions generated: {total_processed}")
    print(f"Total failed: {total_failed}")
    print(f"Report saved: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
