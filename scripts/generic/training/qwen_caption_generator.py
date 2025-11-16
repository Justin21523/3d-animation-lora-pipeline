#!/usr/bin/env python3
"""
Qwen2-VL Caption Generator for Luca (2021) 3D Animation Characters

Generates detailed, film-specific captions using Qwen2-VL-2B.
Optimized for Pixar's Luca with Italian Riviera context.
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings('ignore')


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
    """Generate captions using Qwen2-VL for Luca characters"""

    def __init__(self, device: str = 'cuda', batch_size: int = 4):
        """Initialize Qwen2-VL model"""
        self.device = device
        self.batch_size = batch_size

        # Use local model path from AI warehouse (7B for better quality)
        MODEL_PATH = "/mnt/c/AI_LLM_projects/ai_warehouse/models/vlm/Qwen2-VL-7B-Instruct"

        print(f"Loading Qwen2-VL-7B-Instruct model from {MODEL_PATH}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            device_map="auto" if device == 'cuda' else None,
            local_files_only=True,  # Ensure only local files are used
        )

        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)

        print(f"✓ Qwen2-VL loaded on {device}")

    def generate_caption(self, image_path: Path, cluster_name: str) -> str:
        """Generate caption for a single image"""

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
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        caption = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Post-process
        caption = self._post_process(caption, cluster_name)

        return caption

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

        # Find all images
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(cluster_dir.glob(ext))

        if len(image_paths) == 0:
            return {
                "cluster": cluster_name,
                "total": 0,
                "processed": 0,
            }

        # Create output directories
        images_dir = output_dir / cluster_name / "images"
        captions_dir = output_dir / cluster_name / "captions"
        images_dir.mkdir(parents=True, exist_ok=True)
        captions_dir.mkdir(parents=True, exist_ok=True)

        # Process images
        processed = 0
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
                print(f"✗ Error processing {img_path.name}: {e}")
                continue

        # Save manifest
        manifest_path = output_dir / cluster_name / "captions_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump({
                "cluster": cluster_name,
                "character": CHARACTER_INFO.get(cluster_name, {}).get('name', cluster_name),
                "total_images": len(image_paths),
                "captions": captions_list[:10],  # Sample
            }, f, indent=2, ensure_ascii=False)

        return {
            "cluster": cluster_name,
            "total": len(image_paths),
            "processed": processed,
        }


def main():
    parser = argparse.ArgumentParser(description="Generate captions using Qwen2-VL for Luca characters")
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--clusters', type=str, nargs='+', help='Specific clusters to process')

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LUCA CAPTION GENERATION (Qwen2-VL)")
    print("=" * 70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 70)
    print()

    # Initialize generator
    generator = Qwen2VLCaptionGenerator(device=args.device, batch_size=args.batch_size)

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
        count = len(list(d.glob("*.png")))
        print(f"   - {d.name}: {count} images")
    print()

    # Process each cluster
    all_stats = []
    total_processed = 0

    for cluster_dir in cluster_dirs:
        print(f"\n{'='*60}")
        print(f"Processing {cluster_dir.name}")
        print(f"{'='*60}")

        stats = generator.process_cluster(cluster_dir, args.output_dir, cluster_dir.name)

        all_stats.append(stats)
        total_processed += stats["processed"]

        print(f"   Processed: {stats['processed']}/{stats['total']}")

    # Save report
    report = {
        "model": "Qwen2-VL-2B-Instruct",
        "device": args.device,
        "film": "Luca (2021) Pixar",
        "summary": {
            "total_clusters": len(cluster_dirs),
            "total_captions": total_processed,
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
    print(f"Report saved: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
