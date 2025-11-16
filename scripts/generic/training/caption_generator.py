#!/usr/bin/env python3
"""
Caption Generator for 3D Animation Character Instances (Luca Film)

Generates training captions for character images using Qwen2-VL or BLIP2.
Optimized for Luca (2021 Pixar film) with Italian Riviera setting.

Usage:
    python scripts/generic/training/caption_generator.py \
        --input-dir /path/to/clustered_enhanced \
        --output-dir /path/to/training_data \
        --model qwen2_vl \
        --device cuda
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# Luca film character information
CHARACTER_INFO = {
    'luca_human': {
        'name': 'Luca Paguro',
        'description': 'a young sea monster in human form, curious and cautious',
        'appearance': 'brown curly hair, green eyes, light skin, usually wearing striped shirt',
    },
    'alberto_human': {
        'name': 'Alberto Scorfano',
        'description': 'a confident sea monster in human form, brave and carefree',
        'appearance': 'wild brown hair, tan skin, often shirtless or in simple clothes',
    },
    'guilia': {
        'name': 'Giulia Marcovaldo',
        'description': 'a spirited Italian girl, passionate about winning',
        'appearance': 'red curly hair, freckles, glasses, energetic expression',
    },
    'massimo': {
        'name': 'Massimo Marcovaldo',
        'description': 'Giulia\'s father, a kind fisherman with one arm',
        'appearance': 'large muscular build, black beard, tattoos, missing right arm',
    },
    'ercole': {
        'name': 'Ercole Visconti',
        'description': 'the town bully, arrogant and competitive',
        'appearance': 'slicked back dark hair, athletic build, stylish clothes',
    },
    'luca_seamonster': {
        'name': 'Luca Paguro in sea monster form',
        'description': 'a young sea monster with blue-green scales',
        'appearance': 'teal and purple scales, large expressive eyes, fish-like features',
    },
    'alberto_seamonster': {
        'name': 'Alberto Scorfano in sea monster form',
        'description': 'a sea monster with reddish-brown scales',
        'appearance': 'red-orange scales, confident stance, fish-like features',
    },
    'dad_human': {
        'name': 'Lorenzo Paguro',
        'description': 'Luca\'s cautious father in human form',
        'appearance': 'middle-aged, brown hair, worried expression',
    },
    'mom_human': {
        'name': 'Daniela Paguro',
        'description': 'Luca\'s protective mother in human form',
        'appearance': 'brown hair, determined expression, caring demeanor',
    },
    'dad_seamonster': {
        'name': 'Lorenzo Paguro in sea monster form',
        'description': 'Luca\'s father as a sea monster',
        'appearance': 'adult sea monster with teal scales, worried expression',
    },
    'mom_seamonster': {
        'name': 'Daniela Paguro in sea monster form',
        'description': 'Luca\'s mother as a sea monster',
        'appearance': 'adult sea monster with blue-green scales, protective stance',
    },
    'grandma_seamonster': {
        'name': 'Grandma Paguro in sea monster form',
        'description': 'wise elderly sea monster',
        'appearance': 'aged sea monster with purple-teal scales',
    },
    'cicco': {
        'name': 'Ciccio',
        'description': 'one of Ercole\'s followers',
        'appearance': 'stocky build, simple clothes',
    },
    'guido': {
        'name': 'Guido',
        'description': 'one of Ercole\'s followers',
        'appearance': 'thin build, simple clothes',
    },
    'kids': {
        'name': 'child character',
        'description': 'a young person from Portorosso',
        'appearance': 'child-like proportions, 1960s Italian summer clothing',
    },
    'adults': {
        'name': 'adult character',
        'description': 'a resident of Portorosso',
        'appearance': '1960s Italian coastal town clothing',
    },
    'cat': {
        'name': 'Machiavelli',
        'description': 'Massimo\'s one-eyed cat',
        'appearance': 'grey tabby cat, missing one eye, grumpy expression',
    },
    'fish': {
        'name': 'fish',
        'description': 'sea creature',
        'appearance': 'stylized cartoon fish',
    },
}

# Luca film context for prompts
LUCA_CONTEXT = """This is from Luca (2021), a Pixar animated film set in the Italian Riviera during the 1960s summer.
The story features sea monsters who can transform into humans when dry.
Visual style: smooth 3D rendering, warm Mediterranean lighting, vibrant colors, Pixar's signature character design."""


class CaptionGenerator:
    """Generate captions for 3D animated character images"""

    def __init__(
        self,
        model_name: str = 'blip2',
        device: str = 'cuda',
        max_length: int = 77,
        add_3d_prefix: bool = True,
    ):
        """
        Initialize caption generator

        Args:
            model_name: Model to use ('blip' or 'blip2')
            device: Device to use ('cuda' or 'cpu')
            max_length: Maximum caption length in tokens
            add_3d_prefix: Add 3D animation style prefix
        """
        self.device = device
        self.max_length = max_length
        self.add_3d_prefix = add_3d_prefix

        print(f"Loading {model_name} model...")

        if model_name == 'blip2':
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32
            )
        else:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large",
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32
            )

        self.model.to(device)
        self.model.eval()
        print(f"✓ Model loaded on {device}")

    def generate_caption(
        self,
        image_path: Path,
        character_name: str = None,
    ) -> str:
        """
        Generate caption for a single image

        Args:
            image_path: Path to image
            character_name: Name of the character (optional)

        Returns:
            Generated caption string
        """
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Prepare prompt
        if character_name:
            prompt = f"Question: Describe this 3D animated character. Answer: This is {character_name},"
        else:
            prompt = "Question: Describe this 3D animated character. Answer:"

        # Generate caption
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=3,
                temperature=0.7,
                do_sample=True,
            )

        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Post-process caption
        caption = self._post_process_caption(caption, character_name)

        return caption

    def _post_process_caption(self, caption: str, character_name: str = None) -> str:
        """Post-process generated caption for 3D animation training"""

        # Remove prompt remnants
        caption = caption.replace("Question:", "").replace("Answer:", "").strip()

        # Add 3D animation prefix
        if self.add_3d_prefix:
            prefix = "a 3d animated character, pixar style, smooth shading, "

            # Add character name if provided
            if character_name:
                caption = f"{prefix}{character_name}, {caption}"
            else:
                caption = f"{prefix}{caption}"

        # Clean up
        caption = caption.replace("  ", " ").strip()

        # Ensure it ends with period
        if not caption.endswith('.'):
            caption += '.'

        return caption

    def generate_captions_for_cluster(
        self,
        cluster_dir: Path,
        output_dir: Path,
        cluster_name: str,
    ) -> dict:
        """
        Generate captions for all images in a cluster

        Args:
            cluster_dir: Input cluster directory
            output_dir: Output directory for images and captions
            cluster_name: Name of the cluster

        Returns:
            Statistics dictionary
        """
        # Get character name
        character_name = CHARACTER_NAMES.get(cluster_name, cluster_name)

        # Find all images
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(cluster_dir.glob(ext))

        if len(image_paths) == 0:
            return {
                "cluster": cluster_name,
                "character": character_name,
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
                # Generate caption
                caption = self.generate_caption(img_path, character_name)

                # Save image and caption
                output_img_path = images_dir / img_path.name
                output_txt_path = captions_dir / f"{img_path.stem}.txt"

                # Copy image if not already in output dir
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

        # Save captions manifest
        manifest_path = output_dir / cluster_name / "captions_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump({
                "cluster": cluster_name,
                "character": character_name,
                "total_images": len(image_paths),
                "captions": captions_list,
            }, f, indent=2, ensure_ascii=False)

        return {
            "cluster": cluster_name,
            "character": character_name,
            "total": len(image_paths),
            "processed": processed,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions for 3D animation character instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Input directory containing cluster subdirectories'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for training data'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='blip2',
        choices=['blip', 'blip2'],
        help='Caption model to use'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=77,
        help='Maximum caption length in tokens'
    )

    parser.add_argument(
        '--no-3d-prefix',
        action='store_true',
        help='Do not add 3D animation style prefix'
    )

    parser.add_argument(
        '--clusters',
        type=str,
        nargs='+',
        help='Specific clusters to process (default: all)'
    )

    args = parser.parse_args()

    # Validate paths
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CAPTION GENERATION")
    print("=" * 70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Model:  {args.model}")
    print(f"Device: {args.device}")
    print(f"Max length: {args.max_length} tokens")
    print(f"3D prefix: {not args.no_3d_prefix}")
    print("=" * 70)
    print()

    # Initialize generator
    generator = CaptionGenerator(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        add_3d_prefix=not args.no_3d_prefix,
    )

    # Find all cluster directories
    SKIP_DIRS = {'noise', '__pycache__', '.git', '.DS_Store'}
    cluster_dirs = [
        d for d in args.input_dir.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith('.')
    ]

    # Filter clusters if specified
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

        stats = generator.generate_captions_for_cluster(
            cluster_dir,
            args.output_dir,
            cluster_dir.name,
        )

        all_stats.append(stats)
        total_processed += stats["processed"]

        print(f"   Processed: {stats['processed']}/{stats['total']}")

    # Save summary report
    report = {
        "model": args.model,
        "device": args.device,
        "max_length": args.max_length,
        "add_3d_prefix": not args.no_3d_prefix,
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
