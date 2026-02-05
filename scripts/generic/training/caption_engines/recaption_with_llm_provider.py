#!/usr/bin/env python3
"""
Re-caption dataset images using LLMProvider API for precise, individualized captions.

This script replaces template-based captions with AI-generated captions that
capture the unique details of each image (pose, angle, lighting, expression).

Usage:
    python recaption_with_llm_provider.py \
        --input-dir /path/to/dataset \
        --output-dir /path/to/output \
        --lora-type pose \
        --batch-size 50 \
        --workers 4
"""

import os
import json
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import llm_vendor
from PIL import Image

# Caption generation prompts by LoRA type
CAPTION_PROMPTS = {
    "pose": """Analyze this 3D animated character image and describe it for training a pose LoRA.

Focus on:
1. **Primary Pose**: (standing/sitting/crouching/lying/leaning/etc.) - be specific
2. **Body Position**: (upright/bent/twisted/relaxed/tense)
3. **Limb Details**:
   - Arms: (raised/lowered/crossed/at sides/gesturing)
   - Legs: (together/apart/crossed/bent/straight)
   - Hands: (open/closed/resting/pointing)
4. **Balance & Weight**: (centered/shifted to one side/forward lean)
5. **View Angle**: (front/three-quarter/side/back/angled)
6. **Camera Height**: (eye-level/low-angle/high-angle/slightly above)

Generate a natural, flowing caption (75-150 tokens) that includes:
- Trigger phrase: "a 3d animated character"
- Detailed pose description
- View and camera information
- Lighting quality (if notable)

Example: "a 3d animated character, sitting cross-legged with back straight, hands resting palms-down on knees, shoulders relaxed, three-quarter front view from slightly above, soft diffused studio lighting"

Caption:""",

    "action": """Analyze this 3D animated character image and describe it for training an action LoRA.

Focus on:
1. **Primary Action**: (walking/running/jumping/reaching/climbing/pushing/pulling/etc.)
2. **Motion Phase**: (starting/mid-motion/landing/follow-through)
3. **Body Dynamics**:
   - Energy level: (explosive/controlled/gentle/forceful)
   - Balance: (stable/off-balance/transitioning)
   - Momentum: (accelerating/steady/decelerating)
4. **Limb Coordination**:
   - Arms: (swinging/reaching/pulling/pushing)
   - Legs: (striding/kicking/planted/airborne)
5. **Facial Expression**: (focused/strained/relaxed/excited) if visible
6. **View Angle**: (front/side/three-quarter/back)

Generate a natural, flowing caption (75-150 tokens) that includes:
- Trigger phrase: "a 3d animated character"
- Detailed action description
- Motion quality and energy
- View angle
- Environmental context (if any)

Example: "a 3d animated character, mid-stride walking forward with left leg extended, right arm swinging forward naturally, body upright with slight forward lean, focused expression, side view, even outdoor lighting"

Caption:""",

    "expression": """Analyze this 3D animated character's face and describe it for training an expression LoRA.

Focus on:
1. **Primary Emotion**: (happy/sad/angry/surprised/fearful/disgusted/neutral/mixed)
2. **Facial Features**:
   - Eyes: (wide/narrowed/closed/looking up/down/aside, eyebrows raised/furrowed/relaxed)
   - Mouth: (smiling/frowning/open/closed/pursed/grimacing)
   - Cheeks: (raised/neutral/sunken)
   - Forehead: (smooth/wrinkled/tense)
3. **Intensity**: (subtle/moderate/extreme)
4. **Head Orientation**: (facing forward/tilted/turned)
5. **Gaze Direction**: (at camera/away/down/up/aside)
6. **View Angle**: (close-up/medium/three-quarter/profile)

Generate a natural, flowing caption (75-150 tokens) that includes:
- Trigger phrase: "a 3d animated character"
- Primary emotion
- Detailed facial feature description
- Expression intensity
- Head position and gaze
- View angle and framing

Example: "a 3d animated character, joyful expression with wide genuine smile, eyes slightly squinted with visible crow's feet, eyebrows raised, cheeks lifted, head tilted slightly to the right, looking directly at camera with warm gaze, close-up three-quarter view, soft frontal lighting"

Caption:""",

    "character": """Analyze this 3D animated character image and generate ONLY the training caption. Do NOT include any explanations, introductions, or additional text.

CHARACTER NAME: {character_name}

Focus on describing this specific character:
1. **Character Identity**: Start with "{character_name}," to establish the trigger word
2. **Primary Colors**: Body paint colors, accent colors, trim details
3. **Geometric Shapes**: Overall silhouette, body proportions, distinctive contours
4. **Distinctive Features**: Unique visual elements (decals, markings, emblems, lights, accessories)
5. **Materials & Shading**:
   - Body surface: (glossy enamel/matte paint/metallic/semi-gloss)
   - Trim/details: (chrome/plastic/rubber/glass)
   - Overall material quality: (toy-like CGI/realistic/stylized)
6. **Mode/Form**: For transforming characters, specify the current form (vehicle_mode/robot_mode/hybrid)
7. **View Angle**: (front/side/three-quarter/back/top-down)
8. **Lighting & Environment**: Studio lighting setup, background context

IMPORTANT:
- Return ONLY a natural, flowing caption (75-150 words) without any prefix like "Caption:" or explanatory text
- MUST start with "{character_name}, a 3d animated character"

Style guide:
- Start with "{character_name}, a 3d animated character"
- Use "toy-like CGI", "smooth shading" for 3D style
- Emphasize "glossy clearcoat", "semi-gloss plastic" for materials
- Include specific color names (e.g., "red and white jet plane" not just "colorful")
- Mention distinctive features (e.g., "four signal lights", "blue eyes", specific decals)

Example output (return EXACTLY this format):
jett, a 3d animated character, small red and white jet plane with blue eyes, compact sporty jet proportions with rounded friendly nose, glossy clearcoat red body paint with white trim, four signal lights on wings, semi-gloss plastic joints, vehicle mode, three-quarter front view, studio lighting, neutral background"""
}


class LLMProviderCaptioner:
    """Generate individualized captions using LLMProvider API."""

    def __init__(self, api_key: str, model: str = "llm_provider-3-5-haiku-20241022"):
        self.client = llm_vendor.LLMVendor(api_key=api_key)
        self.model = model
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0
        }

    def generate_caption(
        self,
        image_path: Path,
        lora_type: str,
        character_name: str = "",
        max_retries: int = 3
    ) -> Tuple[str, bool]:
        """
        Generate caption for a single image.

        Args:
            image_path: Path to image file
            lora_type: Type of LoRA (character, pose, action, expression)
            character_name: Character name for character LoRAs (optional)
            max_retries: Number of retries on failure

        Returns:
            (caption, success)
        """
        try:
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()

            # Determine image type
            ext = image_path.suffix.lower()
            media_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp"
            }.get(ext, "image/png")

            # Get prompt for this LoRA type
            prompt = CAPTION_PROMPTS.get(lora_type, CAPTION_PROMPTS["pose"])

            # Format prompt with character name if provided
            if character_name and lora_type == "character":
                prompt = prompt.format(character_name=character_name)

            # Call LLMProvider API with retries
            for attempt in range(max_retries):
                try:
                    message = self.client.messages.create(
                        model=self.model,
                        max_tokens=300,
                        temperature=0.7,  # Slight variation for diversity
                        messages=[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_data
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }]
                    )

                    # Extract caption
                    caption = message.content[0].text.strip()

                    # Clean up caption - handle various prefixes and formats
                    # Remove common prefixes
                    caption = self._extract_caption(caption)

                    # Validate caption
                    if self._validate_caption(caption):
                        self.stats["success"] += 1
                        return caption, True
                    else:
                        if attempt < max_retries - 1:
                            continue
                        else:
                            print(f"⚠️  Invalid caption after {max_retries} attempts: {image_path.name}")
                            self.stats["failed"] += 1
                            return "", False

                except llm_vendor.RateLimitError:
                    import time
                    wait_time = (attempt + 1) * 5
                    print(f"⏳ Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                except Exception as e:
                    if attempt < max_retries - 1:
                        continue
                    else:
                        print(f"❌ Error processing {image_path.name}: {e}")
                        self.stats["failed"] += 1
                        return "", False

            self.stats["failed"] += 1
            return "", False

        except Exception as e:
            print(f"❌ Fatal error processing {image_path.name}: {e}")
            self.stats["failed"] += 1
            return "", False

    def _extract_caption(self, text: str) -> str:
        """
        Extract the actual caption from LLMProvider's response.
        Handles various formats and prefixes.
        """
        # Common prefixes to remove
        prefixes = [
            "caption:",
            "here's a description",
            "here is a description",
            "recommended lora training caption:",
            "training caption:",
        ]

        lines = text.split('\n')

        # Try to find the caption line
        caption_lines = []
        found_caption = False

        for line in lines:
            line_lower = line.lower().strip()

            # Skip empty lines
            if not line_lower:
                continue

            # Check if this is a prefix line
            is_prefix = any(line_lower.startswith(prefix) for prefix in prefixes)

            if is_prefix:
                # Extract text after prefix
                for prefix in prefixes:
                    if line_lower.startswith(prefix):
                        remaining = line[len(prefix):].strip()
                        if remaining:
                            caption_lines.append(remaining)
                        found_caption = True
                        break
            elif found_caption or line_lower.startswith("a 3d animated"):
                # This is caption content
                caption_lines.append(line.strip())
                found_caption = True
            elif not found_caption and "3d animated" in line_lower:
                # Likely the caption
                caption_lines.append(line.strip())
                found_caption = True

        # If we found caption lines, join them
        if caption_lines:
            caption = ' '.join(caption_lines)
            # Remove any remaining prefix
            for prefix in prefixes:
                if caption.lower().startswith(prefix):
                    caption = caption[len(prefix):].strip()
            return caption

        # Fallback: return the original text stripped
        return text.strip()

    def _validate_caption(self, caption: str) -> bool:
        """Validate caption meets requirements."""
        if not caption:
            return False

        # Check length (more lenient: 20-250 words)
        word_count = len(caption.split())
        if word_count < 20 or word_count > 250:
            return False

        # Check for trigger phrase (case-insensitive)
        if "3d animated character" not in caption.lower():
            return False

        return True

    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()


def process_single_image(
    args: Tuple[Path, Path, str, str, bool]
) -> Tuple[Path, str, bool]:
    """
    Process a single image (for parallel execution).

    Args:
        (image_path, output_dir, lora_type, api_key, force)

    Returns:
        (image_path, caption, success)
    """
    image_path, output_dir, lora_type, api_key, force = args

    # Check if caption already exists
    caption_path = output_dir / image_path.with_suffix(".txt").name
    if caption_path.exists() and not force:
        return image_path, None, False  # Skipped

    # Generate caption
    captioner = LLMProviderCaptioner(api_key)
    caption, success = captioner.generate_caption(image_path, lora_type)

    return image_path, caption, success


def main():
    parser = argparse.ArgumentParser(
        description="Re-caption dataset with LLMProvider API for precise descriptions"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing images and old captions"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for new captions (images will be copied/linked)"
    )
    parser.add_argument(
        "--lora-type",
        choices=["pose", "action", "expression", "character"],
        required=True,
        help="Type of LoRA (determines caption focus)"
    )
    parser.add_argument(
        "--character-name",
        type=str,
        default="",
        help="Character name for character LoRAs (e.g., 'jett', 'jerome')"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of images to process in parallel"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing captions"
    )
    parser.add_argument(
        "--backup-original",
        action="store_true",
        help="Backup original captions to .txt.bak files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )

    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("LLM_VENDOR_API_KEY")
    if not api_key:
        print("❌ Error: LLM_VENDOR_API_KEY environment variable not set")
        return 1

    # Validate input directory
    if not args.input_dir.exists():
        print(f"❌ Error: Input directory not found: {args.input_dir}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = [
        f for f in args.input_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"❌ Error: No images found in {args.input_dir}")
        return 1

    print(f"\n{'='*60}")
    print(f"🎨 LLMProvider Caption Generator")
    print(f"{'='*60}")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA Type: {args.lora_type}")
    print(f"Images: {len(image_files)}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Workers: {args.workers}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("🔍 DRY RUN - No changes will be made")
        print(f"Would process {len(image_files)} images")
        return 0

    # Backup original captions if requested
    if args.backup_original:
        print("💾 Backing up original captions...")
        for img_file in image_files:
            old_caption = img_file.with_suffix(".txt")
            if old_caption.exists():
                backup_path = args.output_dir / f"{old_caption.name}.bak"
                import shutil
                shutil.copy2(old_caption, backup_path)
        print(f"✅ Backed up {len(image_files)} captions\n")

    # Initialize captioner
    captioner = LLMProviderCaptioner(api_key)

    # Process images with progress bar
    print(f"🚀 Processing {len(image_files)} images...")
    print(f"⏱️  Estimated time: {len(image_files) * 2 / args.workers / 60:.1f} minutes\n")

    successful = 0
    failed = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_image = {}
        for img_file in image_files:
            # Check if already exists
            caption_path = args.output_dir / img_file.with_suffix(".txt").name
            if caption_path.exists() and not args.force:
                skipped += 1
                continue

            future = executor.submit(
                captioner.generate_caption,
                img_file,
                args.lora_type,
                args.character_name
            )
            future_to_image[future] = img_file

        # Process results with progress bar
        with tqdm(total=len(future_to_image), desc="Generating captions") as pbar:
            for future in as_completed(future_to_image):
                img_file = future_to_image[future]
                try:
                    caption, success = future.result()  # Returns only 2 values: (caption, success)
                except Exception as e:
                    print(f"\n❌ Error processing {img_file.name}: {e}")
                    failed += 1
                    pbar.update(1)
                    continue

                if success:
                    # Save caption
                    caption_path = args.output_dir / img_file.with_suffix(".txt").name
                    caption_path.write_text(caption, encoding="utf-8")

                    # Copy/link image if output dir is different
                    if args.output_dir != args.input_dir:
                        out_img = args.output_dir / img_file.name
                        if not out_img.exists():
                            try:
                                os.link(img_file, out_img)  # Hard link (save space)
                            except:
                                import shutil
                                shutil.copy2(img_file, out_img)

                    successful += 1
                else:
                    failed += 1

                pbar.update(1)

    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ Caption Generation Complete")
    print(f"{'='*60}")
    print(f"Total:     {len(image_files)}")
    print(f"Success:   {successful} ✓")
    print(f"Failed:    {failed} ✗")
    print(f"Skipped:   {skipped} (already exist)")
    print(f"{'='*60}\n")

    if failed > 0:
        print(f"⚠️  {failed} images failed. Check logs above for details.")
        return 1

    print(f"🎉 All captions saved to: {args.output_dir}")

    # Generate report
    report_path = args.output_dir / "recaption_report.json"
    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "lora_type": args.lora_type,
        "total_images": len(image_files),
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "batch_size": args.batch_size,
        "workers": args.workers
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"📊 Report saved to: {report_path}\n")

    return 0


if __name__ == "__main__":
    exit(main())
