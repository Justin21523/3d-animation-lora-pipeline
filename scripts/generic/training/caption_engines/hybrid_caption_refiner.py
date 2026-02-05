#!/usr/bin/env python3
"""
Hybrid caption generation/refinement using LLMProvider API.

For images with original generation prompts:
  - Use prompt refinement mode (LLMProvider corrects/refines existing prompt)
For images without original prompts:
  - Use full VLM generation mode (LLMProvider generates from scratch)

This approach preserves the semantic intent of original prompts while fixing errors.
"""

import os
import json
import base64
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import llm_vendor
from PIL import Image


# ============================================================================
# Prompt Templates
# ============================================================================

REFINEMENT_PROMPTS = {
    "pose": """I have a 3D animated character image with this ORIGINAL prompt used to generate it:

"{original_prompt}"

Please analyze the actual image and REFINE this prompt to accurately describe what you see. Focus on:

1. **Verify/Correct Primary Pose**: Is it really standing/sitting/crouching/lying/leaning as stated?
2. **Verify/Correct Body Position**: upright/bent/twisted/relaxed/tense
3. **Verify/Correct Limb Details**: arm and leg positions
4. **Verify/Correct View Angle**: front/three-quarter/side/back
5. **Verify/Correct Camera Height**: eye-level/low-angle/high-angle

If the original prompt is accurate, you can keep it with minor refinements.
If there are errors (e.g., says "standing" but image shows "sitting"), CORRECT them.

Generate a refined caption (75-150 tokens) that:
- Starts with "a 3d animated character"
- Accurately describes the ACTUAL pose in the image
- Preserves the semantic intent where correct
- Fixes any errors

Refined Caption:""",

    "action": """I have a 3D animated character image with this ORIGINAL prompt used to generate it:

"{original_prompt}"

Please analyze the actual image and REFINE this prompt to accurately describe what you see. Focus on:

1. **Verify/Correct Primary Action**: walking/running/jumping/reaching/etc.
2. **Verify/Correct Motion Phase**: starting/mid-motion/landing
3. **Verify/Correct Energy Level**: explosive/controlled/gentle
4. **Verify/Correct Body Dynamics**: balance, momentum
5. **Verify/Correct View Angle**: front/side/three-quarter

If the original prompt is accurate, keep it with minor refinements.
If there are errors, CORRECT them while preserving the action intent.

Generate a refined caption (75-150 tokens) that:
- Starts with "a 3d animated character"
- Accurately describes the ACTUAL action in the image
- Preserves correct semantic information
- Fixes any errors

Refined Caption:""",

    "expression": """I have a 3D animated character image with this ORIGINAL prompt used to generate it:

"{original_prompt}"

Please analyze the actual image and REFINE this prompt to accurately describe what you see. Focus on:

1. **Verify/Correct Primary Emotion**: happy/sad/angry/surprised/neutral
2. **Verify/Correct Facial Features**: eyes, mouth, eyebrows, cheeks
3. **Verify/Correct Expression Intensity**: subtle/moderate/extreme
4. **Verify/Correct Head Orientation**: facing/tilted/turned
5. **Verify/Correct Gaze Direction**: at camera/away/down/up

If the original prompt is accurate, keep it with minor refinements.
If there are errors in emotion or facial features, CORRECT them.

Generate a refined caption (75-150 tokens) that:
- Starts with "a 3d animated character"
- Accurately describes the ACTUAL expression in the image
- Preserves correct emotional context
- Fixes any errors

Refined Caption:"""
}

GENERATION_PROMPTS = {
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

Caption:""",

    "action": """Analyze this 3D animated character image and describe it for training an action LoRA.

Focus on:
1. **Primary Action**: (walking/running/jumping/reaching/climbing/etc.)
2. **Motion Phase**: (starting/mid-motion/landing/follow-through)
3. **Body Dynamics**: energy level, balance, momentum
4. **Limb Coordination**: arms and legs movement
5. **Facial Expression**: (focused/strained/relaxed/excited) if visible
6. **View Angle**: (front/side/three-quarter/back)

Generate a natural, flowing caption (75-150 tokens) that includes:
- Trigger phrase: "a 3d animated character"
- Detailed action description
- Motion quality and energy
- View angle

Caption:""",

    "expression": """Analyze this 3D animated character's face and describe it for training an expression LoRA.

Focus on:
1. **Primary Emotion**: (happy/sad/angry/surprised/neutral/mixed)
2. **Facial Features**: eyes, mouth, eyebrows, cheeks, forehead
3. **Intensity**: (subtle/moderate/extreme)
4. **Head Orientation**: (facing forward/tilted/turned)
5. **Gaze Direction**: (at camera/away/down/up/aside)
6. **View Angle**: (close-up/medium/three-quarter/profile)

Generate a natural, flowing caption (75-150 tokens) that includes:
- Trigger phrase: "a 3d animated character"
- Primary emotion
- Detailed facial feature description
- Expression intensity

Caption:"""
}


# ============================================================================
# Helper Functions
# ============================================================================

def extract_image_id(filename: str) -> Optional[str]:
    """Extract image_XXXXXX_XX pattern from filename."""
    match = re.search(r'(image_\d+_\d+)', filename)
    if match:
        return match.group(1)
    return None


def load_generation_metadata(generated_data_root: Path, lora_type: str) -> Dict[str, Dict]:
    """
    Load all generation metadata for a lora type across all characters.

    Returns:
        {
            "character_imageID": {
                "character": "alberto",
                "prompt": "...",
                "seed": 12345,
                "filename": "image_000001_00.png"
            }
        }
    """
    prompts = {}

    for char_dir in generated_data_root.iterdir():
        if not char_dir.is_dir():
            continue

        metadata_path = char_dir / lora_type / "generated" / "generation_metadata.json"
        if not metadata_path.exists():
            continue

        with open(metadata_path) as f:
            metadata = json.load(f)

        for img_data in metadata.get("generated_images", []):
            filename = img_data["filename"]
            image_id = extract_image_id(filename)

            if image_id:
                key = f"{char_dir.name}_{image_id}"
                prompts[key] = {
                    "character": char_dir.name,
                    "prompt": img_data["prompt"],
                    "seed": img_data.get("seed"),
                    "filename": filename
                }

    return prompts


def match_image_to_prompt(image_path: Path, prompt_map: Dict) -> Optional[str]:
    """
    Try to find original prompt for an image.

    Returns original prompt if found, None otherwise.
    """
    filename = image_path.name

    # Extract character and image_id from filename
    # Format: alberto_image_000005_03.png or alberto_seamonster_image_000005_03.png
    match = re.match(r'(\w+)_(image_\d+_\d+)', filename)

    if match:
        char = match.group(1)
        image_id = match.group(2)
        key = f"{char}_{image_id}"

        if key in prompt_map:
            return prompt_map[key]["prompt"]

    return None


# ============================================================================
# LLMProvider Captioner
# ============================================================================

class HybridCaptioner:
    """Generate or refine captions using LLMProvider API."""

    def __init__(self, api_key: str, model: str = "llm_provider-3-5-sonnet-20241022"):
        self.client = llm_vendor.LLMVendor(api_key=api_key)
        self.model = model
        self.stats = {
            "total": 0,
            "refined": 0,  # Had original prompt
            "generated": 0,  # No original prompt
            "success": 0,
            "failed": 0
        }

    def process_image(
        self,
        image_path: Path,
        lora_type: str,
        original_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> Tuple[str, bool, str]:
        """
        Generate or refine caption for an image.

        Returns:
            (caption, success, mode)
            mode: "refined" or "generated"
        """
        self.stats["total"] += 1

        try:
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()

            ext = image_path.suffix.lower()
            media_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp"
            }.get(ext, "image/png")

            # Choose mode and prompt
            if original_prompt:
                mode = "refined"
                prompt_template = REFINEMENT_PROMPTS.get(lora_type, REFINEMENT_PROMPTS["pose"])
                prompt = prompt_template.format(original_prompt=original_prompt)
            else:
                mode = "generated"
                prompt = GENERATION_PROMPTS.get(lora_type, GENERATION_PROMPTS["pose"])

            # Call LLMProvider API with retries
            for attempt in range(max_retries):
                try:
                    message = self.client.messages.create(
                        model=self.model,
                        max_tokens=300,
                        temperature=0.7,
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

                    # Clean up
                    for prefix in ["refined caption:", "caption:", "refined:"]:
                        if caption.lower().startswith(prefix):
                            caption = caption[len(prefix):].strip()

                    # Validate
                    if self._validate_caption(caption):
                        self.stats["success"] += 1
                        if mode == "refined":
                            self.stats["refined"] += 1
                        else:
                            self.stats["generated"] += 1
                        return caption, True, mode
                    else:
                        if attempt < max_retries - 1:
                            continue
                        else:
                            self.stats["failed"] += 1
                            return "", False, mode

                except llm_vendor.RateLimitError:
                    import time
                    wait_time = (attempt + 1) * 5
                    time.sleep(wait_time)
                    continue

                except Exception as e:
                    if attempt < max_retries - 1:
                        continue
                    else:
                        self.stats["failed"] += 1
                        return "", False, mode

            self.stats["failed"] += 1
            return "", False, mode

        except Exception as e:
            self.stats["failed"] += 1
            return "", False, "error"

    def _validate_caption(self, caption: str) -> bool:
        """Validate caption meets requirements."""
        if not caption:
            return False

        word_count = len(caption.split())
        if word_count < 30 or word_count > 200:
            return False

        if "3d animated character" not in caption.lower():
            return False

        return True

    def get_stats(self) -> Dict:
        return self.stats.copy()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid caption refinement/generation with LLMProvider API"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory (e.g., .../universal_pose/1_universal_pose)"
    )
    parser.add_argument(
        "--generated-data-root",
        type=Path,
        required=True,
        help="Root of generated data (contains metadata)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for refined captions"
    )
    parser.add_argument(
        "--lora-type",
        choices=["pose", "action", "expression"],
        required=True,
        help="Type of LoRA"
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

    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("LLM_VENDOR_API_KEY")
    if not api_key:
        print("❌ Error: LLM_VENDOR_API_KEY not set")
        return 1

    # Validate paths
    if not args.dataset_dir.exists():
        print(f"❌ Error: Dataset directory not found: {args.dataset_dir}")
        return 1

    if not args.generated_data_root.exists():
        print(f"❌ Error: Generated data root not found: {args.generated_data_root}")
        return 1

    # Create output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load generation metadata
    print("📖 Loading generation metadata...")
    prompt_map = load_generation_metadata(args.generated_data_root, args.lora_type)
    print(f"✅ Loaded {len(prompt_map)} original prompts\n")

    # Find all images
    image_files = list(args.dataset_dir.glob("*.png"))
    if not image_files:
        print(f"❌ Error: No images found in {args.dataset_dir}")
        return 1

    # Match images to prompts
    matched = 0
    unmatched = 0
    for img in image_files:
        if match_image_to_prompt(img, prompt_map):
            matched += 1
        else:
            unmatched += 1

    print(f"{'='*70}")
    print(f"🔍 Hybrid Caption Refinement/Generation")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"LoRA Type: {args.lora_type}")
    print(f"Total Images: {len(image_files)}")
    print(f"  - With original prompts: {matched} (will refine)")
    print(f"  - Without prompts: {unmatched} (will generate)")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.workers}")
    print(f"{'='*70}\n")

    # Initialize captioner
    captioner = HybridCaptioner(api_key)

    # Process images
    print(f"🚀 Processing {len(image_files)} images...")

    successful = 0
    failed = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_image = {}

        for img_file in image_files:
            caption_path = args.output_dir / img_file.with_suffix(".txt").name
            if caption_path.exists() and not args.force:
                skipped += 1
                continue

            original_prompt = match_image_to_prompt(img_file, prompt_map)

            future = executor.submit(
                captioner.process_image,
                img_file,
                args.lora_type,
                original_prompt
            )
            future_to_image[future] = img_file

        with tqdm(total=len(future_to_image), desc="Processing") as pbar:
            for future in as_completed(future_to_image):
                img_file = future_to_image[future]
                caption, success, mode = future.result()

                if success:
                    # Save caption
                    caption_path = args.output_dir / img_file.with_suffix(".txt").name
                    caption_path.write_text(caption, encoding="utf-8")

                    # Copy image
                    if args.output_dir != args.dataset_dir:
                        out_img = args.output_dir / img_file.name
                        if not out_img.exists():
                            try:
                                os.link(img_file, out_img)
                            except:
                                import shutil
                                shutil.copy2(img_file, out_img)

                    successful += 1
                else:
                    failed += 1

                pbar.update(1)

    # Print summary
    stats = captioner.get_stats()

    print(f"\n{'='*70}")
    print(f"✅ Processing Complete")
    print(f"{'='*70}")
    print(f"Total: {len(image_files)}")
    print(f"Success: {successful}")
    print(f"  - Refined (had original): {stats['refined']}")
    print(f"  - Generated (no original): {stats['generated']}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"{'='*70}\n")

    # Save report
    report = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "lora_type": args.lora_type,
        "total_images": len(image_files),
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "refined": stats["refined"],
        "generated": stats["generated"],
        "workers": args.workers
    }

    report_path = args.output_dir / "hybrid_caption_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"📊 Report saved: {report_path}\n")

    return 0


if __name__ == "__main__":
    exit(main())
