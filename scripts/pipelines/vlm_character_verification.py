#!/usr/bin/env python3
"""
VLM-Based Character Verification for 3D Animation
==================================================

Uses Vision-Language Models (Qwen2-VL) to verify character identity
by asking the model to identify who is in each image.

This approach is more effective than ArcFace for 3D animated characters
because VLMs understand semantic features (clothing, hair, body type)
not just facial embeddings.

Author: Claude Code
Date: 2025-11-14
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import shutil
import sys

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(log_file: Path = None) -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_vlm_model(model_name: str = "qwen2_vl", device: str = "cuda"):
    """
    Load Vision-Language Model for character identification

    Args:
        model_name: Model to use (qwen2_vl, internvl2, llava)
        device: Device to run on (cuda/cpu)

    Returns:
        model, processor
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading VLM: {model_name}")

    if model_name == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    logger.info(f"✓ Model loaded on {device}")
    return model, processor


def verify_character_identity(
    image_path: Path,
    model,
    processor,
    character_name: str,
    character_description: str,
    logger: logging.Logger = None
) -> Tuple[bool, float, str]:
    """
    Ask VLM to verify if the image contains the target character

    Args:
        image_path: Path to image
        model: VLM model
        processor: VLM processor
        character_name: Target character name (e.g., "Luca Paguro")
        character_description: Detailed character description

    Returns:
        (is_match, confidence, explanation)
    """
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    # Load image
    image = Image.open(image_path).convert('RGB')

    # Construct verification prompt
    prompt = f"""You are an expert at identifying 3D animated characters from Pixar films.

TARGET CHARACTER: {character_name}
DESCRIPTION: {character_description}

Please analyze this image and answer:
1. Is this {character_name}? (YES/NO)
2. Confidence level (0-100%)
3. Brief explanation of your decision

Focus on:
- Hair color and style
- Age and body type
- Clothing/outfit
- Facial features
- Overall character design

Respond in JSON format:
{{
  "is_target_character": true/false,
  "confidence": 0-100,
  "explanation": "brief explanation",
  "detected_character": "character name if different"
}}"""

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate response
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Parse JSON response
        try:
            result = json.loads(output_text)
            is_match = result.get("is_target_character", False)
            confidence = result.get("confidence", 0) / 100.0
            explanation = result.get("explanation", "")

            if logger:
                logger.debug(f"{image_path.name}: {'✓' if is_match else '✗'} ({confidence:.2%}) - {explanation}")

            return is_match, confidence, explanation

        except json.JSONDecodeError:
            # Fallback: simple text parsing
            is_match = "yes" in output_text.lower() and character_name.lower() in output_text.lower()
            confidence = 0.5 if is_match else 0.0

            if logger:
                logger.warning(f"Failed to parse JSON from VLM, using fallback: {output_text[:100]}")

            return is_match, confidence, output_text

    except Exception as e:
        if logger:
            logger.error(f"VLM inference failed for {image_path.name}: {e}")
        return False, 0.0, str(e)


def verify_dataset(
    input_dir: Path,
    output_dir: Path,
    character_name: str,
    character_description: str,
    confidence_threshold: float = 0.7,
    model_name: str = "qwen2_vl",
    device: str = "cuda",
    logger: logging.Logger = None
):
    """
    Verify all images in dataset using VLM

    Args:
        input_dir: Directory with candidate images
        output_dir: Output directory for verified images
        character_name: Target character name
        character_description: Detailed character description
        confidence_threshold: Minimum confidence to accept (0.0-1.0)
        model_name: VLM model to use
        device: Device (cuda/cpu)
        logger: Logger instance
    """
    logger = logger or logging.getLogger(__name__)

    # Load VLM
    model, processor = load_vlm_model(model_name, device)

    # Find all images
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    logger.info(f"Found {len(image_files)} images to verify")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verification results
    verified = []
    rejected = []

    # Process each image
    for img_path in tqdm(image_files, desc="VLM Verification"):
        is_match, confidence, explanation = verify_character_identity(
            img_path,
            model,
            processor,
            character_name,
            character_description,
            logger
        )

        if is_match and confidence >= confidence_threshold:
            # Copy to output
            shutil.copy2(img_path, output_dir / img_path.name)
            verified.append({
                "image": img_path.name,
                "confidence": confidence,
                "explanation": explanation
            })
        else:
            rejected.append({
                "image": img_path.name,
                "confidence": confidence,
                "explanation": explanation,
                "reason": "low_confidence" if confidence < confidence_threshold else "not_target_character"
            })

    # Save results
    results = {
        "character_name": character_name,
        "total_images": len(image_files),
        "verified": len(verified),
        "rejected": len(rejected),
        "threshold": confidence_threshold,
        "verified_list": verified,
        "rejected_list": rejected
    }

    with open(output_dir / "vlm_verification_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ VLM Verification complete:")
    logger.info(f"  Total: {len(image_files)}")
    logger.info(f"  Verified: {len(verified)} ({len(verified)/len(image_files)*100:.1f}%)")
    logger.info(f"  Rejected: {len(rejected)} ({len(rejected)/len(image_files)*100:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="VLM-based character verification for 3D animation datasets"
    )
    parser.add_argument(
        '--input-dir', type=Path, required=True,
        help='Input directory with candidate images'
    )
    parser.add_argument(
        '--output-dir', type=Path, required=True,
        help='Output directory for verified images'
    )
    parser.add_argument(
        '--character-name', type=str, required=True,
        help='Target character name (e.g., "Luca Paguro")'
    )
    parser.add_argument(
        '--character-description', type=str, required=True,
        help='Detailed character description for VLM'
    )
    parser.add_argument(
        '--confidence-threshold', type=float, default=0.7,
        help='Minimum confidence to accept (default: 0.7)'
    )
    parser.add_argument(
        '--model', type=str, default='qwen2_vl',
        choices=['qwen2_vl', 'internvl2', 'llava'],
        help='VLM model to use (default: qwen2_vl)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--log-file', type=Path, default=None,
        help='Log file path'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_file)

    logger.info("=" * 80)
    logger.info("VLM-Based Character Verification")
    logger.info("=" * 80)
    logger.info(f"Character: {args.character_name}")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Threshold: {args.confidence_threshold}")
    logger.info(f"Model: {args.model}")

    # Run verification
    results = verify_dataset(
        args.input_dir,
        args.output_dir,
        args.character_name,
        args.character_description,
        args.confidence_threshold,
        args.model,
        args.device,
        logger
    )

    logger.info("=" * 80)
    logger.info("✓ Verification Complete")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
