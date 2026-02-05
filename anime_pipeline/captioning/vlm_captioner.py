#!/usr/bin/env python3
"""
VLM-based Caption Generation for Animation Characters.

This module provides caption generation using Vision Language Models (VLMs):
- Qwen2-VL for detailed character descriptions
- InternVL2 as alternative
- Schema-guided outputs for consistent formatting

Supports:
- Single image captioning
- Batch captioning with progress tracking
- Character-aware prompts
- Style-specific templates (2D/3D animation)

Usage:
    from anime_pipeline.captioning import VLMCaptioner

    captioner = VLMCaptioner(
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        device="cuda"
    )

    caption = captioner.generate_caption(
        image_path="/path/to/image.png",
        prefix="a 2d animated character"
    )

Author: Justin Lu
Date: 2025-12-02
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime
import re

# Conditional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logger = logging.getLogger(__name__)


# Caption templates for different styles
CAPTION_TEMPLATES = {
    "2d_character": "a 2d animated {character_name} character, {style}, {pose}, {expression}, {details}",
    "3d_character": "a 3d animated {character_name} character, pixar style, {pose}, {expression}, {lighting}",
    "scene": "{character_name} in {location}, {action}, {mood}, {style}",
    "detailed": "{character_name}, {description}, {pose}, {expression}, {outfit}, {background}",
}

# Default prompts for different animation styles
DEFAULT_PROMPTS = {
    "2d": """Describe this 2D animated character image for AI training.
Include: character appearance, clothing, expression, pose, art style.
Be concise but detailed. Use comma-separated descriptors.
Start with "a 2d animated character" and focus on visual elements.""",

    "3d": """Describe this 3D animated character image for AI training.
Include: character appearance, clothing, expression, pose, lighting, materials.
Be concise but detailed. Use comma-separated descriptors.
Start with "a 3d animated character, pixar style" and focus on visual elements.""",

    "schema": """Analyze this animated character image and provide a structured description.
Return a JSON object with these fields:
- character_type: "human", "animal", or "creature"
- age_group: "child", "teen", "adult", "elderly"
- expression: emotional state
- pose: body position/action
- outfit: clothing description
- art_style: animation style
- lighting: lighting conditions
- final_caption: complete caption for training""",
}


@dataclass
class CaptionResult:
    """Result of caption generation."""
    image_path: str
    caption: str
    tokens: int = 0
    model: str = ""
    generation_time: float = 0.0
    schema_data: Optional[Dict] = None


@dataclass
class BatchCaptionResult:
    """Result of batch caption generation."""
    total_images: int = 0
    successful: int = 0
    failed: int = 0
    avg_tokens: float = 0.0
    total_time: float = 0.0
    results: List[CaptionResult] = field(default_factory=list)


class VLMCaptioner:
    """
    Generate captions using Vision Language Models.

    Supports Qwen2-VL and InternVL2 for animation character captioning.
    Includes stub mode for testing without GPU.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        max_tokens: int = 77,
        use_stub: bool = False,
        load_in_8bit: bool = False,
        verbose: bool = False
    ):
        """
        Initialize VLM captioner.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use ("cuda" or "cpu")
            max_tokens: Maximum tokens in generated caption
            use_stub: Use stub mode for testing
            load_in_8bit: Use 8-bit quantization to save memory
            verbose: Print detailed logging
        """
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.use_stub = use_stub
        self.load_in_8bit = load_in_8bit
        self.verbose = verbose

        self.model = None
        self.processor = None
        self.tokenizer = None
        self._initialized = False

        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        # Check if we should use stub
        if use_stub or not TRANSFORMERS_AVAILABLE or not PIL_AVAILABLE:
            self.use_stub = True
            logger.info("Using stub mode for VLM captioning")
        elif not TORCH_AVAILABLE or (device == "cuda" and not torch.cuda.is_available()):
            self.use_stub = True
            logger.info("CUDA not available, using stub mode")
        else:
            # Don't load model in __init__ - lazy loading
            logger.info(f"VLMCaptioner initialized (model will load on first use): {model_name}")

    def _initialize_model(self) -> bool:
        """
        Initialize the VLM model (lazy loading).

        Returns:
            True if successful, False otherwise
        """
        if self._initialized:
            return True

        if self.use_stub:
            return False

        try:
            logger.info(f"Loading VLM model: {self.model_name}")

            # Determine model type
            model_lower = self.model_name.lower()

            if "qwen" in model_lower:
                self._load_qwen_model()
            elif "internvl" in model_lower:
                self._load_internvl_model()
            else:
                # Generic loading
                self._load_generic_model()

            self._initialized = True
            logger.info("VLM model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")
            self.use_stub = True
            return False

    def _load_qwen_model(self) -> None:
        """Load Qwen2-VL model."""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )

    def _load_internvl_model(self) -> None:
        """Load InternVL2 model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "trust_remote_code": True,
        }

        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        if self.device == "cuda" and not self.load_in_8bit:
            self.model = self.model.cuda()

    def _load_generic_model(self) -> None:
        """Load generic vision-language model."""
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        if self.device == "cuda":
            self.model = self.model.cuda()

    def _stub_generate(
        self,
        image_path: Path,
        prefix: str = "a 2d animated character"
    ) -> str:
        """
        Generate stub caption for testing.

        Args:
            image_path: Path to image
            prefix: Caption prefix

        Returns:
            Stub caption string
        """
        filename = image_path.stem

        # Parse character name from filename if possible
        parts = filename.replace('_', ' ').replace('-', ' ').split()
        char_name = parts[0] if parts else "character"

        stub_elements = [
            prefix,
            char_name,
            "standing pose",
            "neutral expression",
            "simple background",
            "high quality",
            "detailed",
        ]

        return ", ".join(stub_elements)

    def generate_caption(
        self,
        image_path: Path,
        character_info: Optional[Dict] = None,
        prefix: str = "a 2d animated character",
        style: str = "2d",
        custom_prompt: Optional[str] = None
    ) -> CaptionResult:
        """
        Generate caption for a single image.

        Args:
            image_path: Path to image file
            character_info: Optional character metadata
            prefix: Caption prefix
            style: Animation style ("2d" or "3d")
            custom_prompt: Custom prompt to use

        Returns:
            CaptionResult with generated caption
        """
        import time

        image_path = Path(image_path)
        result = CaptionResult(
            image_path=str(image_path),
            caption="",
            model=self.model_name
        )

        if not image_path.exists():
            result.caption = f"Error: Image not found: {image_path}"
            return result

        start_time = time.time()

        # Use stub or real generation
        if self.use_stub:
            result.caption = self._stub_generate(image_path, prefix)
        else:
            if not self._initialized:
                self._initialize_model()

            if self.use_stub:  # Model loading failed
                result.caption = self._stub_generate(image_path, prefix)
            else:
                result.caption = self._real_generate(
                    image_path,
                    character_info,
                    prefix,
                    style,
                    custom_prompt
                )

        result.generation_time = time.time() - start_time
        result.tokens = len(result.caption.split())

        # Truncate if needed
        words = result.caption.split()
        if len(words) > self.max_tokens:
            result.caption = " ".join(words[:self.max_tokens])
            result.tokens = self.max_tokens

        return result

    def _real_generate(
        self,
        image_path: Path,
        character_info: Optional[Dict],
        prefix: str,
        style: str,
        custom_prompt: Optional[str]
    ) -> str:
        """
        Generate caption using actual VLM model.

        Returns:
            Generated caption string
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Build prompt
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = DEFAULT_PROMPTS.get(style, DEFAULT_PROMPTS["2d"])

            # Add character info to prompt
            if character_info:
                char_name = character_info.get("name", "character")
                prompt = f"This is {char_name}. " + prompt

            # Generate based on model type
            model_lower = self.model_name.lower()

            if "qwen" in model_lower:
                caption = self._generate_qwen(image, prompt)
            elif "internvl" in model_lower:
                caption = self._generate_internvl(image, prompt)
            else:
                caption = self._generate_generic(image, prompt)

            # Clean and format caption
            caption = self._clean_caption(caption, prefix)

            return caption

        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            return f"Error: {str(e)[:50]}"

    def _generate_qwen(self, image: Image.Image, prompt: str) -> str:
        """Generate caption using Qwen2-VL."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7
            )

        output = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        # Extract the assistant's response
        if "assistant" in output.lower():
            output = output.split("assistant")[-1].strip()

        return output

    def _generate_internvl(self, image: Image.Image, prompt: str) -> str:
        """Generate caption using InternVL2."""
        # InternVL2 specific processing
        pixel_values = self._process_internvl_image(image)

        generation_config = {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7
        }

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config
            )

        return response

    def _process_internvl_image(self, image: Image.Image) -> torch.Tensor:
        """Process image for InternVL2."""
        # Simplified - actual implementation needs InternVL preprocessing
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        pixel_values = transform(image).unsqueeze(0)

        if self.device == "cuda":
            pixel_values = pixel_values.cuda().half()

        return pixel_values

    def _generate_generic(self, image: Image.Image, prompt: str) -> str:
        """Generate caption using generic model."""
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7
            )

        return self.processor.decode(output_ids[0], skip_special_tokens=True)

    def _clean_caption(self, caption: str, prefix: str) -> str:
        """
        Clean and format generated caption.

        Args:
            caption: Raw generated caption
            prefix: Expected prefix

        Returns:
            Cleaned caption
        """
        # Remove any leading/trailing whitespace
        caption = caption.strip()

        # Remove common artifacts
        artifacts = ["<image>", "</image>", "<|im_end|>", "<|endoftext|>"]
        for artifact in artifacts:
            caption = caption.replace(artifact, "")

        # Remove JSON-like structures if present (for schema prompts)
        if caption.startswith("{") and caption.endswith("}"):
            try:
                data = json.loads(caption)
                if "final_caption" in data:
                    caption = data["final_caption"]
            except json.JSONDecodeError:
                pass

        # Ensure prefix is present
        if not caption.lower().startswith(prefix.lower()[:20]):
            caption = f"{prefix}, {caption}"

        # Clean up punctuation and spacing
        caption = re.sub(r'\s+', ' ', caption)
        caption = re.sub(r',\s*,', ',', caption)
        caption = caption.strip(', ')

        return caption

    def batch_caption(
        self,
        images_dir: Path,
        output_dir: Path,
        batch_size: int = 4,
        character_info: Optional[Dict] = None,
        prefix: str = "a 2d animated character",
        style: str = "2d",
        overwrite: bool = False
    ) -> BatchCaptionResult:
        """
        Generate captions for all images in a directory.

        Args:
            images_dir: Directory containing images
            output_dir: Directory for caption files
            batch_size: Images to process at once
            character_info: Optional character metadata
            prefix: Caption prefix
            style: Animation style
            overwrite: Overwrite existing captions

        Returns:
            BatchCaptionResult with all results
        """
        import time

        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = BatchCaptionResult()

        # Find images
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        result.total_images = len(image_files)
        logger.info(f"Found {result.total_images} images to caption")

        if not image_files:
            return result

        start_time = time.time()

        # Process images
        for i, image_path in enumerate(image_files):
            # Check if caption already exists
            caption_path = output_dir / f"{image_path.stem}.txt"

            if caption_path.exists() and not overwrite:
                logger.debug(f"Skipping (exists): {image_path.name}")
                continue

            try:
                caption_result = self.generate_caption(
                    image_path,
                    character_info,
                    prefix,
                    style
                )

                # Save caption
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption_result.caption)

                result.results.append(caption_result)
                result.successful += 1

                if self.verbose and (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{result.total_images}")

            except Exception as e:
                logger.error(f"Error captioning {image_path.name}: {e}")
                result.failed += 1

        result.total_time = time.time() - start_time

        if result.results:
            result.avg_tokens = sum(r.tokens for r in result.results) / len(result.results)

        logger.info(
            f"Batch captioning complete: {result.successful} success, "
            f"{result.failed} failed, {result.total_time:.1f}s"
        )

        return result

    def schema_guided_caption(
        self,
        image_path: Path,
        schema: Optional[Dict] = None
    ) -> Dict:
        """
        Generate structured caption following a schema.

        Args:
            image_path: Path to image
            schema: Optional custom schema

        Returns:
            Dictionary with schema fields filled
        """
        default_schema = {
            "character_type": "",
            "expression": "",
            "pose": "",
            "outfit": "",
            "art_style": "",
            "lighting": "",
            "background": "",
            "final_caption": ""
        }

        schema = schema or default_schema

        # Generate with schema prompt
        result = self.generate_caption(
            image_path,
            custom_prompt=DEFAULT_PROMPTS["schema"]
        )

        # Try to parse JSON from result
        try:
            # Find JSON in response
            caption_text = result.caption

            # Look for JSON object
            start = caption_text.find('{')
            end = caption_text.rfind('}') + 1

            if start != -1 and end > start:
                json_str = caption_text[start:end]
                parsed = json.loads(json_str)

                # Merge with default schema
                for key in schema:
                    if key in parsed:
                        schema[key] = parsed[key]

        except json.JSONDecodeError:
            # Fallback: use regular caption
            schema["final_caption"] = result.caption

        return schema


def apply_template(template_name: str, **kwargs) -> str:
    """
    Apply a caption template with provided values.

    Args:
        template_name: Name of template to use
        **kwargs: Template values

    Returns:
        Formatted caption string
    """
    if template_name not in CAPTION_TEMPLATES:
        template_name = "2d_character"

    template = CAPTION_TEMPLATES[template_name]

    # Fill in template with defaults for missing values
    defaults = {
        "character_name": "character",
        "style": "animated",
        "pose": "standing",
        "expression": "neutral",
        "details": "",
        "lighting": "studio lighting",
        "location": "",
        "action": "",
        "mood": "",
        "outfit": "",
        "description": "",
        "background": ""
    }

    defaults.update(kwargs)

    try:
        return template.format(**defaults)
    except KeyError as e:
        logger.warning(f"Missing template key: {e}")
        return template.format_map(defaults)


# Create __init__.py content
def create_init_file():
    """Generate __init__.py for captioning module."""
    return '''"""
Captioning module for 2D Animation LoRA Pipeline.

Provides VLM-based caption generation for animation characters:
- VLMCaptioner: Main caption generation class
- Caption templates for 2D/3D animation styles
- Schema-guided structured outputs

Usage:
    from anime_pipeline.captioning import VLMCaptioner, apply_template

    captioner = VLMCaptioner(device="cuda")
    result = captioner.generate_caption(image_path)
"""

from .vlm_captioner import (
    VLMCaptioner,
    CaptionResult,
    BatchCaptionResult,
    apply_template,
    CAPTION_TEMPLATES,
    DEFAULT_PROMPTS,
)

__all__ = [
    "VLMCaptioner",
    "CaptionResult",
    "BatchCaptionResult",
    "apply_template",
    "CAPTION_TEMPLATES",
    "DEFAULT_PROMPTS",
]
'''


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="VLM Caption Generation")
    parser.add_argument("--image", type=str, help="Single image to caption")
    parser.add_argument("--images-dir", type=str, help="Directory of images to caption")
    parser.add_argument("--output-dir", type=str, help="Output directory for captions")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="VLM model")
    parser.add_argument("--prefix", type=str, default="a 2d animated character", help="Caption prefix")
    parser.add_argument("--style", type=str, default="2d", choices=["2d", "3d"], help="Animation style")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--stub", action="store_true", help="Use stub mode (no GPU)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    captioner = VLMCaptioner(
        model_name=args.model,
        device=args.device,
        use_stub=args.stub,
        verbose=args.verbose
    )

    if args.image:
        # Single image
        result = captioner.generate_caption(
            Path(args.image),
            prefix=args.prefix,
            style=args.style
        )
        print(f"\nCaption: {result.caption}")
        print(f"Tokens: {result.tokens}")
        print(f"Time: {result.generation_time:.2f}s")

    elif args.images_dir and args.output_dir:
        # Batch processing
        result = captioner.batch_caption(
            Path(args.images_dir),
            Path(args.output_dir),
            prefix=args.prefix,
            style=args.style
        )
        print(f"\nBatch Results:")
        print(f"  Total: {result.total_images}")
        print(f"  Success: {result.successful}")
        print(f"  Failed: {result.failed}")
        print(f"  Avg tokens: {result.avg_tokens:.1f}")
        print(f"  Time: {result.total_time:.1f}s")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
