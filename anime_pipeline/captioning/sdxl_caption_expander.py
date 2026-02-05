#!/usr/bin/env python3
"""
SDXL Caption Expander - Expand short captions to SDXL-optimized longer captions.

SDXL benefits from longer, more descriptive captions (77-150 tokens) compared to
SD1.5's optimal range of 40-77 tokens. This module uses OpenAI GPT-4o-mini to
intelligently expand short captions while maintaining accuracy and adding
SDXL-specific quality tags.

Features:
- Expand short captions (40-77 tokens) to SDXL-optimized length (77-150 tokens)
- Add quality prefixes and technical details
- Generate appropriate negative prompts
- Async batch processing with rate limiting
- Cost tracking and logging
- Stub mode for testing without API calls

Usage:
    from anime_pipeline.captioning.sdxl_caption_expander import SDXLCaptionExpander

    expander = SDXLCaptionExpander()  # Uses OPENAI_API_KEY from env
    result = expander.expand_caption(
        "a young boy with brown hair, smiling",
        style="2d"
    )
    print(result.main_prompt)

Author: Justin Lu
Date: 2025-12-03
"""

import os
import sys
import asyncio
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Conditional imports
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    AsyncOpenAI = None

logger = logging.getLogger(__name__)


# SDXL Quality Prefixes by style
SDXL_QUALITY_PREFIXES = {
    "2d": "masterpiece, best quality, highly detailed 2d animation",
    "3d": "masterpiece, best quality, highly detailed 3d render, pixar style",
    "anime": "masterpiece, best quality, highly detailed anime art",
    "realistic": "masterpiece, best quality, highly detailed, photorealistic",
    "default": "masterpiece, best quality, highly detailed",
}

# Technical detail templates by style
SDXL_TECHNICAL_DETAILS = {
    "2d": "clean lines, vibrant colors, professional animation, studio quality",
    "3d": "subsurface scattering, ray tracing, ambient occlusion, soft shadows, PBR materials",
    "anime": "sharp lines, cel shading, anime aesthetic, high contrast",
    "realistic": "natural lighting, 8k uhd, dslr, soft lighting, high quality",
    "default": "professional lighting, high resolution, detailed textures",
}

# Default negative prompts by style
SDXL_NEGATIVE_PROMPTS = {
    "2d": "blurry, low quality, bad anatomy, deformed, ugly, duplicate, morbid, mutated, poorly drawn, bad proportions, extra limbs, missing limbs, disconnected limbs",
    "3d": "blurry, low quality, bad anatomy, deformed, ugly, 2d, flat, sketch, drawing, poorly rendered, bad textures, low poly, missing fingers",
    "anime": "blurry, low quality, bad anatomy, deformed, ugly, duplicate, realistic, 3d render, photo, western cartoon, poorly drawn",
    "realistic": "cartoon, anime, drawing, painting, illustration, cgi, low quality, blurry, bad anatomy, deformed",
    "default": "blurry, low quality, bad anatomy, deformed, ugly, duplicate, poorly drawn, bad proportions",
}

# Cost per 1K tokens (as of Dec 2024)
OPENAI_PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}


@dataclass
class ExpandedCaption:
    """Result of caption expansion for SDXL training."""
    original: str               # Original short caption
    main_prompt: str            # Full expanded description
    quality_prefix: str         # "masterpiece, best quality, highly detailed"
    technical_details: str      # Style-specific technical terms
    negative_prompt: str        # Auto-generated negative prompt
    full_caption: str           # Combined caption for training
    token_count: int            # Estimated token count
    style: str                  # Style used for expansion
    model: str                  # Model used for generation
    generation_time: float      # Time taken in seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original": self.original,
            "main_prompt": self.main_prompt,
            "quality_prefix": self.quality_prefix,
            "technical_details": self.technical_details,
            "negative_prompt": self.negative_prompt,
            "full_caption": self.full_caption,
            "token_count": self.token_count,
            "style": self.style,
            "model": self.model,
            "generation_time": self.generation_time,
        }


@dataclass
class BatchExpandResult:
    """Result of batch caption expansion."""
    total_captions: int
    successful: int
    failed: int
    avg_token_count: float
    total_time: float
    results: List[ExpandedCaption]
    errors: List[Tuple[str, str]] = field(default_factory=list)  # (caption, error)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_captions": self.total_captions,
            "successful": self.successful,
            "failed": self.failed,
            "avg_token_count": self.avg_token_count,
            "total_time": self.total_time,
            "results": [r.to_dict() for r in self.results],
            "errors": self.errors,
        }


class SDXLCaptionExpander:
    """
    Expand short captions to SDXL-optimized longer captions.

    Uses OpenAI GPT-4o-mini to intelligently expand captions while:
    - Maintaining factual accuracy
    - Adding appropriate style details
    - Including quality tags
    - Generating negative prompts
    """

    # System prompt for caption expansion
    SYSTEM_PROMPT = """You are an expert at expanding image captions for SDXL training.

Your task is to expand short captions into longer, more detailed versions that work well with SDXL.

Guidelines:
1. PRESERVE the original meaning and all factual details
2. ADD visual details that would naturally be present (lighting, composition, background elements)
3. EXPAND character descriptions (clothing details, pose nuances, expression subtleties)
4. INCLUDE artistic style descriptors appropriate to the image type
5. MAINTAIN a natural, flowing description (not just keyword stuffing)
6. TARGET 77-120 tokens (current SD1.5 captions are usually 40-60 tokens)
7. DO NOT add details that contradict or weren't implied in the original
8. DO NOT add fictional elements or change the scene dramatically

Output format: Return ONLY the expanded caption as plain text, no quotes or formatting."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 200,
        temperature: float = 0.3,
        concurrent_requests: int = 10,
        rate_limit_rpm: int = 500,
        use_stub: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize SDXL Caption Expander.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model to use. "gpt-4o-mini" recommended for cost efficiency.
            max_tokens: Maximum tokens in expanded caption
            temperature: Sampling temperature (0.0-1.0). Lower = more consistent.
            concurrent_requests: Max parallel API requests
            rate_limit_rpm: Rate limit (requests per minute)
            use_stub: Use stub mode for testing without API
            verbose: Enable verbose logging
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrent_requests = concurrent_requests
        self.rate_limit_rpm = rate_limit_rpm
        self.use_stub = use_stub
        self.verbose = verbose

        # Cost tracking
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_captions = 0

        # Rate limiting
        self._request_times: List[float] = []

        # Setup logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        # Initialize client
        if use_stub or not OPENAI_AVAILABLE or not self.api_key:
            self.use_stub = True
            if not self.api_key:
                logger.warning("OPENAI_API_KEY not found in environment")
            logger.info("Using stub mode for SDXL caption expansion")
            self.client = None
            self.async_client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)
            logger.info(f"SDXLCaptionExpander initialized with model: {model}")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)."""
        return len(text) // 4 + 1

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        pricing = OPENAI_PRICING.get(self.model, OPENAI_PRICING["gpt-4o-mini"])
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    def _get_quality_prefix(self, style: str) -> str:
        """Get quality prefix for style."""
        return SDXL_QUALITY_PREFIXES.get(style, SDXL_QUALITY_PREFIXES["default"])

    def _get_technical_details(self, style: str) -> str:
        """Get technical details for style."""
        return SDXL_TECHNICAL_DETAILS.get(style, SDXL_TECHNICAL_DETAILS["default"])

    def _get_negative_prompt(self, style: str) -> str:
        """Get negative prompt for style."""
        return SDXL_NEGATIVE_PROMPTS.get(style, SDXL_NEGATIVE_PROMPTS["default"])

    def _build_user_prompt(self, caption: str, style: str) -> str:
        """Build the user prompt for caption expansion."""
        style_hints = {
            "2d": "This is a 2D animated character. Focus on animation style, line work, and color choices.",
            "3d": "This is a 3D animated character (Pixar-style). Focus on rendering quality, materials, and lighting.",
            "anime": "This is an anime-style character. Focus on anime aesthetics, cel shading, and stylization.",
            "realistic": "This is a realistic image. Focus on photographic qualities and natural details.",
        }

        style_hint = style_hints.get(style, "Focus on visual details appropriate to the image style.")

        return f"""Expand this caption for SDXL training:

Original caption: "{caption}"

Style context: {style_hint}

Expand this into a detailed 77-120 token caption that maintains all original details while adding appropriate visual descriptions."""

    def _stub_expand(self, caption: str, style: str) -> str:
        """Generate stub expanded caption for testing."""
        import random

        # Add some random but plausible expansions
        lighting_options = ["soft lighting", "dramatic lighting", "natural lighting", "studio lighting"]
        composition_options = ["centered composition", "dynamic angle", "close-up view", "medium shot"]
        detail_options = ["intricate details", "high detail", "fine textures", "sharp focus"]

        expanded = f"{caption}, {random.choice(lighting_options)}, {random.choice(composition_options)}, {random.choice(detail_options)}"

        return expanded

    def expand_caption(
        self,
        caption: str,
        style: str = "2d",
        include_quality_prefix: bool = True,
        include_technical: bool = True,
        custom_negative: Optional[str] = None,
    ) -> ExpandedCaption:
        """
        Expand a single caption for SDXL training.

        Args:
            caption: Original short caption
            style: Animation style ("2d", "3d", "anime", "realistic")
            include_quality_prefix: Whether to prepend quality tags
            include_technical: Whether to append technical details
            custom_negative: Custom negative prompt (uses default if None)

        Returns:
            ExpandedCaption with all components
        """
        start_time = time.time()

        quality_prefix = self._get_quality_prefix(style) if include_quality_prefix else ""
        technical_details = self._get_technical_details(style) if include_technical else ""
        negative_prompt = custom_negative or self._get_negative_prompt(style)

        if self.use_stub:
            # Stub mode - generate synthetic expansion
            expanded = self._stub_expand(caption, style)

            # Build full caption
            parts = [p for p in [quality_prefix, expanded, technical_details] if p]
            full_caption = ", ".join(parts)

            return ExpandedCaption(
                original=caption,
                main_prompt=expanded,
                quality_prefix=quality_prefix,
                technical_details=technical_details,
                negative_prompt=negative_prompt,
                full_caption=full_caption,
                token_count=self._estimate_tokens(full_caption),
                style=style,
                model="stub",
                generation_time=time.time() - start_time,
            )

        try:
            # Call OpenAI API
            user_prompt = self._build_user_prompt(caption, style)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            expanded = response.choices[0].message.content.strip()

            # Track usage
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self._estimate_cost(input_tokens, output_tokens)

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self.total_captions += 1

            # Build full caption
            parts = [p for p in [quality_prefix, expanded, technical_details] if p]
            full_caption = ", ".join(parts)

            return ExpandedCaption(
                original=caption,
                main_prompt=expanded,
                quality_prefix=quality_prefix,
                technical_details=technical_details,
                negative_prompt=negative_prompt,
                full_caption=full_caption,
                token_count=self._estimate_tokens(full_caption),
                style=style,
                model=self.model,
                generation_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Error expanding caption: {e}")
            # Fall back to stub expansion
            expanded = self._stub_expand(caption, style)
            parts = [p for p in [quality_prefix, expanded, technical_details] if p]
            full_caption = ", ".join(parts)

            return ExpandedCaption(
                original=caption,
                main_prompt=expanded,
                quality_prefix=quality_prefix,
                technical_details=technical_details,
                negative_prompt=negative_prompt,
                full_caption=full_caption,
                token_count=self._estimate_tokens(full_caption),
                style=style,
                model="error",
                generation_time=time.time() - start_time,
            )

    async def _expand_caption_async(
        self,
        caption: str,
        style: str,
        include_quality_prefix: bool,
        include_technical: bool,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> ExpandedCaption:
        """Async version of expand_caption for batch processing."""
        if semaphore:
            async with semaphore:
                return await self._expand_caption_async_impl(
                    caption, style, include_quality_prefix, include_technical
                )
        return await self._expand_caption_async_impl(
            caption, style, include_quality_prefix, include_technical
        )

    async def _expand_caption_async_impl(
        self,
        caption: str,
        style: str,
        include_quality_prefix: bool,
        include_technical: bool,
    ) -> ExpandedCaption:
        """Implementation of async caption expansion."""
        start_time = time.time()

        quality_prefix = self._get_quality_prefix(style) if include_quality_prefix else ""
        technical_details = self._get_technical_details(style) if include_technical else ""
        negative_prompt = self._get_negative_prompt(style)

        if self.use_stub:
            await asyncio.sleep(0.05)  # Simulate API delay
            expanded = self._stub_expand(caption, style)
            parts = [p for p in [quality_prefix, expanded, technical_details] if p]
            full_caption = ", ".join(parts)

            return ExpandedCaption(
                original=caption,
                main_prompt=expanded,
                quality_prefix=quality_prefix,
                technical_details=technical_details,
                negative_prompt=negative_prompt,
                full_caption=full_caption,
                token_count=self._estimate_tokens(full_caption),
                style=style,
                model="stub",
                generation_time=time.time() - start_time,
            )

        try:
            # Rate limiting
            await self._wait_for_rate_limit()

            user_prompt = self._build_user_prompt(caption, style)

            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            expanded = response.choices[0].message.content.strip()

            # Track usage
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self._estimate_cost(input_tokens, output_tokens)

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self.total_captions += 1

            parts = [p for p in [quality_prefix, expanded, technical_details] if p]
            full_caption = ", ".join(parts)

            return ExpandedCaption(
                original=caption,
                main_prompt=expanded,
                quality_prefix=quality_prefix,
                technical_details=technical_details,
                negative_prompt=negative_prompt,
                full_caption=full_caption,
                token_count=self._estimate_tokens(full_caption),
                style=style,
                model=self.model,
                generation_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Error expanding caption: {e}")
            expanded = self._stub_expand(caption, style)
            parts = [p for p in [quality_prefix, expanded, technical_details] if p]
            full_caption = ", ".join(parts)

            return ExpandedCaption(
                original=caption,
                main_prompt=expanded,
                quality_prefix=quality_prefix,
                technical_details=technical_details,
                negative_prompt=negative_prompt,
                full_caption=full_caption,
                token_count=self._estimate_tokens(full_caption),
                style=style,
                model="error",
                generation_time=time.time() - start_time,
            )

    async def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.rate_limit_rpm:
            wait_time = 60 - (now - self._request_times[0]) + 0.1
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

        self._request_times.append(time.time())

    def batch_expand(
        self,
        captions: List[str],
        style: str = "2d",
        include_quality_prefix: bool = True,
        include_technical: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> BatchExpandResult:
        """
        Expand multiple captions with async processing.

        Args:
            captions: List of short captions to expand
            style: Animation style for all captions
            include_quality_prefix: Whether to prepend quality tags
            include_technical: Whether to append technical details
            progress_callback: Optional callback(current, total) for progress

        Returns:
            BatchExpandResult with all expanded captions and statistics
        """
        start_time = time.time()
        results: List[ExpandedCaption] = []
        errors: List[Tuple[str, str]] = []

        async def process_all():
            semaphore = asyncio.Semaphore(self.concurrent_requests)
            tasks = [
                self._expand_caption_async(
                    caption, style, include_quality_prefix, include_technical, semaphore
                )
                for caption in captions
            ]

            for i, task in enumerate(asyncio.as_completed(tasks)):
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    errors.append((captions[i] if i < len(captions) else "unknown", str(e)))

                if progress_callback:
                    progress_callback(i + 1, len(captions))

        asyncio.run(process_all())

        # Calculate statistics
        successful = sum(1 for r in results if r.model != "error")
        failed = len(results) - successful + len(errors)
        avg_tokens = sum(r.token_count for r in results) / len(results) if results else 0

        return BatchExpandResult(
            total_captions=len(captions),
            successful=successful,
            failed=failed,
            avg_token_count=avg_tokens,
            total_time=time.time() - start_time,
            results=results,
            errors=errors,
        )

    def expand_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        style: str = "2d",
        include_quality_prefix: bool = True,
        include_technical: bool = True,
        save_metadata: bool = True,
    ) -> BatchExpandResult:
        """
        Expand all captions in a directory of .txt files.

        Reads .txt files from input_dir, expands captions, and saves to output_dir.

        Args:
            input_dir: Directory containing .txt caption files
            output_dir: Where to save expanded captions (default: input_dir + "_sdxl")
            style: Animation style
            include_quality_prefix: Whether to prepend quality tags
            include_technical: Whether to append technical details
            save_metadata: Whether to save expansion metadata JSON

        Returns:
            BatchExpandResult with statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir.parent / f"{input_dir.name}_sdxl"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all .txt files
        txt_files = list(input_dir.glob("*.txt"))
        logger.info(f"Found {len(txt_files)} caption files in {input_dir}")

        if not txt_files:
            return BatchExpandResult(
                total_captions=0,
                successful=0,
                failed=0,
                avg_token_count=0.0,
                total_time=0.0,
                results=[],
            )

        # Read all captions
        captions_with_paths: List[Tuple[Path, str]] = []
        for txt_file in txt_files:
            try:
                caption = txt_file.read_text(encoding="utf-8").strip()
                captions_with_paths.append((txt_file, caption))
            except Exception as e:
                logger.warning(f"Failed to read {txt_file}: {e}")

        # Extract just captions for batch processing
        captions = [c for _, c in captions_with_paths]

        # Expand all captions
        batch_result = self.batch_expand(
            captions,
            style=style,
            include_quality_prefix=include_quality_prefix,
            include_technical=include_technical,
            progress_callback=lambda c, t: logger.info(f"Progress: {c}/{t}"),
        )

        # Save expanded captions
        for (txt_path, _), result in zip(captions_with_paths, batch_result.results):
            output_path = output_dir / txt_path.name
            output_path.write_text(result.full_caption, encoding="utf-8")

            # Optionally save negative prompt
            neg_path = output_dir / f"{txt_path.stem}_negative.txt"
            neg_path.write_text(result.negative_prompt, encoding="utf-8")

        # Save metadata
        if save_metadata:
            metadata_path = output_dir / "expansion_metadata.json"
            metadata = {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "style": style,
                "include_quality_prefix": include_quality_prefix,
                "include_technical": include_technical,
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
                "statistics": {
                    "total_captions": batch_result.total_captions,
                    "successful": batch_result.successful,
                    "failed": batch_result.failed,
                    "avg_token_count": batch_result.avg_token_count,
                    "total_time": batch_result.total_time,
                },
                "cost_report": self.get_cost_report(),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logger.info(f"Saved {len(batch_result.results)} expanded captions to {output_dir}")
        return batch_result

    def get_cost_report(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_captions": self.total_captions,
            "avg_cost_per_caption": round(self.total_cost / max(self.total_captions, 1), 6),
            "model": self.model,
        }

    def reset_cost_tracking(self):
        """Reset cost tracking counters."""
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_captions = 0


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SDXL Caption Expander")
    parser.add_argument("input", help="Caption string or directory of .txt files")
    parser.add_argument("--output", help="Output directory (for directory mode)")
    parser.add_argument("--style", default="2d", choices=["2d", "3d", "anime", "realistic"],
                        help="Animation style")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--no-quality", action="store_true", help="Disable quality prefix")
    parser.add_argument("--no-technical", action="store_true", help="Disable technical details")
    parser.add_argument("--stub", action="store_true", help="Use stub mode (no API calls)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    expander = SDXLCaptionExpander(
        model=args.model,
        use_stub=args.stub,
        verbose=args.verbose,
    )

    input_path = Path(args.input)

    if input_path.is_dir():
        # Directory mode
        result = expander.expand_directory(
            input_path,
            output_dir=args.output,
            style=args.style,
            include_quality_prefix=not args.no_quality,
            include_technical=not args.no_technical,
        )
        print(f"\nExpanded {result.successful}/{result.total_captions} captions")
        print(f"Average token count: {result.avg_token_count:.1f}")
        print(f"Total time: {result.total_time:.2f}s")
        print(f"Cost report: {expander.get_cost_report()}")
    else:
        # Single caption mode
        caption = args.input
        result = expander.expand_caption(
            caption,
            style=args.style,
            include_quality_prefix=not args.no_quality,
            include_technical=not args.no_technical,
        )
        print(f"\nOriginal: {result.original}")
        print(f"\nExpanded: {result.main_prompt}")
        print(f"\nFull caption: {result.full_caption}")
        print(f"\nNegative: {result.negative_prompt}")
        print(f"\nToken count: {result.token_count}")
        print(f"Time: {result.generation_time:.2f}s")
        print(f"Cost report: {expander.get_cost_report()}")
