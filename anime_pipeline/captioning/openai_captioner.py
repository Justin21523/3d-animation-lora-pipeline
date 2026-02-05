#!/usr/bin/env python3
"""
OpenAI GPT-4V Caption Generation for Animation Characters.

Drop-in replacement for VLMCaptioner using OpenAI API instead of local GPU models.
Reads OPENAI_API_KEY from environment (set in ~/.bashrc).

Features:
- Async batch processing for high throughput
- Rate limiting and retry logic
- Cost tracking and logging
- Same CaptionResult interface as VLMCaptioner

Usage:
    from anime_pipeline.captioning.openai_captioner import OpenAICaptioner

    captioner = OpenAICaptioner()  # Uses OPENAI_API_KEY from env
    result = captioner.generate_caption(
        image_path="/path/to/image.png",
        prefix="a 2d animated character"
    )

Author: Justin Lu
Date: 2025-12-03
"""

import os
import sys
import base64
import asyncio
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# Import shared types
from .vlm_captioner import CaptionResult, BatchCaptionResult, CAPTION_TEMPLATES, DEFAULT_PROMPTS

logger = logging.getLogger(__name__)


# Cost per 1K tokens (as of Dec 2024)
OPENAI_PRICING = {
    "gpt-4-vision-preview": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

# Image detail cost estimates (approximate tokens)
IMAGE_TOKEN_ESTIMATES = {
    "low": 85,      # ~$0.003 per image at low detail
    "high": 765,    # ~$0.03 per image at high detail (512x512 tiles)
    "auto": 400,    # Average estimate
}


class OpenAICaptioner:
    """
    Generate captions using OpenAI GPT-4V API.

    Drop-in replacement for VLMCaptioner, using OpenAI API instead of local GPU.
    Supports batch processing with rate limiting and cost tracking.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 150,
        temperature: float = 0.3,
        detail: str = "low",
        concurrent_requests: int = 10,
        rate_limit_rpm: int = 500,
        use_stub: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize OpenAI captioner.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model to use. Options: "gpt-4-vision-preview", "gpt-4o", "gpt-4o-mini"
            max_tokens: Maximum tokens in generated caption
            temperature: Sampling temperature (0.0-1.0)
            detail: Image detail level: "low" (~$0.003), "high" (~$0.03), "auto"
            concurrent_requests: Max parallel API requests
            rate_limit_rpm: Rate limit (requests per minute)
            use_stub: Use stub mode for testing without API
            verbose: Enable verbose logging
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.detail = detail
        self.concurrent_requests = concurrent_requests
        self.rate_limit_rpm = rate_limit_rpm
        self.use_stub = use_stub
        self.verbose = verbose

        # Cost tracking
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_images = 0

        # Rate limiting
        self._request_times: List[float] = []
        self._semaphore = None

        # Setup logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        # Check if we should use stub mode
        if use_stub or not OPENAI_AVAILABLE or not self.api_key:
            self.use_stub = True
            if not self.api_key:
                logger.warning("OPENAI_API_KEY not found in environment")
            logger.info("Using stub mode for OpenAI captioning")
            self.client = None
            self.async_client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)
            logger.info(f"OpenAICaptioner initialized with model: {model}")

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 for API."""
        image_path = Path(image_path)
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _get_image_mime_type(self, image_path: Union[str, Path]) -> str:
        """Get MIME type from image extension."""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(ext, "image/png")

    def _build_prompt(
        self,
        prefix: str = "a 2d animated character",
        style: str = "2d",
        custom_prompt: Optional[str] = None,
    ) -> str:
        """Build the caption prompt."""
        if custom_prompt:
            return custom_prompt

        base_prompt = DEFAULT_PROMPTS.get(style, DEFAULT_PROMPTS["2d"])
        return base_prompt

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        pricing = OPENAI_PRICING.get(self.model, OPENAI_PRICING["gpt-4o-mini"])
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    def _stub_caption(self, prefix: str = "a 2d animated character") -> str:
        """Generate stub caption for testing."""
        import random
        expressions = ["happy", "sad", "neutral", "excited", "thoughtful"]
        poses = ["standing", "sitting", "walking", "gesturing"]
        styles = ["cartoon style", "anime style", "hand-drawn"]

        return (
            f"{prefix}, {random.choice(expressions)} expression, "
            f"{random.choice(poses)}, {random.choice(styles)}, "
            f"colorful, detailed character design"
        )

    def generate_caption(
        self,
        image_path: Union[str, Path],
        prefix: str = "a 2d animated character",
        style: str = "2d",
        custom_prompt: Optional[str] = None,
    ) -> CaptionResult:
        """
        Generate caption for a single image.

        Args:
            image_path: Path to image file
            prefix: Caption prefix (e.g., "a 2d animated character")
            style: Animation style ("2d" or "3d")
            custom_prompt: Custom prompt to use instead of default

        Returns:
            CaptionResult with caption and metadata
        """
        image_path = Path(image_path)
        start_time = time.time()

        if self.use_stub:
            caption = self._stub_caption(prefix)
            return CaptionResult(
                image_path=str(image_path),
                caption=caption,
                tokens=len(caption.split()),
                model="stub",
                generation_time=time.time() - start_time,
            )

        try:
            # Encode image
            base64_image = self._encode_image(image_path)
            mime_type = self._get_image_mime_type(image_path)

            # Build prompt
            prompt = self._build_prompt(prefix, style, custom_prompt)
            prompt_with_prefix = f"{prompt}\n\nStart the caption with: \"{prefix}\""

            # Call API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_with_prefix},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": self.detail,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            # Extract caption
            caption = response.choices[0].message.content.strip()

            # Track usage
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self._estimate_cost(input_tokens, output_tokens)

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self.total_images += 1

            return CaptionResult(
                image_path=str(image_path),
                caption=caption,
                tokens=output_tokens,
                model=self.model,
                generation_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return CaptionResult(
                image_path=str(image_path),
                caption=self._stub_caption(prefix),
                tokens=0,
                model="error",
                generation_time=time.time() - start_time,
            )

    async def _generate_caption_async(
        self,
        image_path: Union[str, Path],
        prefix: str = "a 2d animated character",
        style: str = "2d",
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> CaptionResult:
        """Async version of generate_caption for batch processing."""
        if semaphore:
            async with semaphore:
                return await self._generate_caption_async_impl(image_path, prefix, style)
        return await self._generate_caption_async_impl(image_path, prefix, style)

    async def _generate_caption_async_impl(
        self,
        image_path: Union[str, Path],
        prefix: str,
        style: str,
    ) -> CaptionResult:
        """Implementation of async caption generation."""
        image_path = Path(image_path)
        start_time = time.time()

        if self.use_stub:
            await asyncio.sleep(0.1)  # Simulate API delay
            caption = self._stub_caption(prefix)
            return CaptionResult(
                image_path=str(image_path),
                caption=caption,
                tokens=len(caption.split()),
                model="stub",
                generation_time=time.time() - start_time,
            )

        try:
            # Encode image
            base64_image = self._encode_image(image_path)
            mime_type = self._get_image_mime_type(image_path)

            # Build prompt
            prompt = self._build_prompt(prefix, style)
            prompt_with_prefix = f"{prompt}\n\nStart the caption with: \"{prefix}\""

            # Rate limiting
            await self._wait_for_rate_limit()

            # Call API
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_with_prefix},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": self.detail,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            caption = response.choices[0].message.content.strip()

            # Track usage
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self._estimate_cost(input_tokens, output_tokens)

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self.total_images += 1

            return CaptionResult(
                image_path=str(image_path),
                caption=caption,
                tokens=output_tokens,
                model=self.model,
                generation_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return CaptionResult(
                image_path=str(image_path),
                caption=self._stub_caption(prefix),
                tokens=0,
                model="error",
                generation_time=time.time() - start_time,
            )

    async def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        # Clean old request times
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.rate_limit_rpm:
            # Wait until oldest request is 60s old
            wait_time = 60 - (now - self._request_times[0]) + 0.1
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

        self._request_times.append(time.time())

    def batch_caption(
        self,
        image_paths: List[Union[str, Path]],
        prefix: str = "a 2d animated character",
        style: str = "2d",
        progress_callback: Optional[callable] = None,
    ) -> BatchCaptionResult:
        """
        Generate captions for multiple images with async processing.

        Args:
            image_paths: List of paths to image files
            prefix: Caption prefix for all images
            style: Animation style ("2d" or "3d")
            progress_callback: Optional callback(current, total) for progress

        Returns:
            BatchCaptionResult with all captions and statistics
        """
        start_time = time.time()
        results: List[CaptionResult] = []

        async def process_all():
            semaphore = asyncio.Semaphore(self.concurrent_requests)
            tasks = [
                self._generate_caption_async(path, prefix, style, semaphore)
                for path in image_paths
            ]

            for i, task in enumerate(asyncio.as_completed(tasks)):
                result = await task
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, len(image_paths))

        # Run async processing
        asyncio.run(process_all())

        # Calculate statistics
        successful = sum(1 for r in results if r.model != "error")
        failed = len(results) - successful
        avg_tokens = sum(r.tokens for r in results) / len(results) if results else 0

        return BatchCaptionResult(
            total_images=len(image_paths),
            successful=successful,
            failed=failed,
            avg_tokens=avg_tokens,
            total_time=time.time() - start_time,
            results=results,
        )

    def caption_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        prefix: str = "a 2d animated character",
        style: str = "2d",
        extensions: tuple = (".png", ".jpg", ".jpeg", ".webp"),
    ) -> BatchCaptionResult:
        """
        Caption all images in a directory and save to .txt files.

        Args:
            input_dir: Directory containing images
            output_dir: Where to save caption files (default: same as input)
            prefix: Caption prefix
            style: Animation style
            extensions: Image file extensions to process

        Returns:
            BatchCaptionResult with statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_dir.glob(f"*{ext}"))
            image_paths.extend(input_dir.glob(f"*{ext.upper()}"))

        logger.info(f"Found {len(image_paths)} images in {input_dir}")

        # Generate captions
        batch_result = self.batch_caption(
            image_paths,
            prefix=prefix,
            style=style,
            progress_callback=lambda c, t: logger.info(f"Progress: {c}/{t}"),
        )

        # Save captions to .txt files
        for result in batch_result.results:
            image_name = Path(result.image_path).stem
            caption_path = output_dir / f"{image_name}.txt"
            caption_path.write_text(result.caption, encoding="utf-8")

        logger.info(f"Saved {len(batch_result.results)} captions to {output_dir}")
        return batch_result

    def get_cost_report(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_images": self.total_images,
            "avg_cost_per_image": round(self.total_cost / max(self.total_images, 1), 4),
            "model": self.model,
            "detail": self.detail,
        }

    def reset_cost_tracking(self):
        """Reset cost tracking counters."""
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_images = 0


def apply_template(
    template_name: str,
    **kwargs
) -> str:
    """
    Apply a caption template with provided values.

    Args:
        template_name: Name of template from CAPTION_TEMPLATES
        **kwargs: Values to fill in template

    Returns:
        Formatted caption string
    """
    template = CAPTION_TEMPLATES.get(template_name, CAPTION_TEMPLATES["2d_character"])
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.warning(f"Missing template key: {e}")
        return template


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI GPT-4V Caption Generator")
    parser.add_argument("image_path", help="Path to image file or directory")
    parser.add_argument("--prefix", default="a 2d animated character", help="Caption prefix")
    parser.add_argument("--style", default="2d", choices=["2d", "3d"], help="Animation style")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--detail", default="low", choices=["low", "high", "auto"], help="Image detail level")
    parser.add_argument("--output-dir", help="Output directory for captions")
    parser.add_argument("--stub", action="store_true", help="Use stub mode (no API calls)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    captioner = OpenAICaptioner(
        model=args.model,
        detail=args.detail,
        use_stub=args.stub,
        verbose=args.verbose,
    )

    path = Path(args.image_path)

    if path.is_dir():
        # Caption entire directory
        result = captioner.caption_directory(
            path,
            output_dir=args.output_dir,
            prefix=args.prefix,
            style=args.style,
        )
        print(f"\nProcessed {result.total_images} images")
        print(f"Successful: {result.successful}, Failed: {result.failed}")
        print(f"Total time: {result.total_time:.2f}s")
        print(f"Cost report: {captioner.get_cost_report()}")
    else:
        # Single image
        result = captioner.generate_caption(
            path,
            prefix=args.prefix,
            style=args.style,
        )
        print(f"\nCaption: {result.caption}")
        print(f"Tokens: {result.tokens}")
        print(f"Time: {result.generation_time:.2f}s")
        print(f"Cost report: {captioner.get_cost_report()}")
