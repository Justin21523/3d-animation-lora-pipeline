#!/usr/bin/env python3
"""
Download VLM models for 3D animation captioning

Downloads Qwen2-VL and InternVL2 models with quantization support for efficient inference.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import argparse

def download_model_hf(
    model_id: str,
    save_dir: Path,
    quantization: Optional[str] = None
) -> bool:
    """
    Download model from Hugging Face Hub

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen2-VL-7B-Instruct")
        save_dir: Local directory to save model
        quantization: Quantization method (None, "int8", "int4")

    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import snapshot_download

        print(f"\n{'='*60}")
        print(f"Downloading: {model_id}")
        print(f"Save to: {save_dir}")
        if quantization:
            print(f"Quantization: {quantization}")
        print(f"{'='*60}\n")

        # Create save directory
        save_dir.mkdir(parents=True, exist_ok=True)

        # Download model
        snapshot_download(
            repo_id=model_id,
            local_dir=str(save_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4
        )

        print(f"\n✓ Successfully downloaded {model_id}")
        return True

    except Exception as e:
        print(f"\n✗ Failed to download {model_id}: {e}")
        return False


def download_qwen2_vl(base_dir: Path, variant: str = "7B") -> bool:
    """
    Download Qwen2-VL model

    Args:
        base_dir: Base directory for VLM models
        variant: Model size variant (7B, 2B)

    Returns:
        True if successful
    """
    model_ids = {
        "7B": "Qwen/Qwen2-VL-7B-Instruct",
        "2B": "Qwen/Qwen2-VL-2B-Instruct"
    }

    model_id = model_ids.get(variant)
    if not model_id:
        print(f"Unknown variant: {variant}. Available: {list(model_ids.keys())}")
        return False

    save_dir = base_dir / f"Qwen2-VL-{variant}-Instruct"
    return download_model_hf(model_id, save_dir)


def download_internvl2(base_dir: Path, variant: str = "8B") -> bool:
    """
    Download InternVL2 model

    Args:
        base_dir: Base directory for VLM models
        variant: Model size variant (8B, 4B, 2B)

    Returns:
        True if successful
    """
    model_ids = {
        "26B": "OpenGVLab/InternVL2-26B",
        "8B": "OpenGVLab/InternVL2-8B",
        "4B": "OpenGVLab/InternVL2-4B",
        "2B": "OpenGVLab/InternVL2-2B"
    }

    model_id = model_ids.get(variant)
    if not model_id:
        print(f"Unknown variant: {variant}. Available: {list(model_ids.keys())}")
        return False

    save_dir = base_dir / f"InternVL2-{variant}"
    return download_model_hf(model_id, save_dir)


def download_blip2(base_dir: Path) -> bool:
    """
    Download BLIP2 model (fallback option)

    Args:
        base_dir: Base directory for VLM models

    Returns:
        True if successful
    """
    model_id = "Salesforce/blip2-opt-2.7b"
    save_dir = base_dir / "blip2-opt-2.7b"
    return download_model_hf(model_id, save_dir)


def download_clip_models(base_dir: Path) -> bool:
    """
    Download CLIP models for embeddings

    Args:
        base_dir: Base directory for CLIP models

    Returns:
        True if successful
    """
    try:
        import open_clip
        import torch

        base_dir.mkdir(parents=True, exist_ok=True)

        models = [
            ("ViT-L-14", "openai"),
            ("ViT-B-32", "openai")
        ]

        for model_name, pretrained in models:
            print(f"\nDownloading CLIP: {model_name} ({pretrained})")
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=pretrained,
                    cache_dir=str(base_dir)
                )
                print(f"✓ Downloaded {model_name}")
            except Exception as e:
                print(f"✗ Failed to download {model_name}: {e}")
                return False

        return True

    except ImportError:
        print("✗ open_clip not installed. Installing...")
        os.system("conda run -n ai_env pip install open-clip-torch")
        return download_clip_models(base_dir)


def main():
    parser = argparse.ArgumentParser(description="Download VLM models for 3D animation captioning")
    parser.add_argument(
        "--model-warehouse",
        type=str,
        default="/mnt/c/AI_LLM_projects/ai_warehouse/models",
        help="Base model warehouse directory"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["qwen2_vl", "internvl2", "blip2", "clip", "all"],
        default=["all"],
        help="Models to download"
    )
    parser.add_argument(
        "--qwen-variant",
        type=str,
        default="7B",
        choices=["7B", "2B"],
        help="Qwen2-VL model size"
    )
    parser.add_argument(
        "--internvl-variant",
        type=str,
        default="8B",
        choices=["26B", "8B", "4B", "2B"],
        help="InternVL2 model size"
    )

    args = parser.parse_args()

    base_dir = Path(args.model_warehouse)
    vlm_dir = base_dir / "vlm"
    clip_dir = base_dir / "clip"

    # Expand "all" to all models
    models = args.models
    if "all" in models:
        models = ["qwen2_vl", "internvl2", "blip2", "clip"]

    print("\n" + "="*60)
    print("VLM Model Download for 3D Animation Pipeline")
    print("="*60)
    print(f"Model warehouse: {base_dir}")
    print(f"Models to download: {', '.join(models)}")
    print("="*60 + "\n")

    # Check for huggingface_hub
    try:
        import huggingface_hub
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("conda run -n ai_env pip install huggingface_hub")

    results = {}

    # Download each model
    if "qwen2_vl" in models:
        results["Qwen2-VL"] = download_qwen2_vl(vlm_dir, args.qwen_variant)

    if "internvl2" in models:
        results["InternVL2"] = download_internvl2(vlm_dir, args.internvl_variant)

    if "blip2" in models:
        results["BLIP2"] = download_blip2(vlm_dir)

    if "clip" in models:
        results["CLIP"] = download_clip_models(clip_dir)

    # Print summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    for model, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{model:20s} {status}")
    print("="*60 + "\n")

    # Exit with appropriate code
    if all(results.values()):
        print("All models downloaded successfully!")
        return 0
    else:
        print("Some models failed to download. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
