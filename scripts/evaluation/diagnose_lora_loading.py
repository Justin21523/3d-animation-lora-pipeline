#!/usr/bin/env python3
"""
Diagnostic test to verify LoRA is actually being loaded and applied.
Generates the same prompt with/without LoRA and at different scales.
"""

import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
import json

# Paths
LORA_PATH = "/mnt/data/ai_data/models/lora_sdxl/inazuma-eleven/endou_mamoru_identity/inazuma_endou_mamoru_lora_sdxl.safetensors"
BASE_MODEL = "/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
VAE_PATH = "/mnt/c/ai_models/stable-diffusion/vae/sdxl_vae.safetensors"
OUTPUT_DIR = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/outputs/lora_testing/endou_diagnostic")

# Test prompt (same for all)
TEST_PROMPT = "inazuma_endou_mamoru, timeline_original, anime style, 1boy, teenage boy, goalkeeper uniform, orange headband, spiky brown hair, big round brown eyes, white and blue soccer jersey, standing pose, neutral expression, stadium background, daylight, medium shot, high quality, masterpiece, official art"

NEGATIVE_PROMPT = """
lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,
deformed, disfigured, ugly, mutilated, extra limbs, missing limbs, floating limbs, disconnected limbs, malformed hands, long neck, long body, mutated hands and fingers, poorly drawn hands, poorly drawn face,
mutation, bad proportions, gross proportions, duplicate, morbid, extra legs, extra arms, disfigured face, cloned face, multiple heads, multiple people, 2girls, 2boys, multiple boys, group, crowd,
3d, realistic, photo, photorealistic, cgi, render, unfinished, black and white, monochrome, greyscale,
bad eyes, cross-eyed, uneven eyes, lazy eye, asymmetric eyes, watermark, logo, signature, text overlay, title, subtitle, date, copyright,
blur, blurry, out of focus, depth of field bokeh, motion blur, bad quality, compressed, low resolution, pixelated,
nude, nsfw, explicit, sexual, pornographic, adult content, violence, gore, blood
""".strip()

# Settings
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 7.5
WIDTH = 1024
HEIGHT = 1024
SEED = 42


def load_pipeline(device="cuda"):
    """Load SDXL pipeline"""
    print("載入 SDXL pipeline...")

    # Load VAE
    vae = AutoencoderKL.from_single_file(
        VAE_PATH,
        torch_dtype=torch.float16
    )

    # Load pipeline
    pipe = StableDiffusionXLPipeline.from_single_file(
        BASE_MODEL,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.to(device)

    # Optimizations
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✓ xformers enabled")
    except:
        print("⚠ xformers not available, using default attention")
    pipe.enable_vae_tiling()

    print("✓ Pipeline loaded")
    return pipe


def generate_baseline(pipe, seed):
    """Generate WITHOUT LoRA (baseline)"""
    print("\n[1/5] Generating BASELINE (no LoRA)...")

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    image = pipe(
        prompt=TEST_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        width=WIDTH,
        height=HEIGHT,
        generator=generator
    ).images[0]

    return image


def generate_with_lora(pipe, lora_scale, seed):
    """Generate WITH LoRA at specific scale"""
    print(f"\n[LoRA Scale {lora_scale}] Generating...")

    # Load LoRA weights
    print(f"  Loading LoRA from: {LORA_PATH}")
    pipe.load_lora_weights(LORA_PATH)

    # Check if LoRA was loaded
    if hasattr(pipe, 'text_encoder'):
        print(f"  Text encoder type: {type(pipe.text_encoder)}")
    if hasattr(pipe, 'unet'):
        print(f"  UNet type: {type(pipe.unet)}")

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # Generate with LoRA
    image = pipe(
        prompt=TEST_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        width=WIDTH,
        height=HEIGHT,
        generator=generator,
        cross_attention_kwargs={"scale": lora_scale}
    ).images[0]

    # Unload LoRA
    pipe.unload_lora_weights()
    print(f"  ✓ Generated with LoRA scale {lora_scale}")

    return image


def add_label(image, text):
    """Add label to image"""
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (image.width - text_width) // 2
    y = 20

    # Background
    padding = 15
    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill=(0, 0, 0, 220)
    )

    # Text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return image


def create_comparison_grid(images, labels):
    """Create side-by-side comparison"""
    w, h = images[0].size
    cols = len(images)

    grid = Image.new('RGB', (w * cols, h), color=(255, 255, 255))

    for idx, (img, label) in enumerate(zip(images, labels)):
        labeled = add_label(img.copy(), label)
        grid.paste(labeled, (idx * w, 0))

    return grid


def main():
    print("="*70)
    print("🔍 LoRA Loading Diagnostic Test")
    print("="*70)
    print(f"LoRA: {LORA_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Test: Same prompt with/without LoRA at multiple scales")
    print()

    # Verify LoRA file exists
    lora_file = Path(LORA_PATH)
    if not lora_file.exists():
        print(f"❌ ERROR: LoRA file not found at {LORA_PATH}")
        return

    print(f"✓ LoRA file found: {lora_file.stat().st_size / (1024**2):.1f} MB")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    pipe = load_pipeline()

    # Test configurations
    tests = [
        ("baseline", None, "No LoRA (Baseline)"),
        ("lora_0.5", 0.5, "LoRA Scale 0.5"),
        ("lora_0.8", 0.8, "LoRA Scale 0.8"),
        ("lora_1.0", 1.0, "LoRA Scale 1.0"),
        ("lora_1.2", 1.2, "LoRA Scale 1.2")
    ]

    images = []
    labels = []
    results = {}

    # Generate baseline
    baseline_img = generate_baseline(pipe, SEED)
    baseline_path = OUTPUT_DIR / "00_baseline.png"
    baseline_img.save(baseline_path)
    images.append(baseline_img)
    labels.append("No LoRA")
    results["baseline"] = str(baseline_path)

    # Generate with LoRA at different scales
    for idx, (test_id, scale, label) in enumerate(tests[1:], 2):
        img = generate_with_lora(pipe, scale, SEED)
        img_path = OUTPUT_DIR / f"{idx:02d}_{test_id}.png"
        img.save(img_path)
        images.append(img)
        labels.append(label)
        results[test_id] = str(img_path)

    # Create comparison grid
    print("\n創建對比網格...")
    grid = create_comparison_grid(images, labels)
    grid_path = OUTPUT_DIR / "comparison_grid.png"
    grid.save(grid_path)

    # Save metadata
    metadata = {
        "lora_path": str(LORA_PATH),
        "lora_size_mb": lora_file.stat().st_size / (1024**2),
        "test_prompt": TEST_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "settings": {
            "steps": NUM_INFERENCE_STEPS,
            "guidance": GUIDANCE_SCALE,
            "resolution": f"{WIDTH}x{HEIGHT}",
            "seed": SEED
        },
        "results": results
    }

    metadata_path = OUTPUT_DIR / "diagnostic_results.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("✅ 診斷測試完成！")
    print(f"{'='*70}")
    print(f"\n輸出目錄: {OUTPUT_DIR}")
    print(f"\n個別圖片:")
    for test_id, path in results.items():
        print(f"  {test_id}: {Path(path).name}")
    print(f"\n對比網格: {grid_path.name}")
    print(f"Metadata: {metadata_path.name}")
    print()
    print("請檢查圖片差異:")
    print("  - 如果所有圖片都一樣 → LoRA 沒有被正確載入")
    print("  - 如果 LoRA 圖片有差異但不像角色 → 訓練問題")
    print("  - 如果 LoRA 圖片與 baseline 不同且像角色 → LoRA 正常工作")


if __name__ == "__main__":
    main()
