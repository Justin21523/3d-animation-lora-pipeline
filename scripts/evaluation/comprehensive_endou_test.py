#!/usr/bin/env python3
"""
Comprehensive LoRA Testing for Endou Mamoru
Tests all checkpoints with detailed prompts and negative prompts
"""

import os
import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime

# Paths
LORA_DIR = Path("/mnt/data/ai_data/models/lora_sdxl/inazuma-eleven/endou_mamoru_identity")
BASE_MODEL = "/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
VAE_PATH = "/mnt/c/ai_models/stable-diffusion/vae/sdxl_vae.safetensors"
OUTPUT_DIR = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/outputs/lora_testing/endou_mamoru_comprehensive")

# Test Configuration
TEST_PROMPTS = {
    "timeline_original_neutral": {
        "prompt": "inazuma_endou_mamoru, timeline_original, anime style, 1boy, teenage boy, goalkeeper uniform, orange headband, spiky brown hair, big round brown eyes, white and blue soccer jersey, standing pose, neutral expression, stadium background, daylight, medium shot, high quality, masterpiece, official art",
        "description": "Original Timeline - 中立姿勢"
    },
    "timeline_original_action": {
        "prompt": "inazuma_endou_mamoru, timeline_original, anime style, 1boy, teenage boy, goalkeeper, orange headband visible, spiky brown hair, determined expression, goalkeeper gloves, diving save action, dynamic pose, intense focus, soccer stadium, dramatic lighting, action scene, high quality, masterpiece",
        "description": "Original Timeline - 守門動作"
    },
    "timeline_go_adult": {
        "prompt": "inazuma_endou_mamoru, timeline_go, anime style, 1man, adult, coach outfit, orange headband, mature face, confident smile, casual jacket, jeans, standing, training ground background, warm lighting, high quality, masterpiece, official art",
        "description": "GO Timeline - 成人教練"
    },
    "timeline_ares_match": {
        "prompt": "inazuma_endou_mamoru, timeline_ares, anime style, 1boy, teenage boy, soccer match scene, goalkeeper uniform, orange headband, spiky brown hair, shouting expression, fist pump gesture, team spirit, stadium crowd, energetic atmosphere, high quality, masterpiece",
        "description": "Ares Timeline - 比賽場景"
    },
    "timeline_orion_captain": {
        "prompt": "inazuma_endou_mamoru, timeline_orion, anime style, 1boy, teenage boy, captain armband, orange headband, leadership pose, confident expression, pointing forward, team huddle, national team uniform, inspiring atmosphere, high quality, masterpiece, official art",
        "description": "Orion Timeline - 隊長形象"
    },
    "close_up_portrait": {
        "prompt": "inazuma_endou_mamoru, anime style, close-up portrait, 1boy, teenage boy, orange headband, spiky brown hair, big round brown eyes, friendly smile, detailed eyes, detailed face, professional anime art, studio lighting, white background, high quality, masterpiece",
        "description": "特寫肖像"
    },
    "training_scene": {
        "prompt": "inazuma_endou_mamoru, anime style, 1boy, teenage boy, goalkeeper training, orange headband, goalkeeper gloves, catching soccer ball, training ground, daytime, energetic pose, focused expression, high quality, masterpiece",
        "description": "訓練場景"
    },
    "full_body_standing": {
        "prompt": "inazuma_endou_mamoru, anime style, full body, 1boy, teenage boy, goalkeeper uniform, orange headband, spiky brown hair, standing straight, hands on hips, confident pose, white background, high quality, masterpiece, character reference sheet",
        "description": "全身站立 - 角色參考"
    },
    "negative_test_generic": {
        "prompt": "anime boy, soccer player, generic style, orange headband, goalkeeper",
        "description": "負面測試 - 通用提示詞（不應觸發角色）"
    },
    "negative_test_no_trigger": {
        "prompt": "anime style, 1boy, teenage boy, soccer goalkeeper, orange headband, brown hair, brown eyes, smiling",
        "description": "負面測試 - 無觸發詞（不應觸發角色）"
    }
}

# Comprehensive Negative Prompt
NEGATIVE_PROMPT = """
lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,
deformed, disfigured, ugly, mutilated, extra limbs, missing limbs, floating limbs, disconnected limbs, malformed hands, long neck, long body, mutated hands and fingers, poorly drawn hands, poorly drawn face,
mutation, bad proportions, gross proportions, duplicate, morbid, trout pout, extra legs, extra arms, disfigured face, cloned face, multiple heads, multiple people, 2girls, 2boys, multiple boys, group,
3d, realistic, photo, photorealistic, cgi, render, sketch, unfinished, black and white, monochrome, greyscale,
bad eyes, cross-eyed, uneven eyes, lazy eye, watermark, logo, signature, text, title, subtitle, date, copyright,
blur, blurry, out of focus, depth of field, bokeh, motion blur, bad quality, compressed, low resolution,
nude, nsfw, explicit, sexual, pornographic, adult content
""".strip()

# LoRA Scale Tests
LORA_SCALES = [0.6, 0.7, 0.8, 0.9, 1.0]

# Generation Settings
GEN_SETTINGS = {
    "num_inference_steps": 40,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024,
    "seed": 42
}


def load_pipeline(device="cuda"):
    """Load SDXL pipeline with VAE"""
    print("Loading SDXL pipeline...")

    # Load VAE
    vae = AutoencoderKL.from_single_file(
        VAE_PATH,
        torch_dtype=torch.float16
    )

    # Load pipeline from single file
    pipe = StableDiffusionXLPipeline.from_single_file(
        BASE_MODEL,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.to(device)

    # Enable optimizations
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_tiling()

    print("✓ Pipeline loaded")
    return pipe


def generate_image(pipe, prompt, negative_prompt, lora_path, lora_scale, seed):
    """Generate single image with LoRA"""

    # Load LoRA
    pipe.load_lora_weights(str(lora_path))

    # Set seed
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # Generate
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=GEN_SETTINGS["num_inference_steps"],
        guidance_scale=GEN_SETTINGS["guidance_scale"],
        width=GEN_SETTINGS["width"],
        height=GEN_SETTINGS["height"],
        generator=generator,
        cross_attention_kwargs={"scale": lora_scale}
    ).images[0]

    # Unload LoRA
    pipe.unload_lora_weights()

    return image


def add_text_to_image(image, text, position="bottom"):
    """Add text label to image"""
    draw = ImageDraw.Draw(image)

    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position
    if position == "bottom":
        x = (image.width - text_width) // 2
        y = image.height - text_height - 20
    elif position == "top":
        x = (image.width - text_width) // 2
        y = 20

    # Draw background rectangle
    padding = 10
    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill=(0, 0, 0, 200)
    )

    # Draw text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return image


def create_comparison_grid(images, labels, cols=3):
    """Create comparison grid of images"""
    rows = (len(images) + cols - 1) // cols

    w, h = images[0].size
    grid = Image.new('RGB', (w * cols, h * rows), color=(255, 255, 255))

    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols

        # Add label
        img_with_label = add_text_to_image(img.copy(), label, "top")

        # Paste into grid
        grid.paste(img_with_label, (col * w, row * h))

    return grid


def test_checkpoint(pipe, checkpoint_path, output_dir):
    """Test single checkpoint with all prompts"""

    checkpoint_name = checkpoint_path.stem
    print(f"\n{'='*60}")
    print(f"Testing: {checkpoint_name}")
    print(f"{'='*60}")

    # Create output directory
    ckpt_output_dir = output_dir / checkpoint_name
    ckpt_output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": checkpoint_name,
        "timestamp": datetime.now().isoformat(),
        "prompts": {}
    }

    # Test each prompt
    for prompt_id, prompt_data in TEST_PROMPTS.items():
        print(f"\n{prompt_data['description']}...")

        prompt_output_dir = ckpt_output_dir / prompt_id
        prompt_output_dir.mkdir(exist_ok=True)

        prompt_results = {
            "description": prompt_data["description"],
            "prompt": prompt_data["prompt"],
            "scales": {}
        }

        # Test different LoRA scales
        for lora_scale in LORA_SCALES:
            print(f"  LoRA scale: {lora_scale}")

            try:
                image = generate_image(
                    pipe,
                    prompt=prompt_data["prompt"],
                    negative_prompt=NEGATIVE_PROMPT,
                    lora_path=checkpoint_path,
                    lora_scale=lora_scale,
                    seed=GEN_SETTINGS["seed"]
                )

                # Save image
                output_path = prompt_output_dir / f"scale_{lora_scale:.1f}.png"
                image.save(output_path)

                prompt_results["scales"][lora_scale] = str(output_path)

            except Exception as e:
                print(f"    Error: {e}")
                prompt_results["scales"][lora_scale] = f"Error: {e}"

        results["prompts"][prompt_id] = prompt_results

    # Save results JSON
    results_path = ckpt_output_dir / "test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Checkpoint testing completed")
    print(f"  Results: {results_path}")


def create_summary_grid(output_dir):
    """Create summary comparison grid across all checkpoints"""
    print(f"\n{'='*60}")
    print("Creating summary comparison grid...")
    print(f"{'='*60}")

    # Get all checkpoint dirs
    checkpoint_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])

    # For each prompt, create a grid comparing all checkpoints at scale 0.8
    for prompt_id in TEST_PROMPTS.keys():
        images = []
        labels = []

        for ckpt_dir in checkpoint_dirs:
            img_path = ckpt_dir / prompt_id / "scale_0.8.png"
            if img_path.exists():
                images.append(Image.open(img_path))
                labels.append(ckpt_dir.name.replace("inazuma_endou_mamoru_lora_sdxl-", "Epoch "))

        if images:
            grid = create_comparison_grid(images, labels, cols=3)
            grid_path = output_dir / f"comparison_{prompt_id}.png"
            grid.save(grid_path)
            print(f"✓ Created: {grid_path.name}")


def main():
    print("="*60)
    print("Endou Mamoru LoRA Comprehensive Testing")
    print("="*60)
    print(f"LoRA Directory: {LORA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Test Prompts: {len(TEST_PROMPTS)}")
    print(f"LoRA Scales: {LORA_SCALES}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    pipe = load_pipeline()

    # Find all checkpoints
    checkpoints = sorted(LORA_DIR.glob("*.safetensors"))
    print(f"Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.name}")
    print()

    # Test each checkpoint
    for checkpoint_path in checkpoints:
        test_checkpoint(pipe, checkpoint_path, OUTPUT_DIR)

    # Create summary grids
    create_summary_grid(OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("✅ All testing completed!")
    print(f"{'='*60}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print()
    print("Check the following:")
    print("  1. Individual checkpoint results in each subdirectory")
    print("  2. Comparison grids: comparison_*.png")
    print("  3. test_results.json for detailed metadata")


if __name__ == "__main__":
    main()
