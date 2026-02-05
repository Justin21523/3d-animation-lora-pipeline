#!/usr/bin/env python3
"""
Test Final Endou Mamoru LoRA with comprehensive prompts
"""

import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime

# Paths
LORA_PATH = "/mnt/data/ai_data/models/lora_sdxl/inazuma-eleven/endou_mamoru_identity/inazuma_endou_mamoru_lora_sdxl.safetensors"
BASE_MODEL = "/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
VAE_PATH = "/mnt/c/ai_models/stable-diffusion/vae/sdxl_vae.safetensors"
OUTPUT_DIR = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/outputs/lora_testing/endou_mamoru_final")

# Comprehensive Test Prompts
TEST_PROMPTS = {
    "timeline_original_neutral": {
        "prompt": "inazuma_endou_mamoru, timeline_original, anime style, 1boy, teenage boy, goalkeeper uniform, orange headband, spiky brown hair, big round brown eyes, white and blue soccer jersey, standing pose, neutral expression, stadium background, daylight, medium shot, high quality, masterpiece, official art",
        "description": "Original - 中立站姿"
    },
    "timeline_original_action": {
        "prompt": "inazuma_endou_mamoru, timeline_original, anime style, 1boy, goalkeeper, orange headband visible, spiky brown hair, determined expression, goalkeeper gloves, diving save action, dynamic pose, intense focus, soccer stadium, dramatic lighting, action scene, high quality, masterpiece, best quality",
        "description": "Original - 守門動作"
    },
    "timeline_go_adult_coach": {
        "prompt": "inazuma_endou_mamoru, timeline_go, anime style, 1man, adult, coach outfit, orange headband, mature appearance, confident smile, casual jacket, standing, training ground, warm lighting, high quality, masterpiece, official art",
        "description": "GO - 成人教練"
    },
    "timeline_ares_match": {
        "prompt": "inazuma_endou_mamoru, timeline_ares, anime style, 1boy, teenage boy, soccer match, goalkeeper uniform, orange headband, spiky brown hair, shouting, fist pump, energetic, stadium crowd background, high quality, masterpiece",
        "description": "Ares - 比賽場景"
    },
    "timeline_orion_captain": {
        "prompt": "inazuma_endou_mamoru, timeline_orion, anime style, 1boy, teenage boy, captain armband, orange headband, leadership pose, confident expression, pointing forward, national team uniform, inspiring, high quality, masterpiece, official art",
        "description": "Orion - 隊長"
    },
    "close_up_portrait": {
        "prompt": "inazuma_endou_mamoru, anime style, close-up portrait, 1boy, teenage boy, orange headband, spiky brown hair, big round brown eyes, friendly smile, detailed eyes, detailed face, professional anime art, studio lighting, white background, high quality, masterpiece, best quality",
        "description": "特寫肖像"
    },
    "full_body_reference": {
        "prompt": "inazuma_endou_mamoru, anime style, full body, 1boy, teenage boy, goalkeeper uniform, orange headband, spiky brown hair, standing straight, confident pose, white background, character sheet, high quality, masterpiece, official art",
        "description": "全身參考"
    },
    "training_scene": {
        "prompt": "inazuma_endou_mamoru, anime style, 1boy, goalkeeper training, orange headband, goalkeeper gloves, catching soccer ball, training ground, daytime, energetic, focused expression, high quality, masterpiece",
        "description": "訓練場景"
    },
    "casual_outfit": {
        "prompt": "inazuma_endou_mamoru, anime style, 1boy, teenage boy, casual clothes, orange headband, spiky brown hair, t-shirt, jeans, relaxed pose, school background, friendly smile, high quality, masterpiece",
        "description": "便服造型"
    },
    "technique_god_hand": {
        "prompt": "inazuma_endou_mamoru, anime style, 1boy, goalkeeper, orange headband, God Hand technique, glowing golden energy, catching ball, dramatic pose, special move, intense expression, stadium, high quality, masterpiece, official art",
        "description": "必殺技 - God Hand"
    }
}

# Comprehensive Negative Prompt (SDXL Optimized)
NEGATIVE_PROMPT = """
lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,
deformed, disfigured, ugly, mutilated, extra limbs, missing limbs, floating limbs, disconnected limbs, malformed hands, long neck, long body, mutated hands and fingers, poorly drawn hands, poorly drawn face,
mutation, bad proportions, gross proportions, duplicate, morbid, extra legs, extra arms, disfigured face, cloned face, multiple heads, multiple people, 2girls, 2boys, multiple boys, group, crowd,
3d, realistic, photo, photorealistic, cgi, render, unfinished, black and white, monochrome, greyscale,
bad eyes, cross-eyed, uneven eyes, lazy eye, asymmetric eyes, watermark, logo, signature, text overlay, title, subtitle, date, copyright,
blur, blurry, out of focus, depth of field bokeh, motion blur, bad quality, compressed, low resolution, pixelated,
nude, nsfw, explicit, sexual, pornographic, adult content, violence, gore, blood
""".strip()

# Test Settings
LORA_SCALE = 0.8  # Default LoRA strength
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 7.5
WIDTH = 1024
HEIGHT = 1024
SEED = 42


def load_pipeline(device="cuda"):
    """Load SDXL pipeline"""
    print("正在載入 SDXL pipeline...")

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
    except:
        print("⚠ xformers not available, using default attention")
    pipe.enable_vae_tiling()

    print("✓ Pipeline 載入完成")
    return pipe


def generate_with_lora(pipe, prompt, negative_prompt, lora_scale, seed):
    """Generate image with LoRA"""

    # Load LoRA
    pipe.load_lora_weights(LORA_PATH)

    # Generate
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        width=WIDTH,
        height=HEIGHT,
        generator=generator,
        cross_attention_kwargs={"scale": lora_scale}
    ).images[0]

    # Unload LoRA
    pipe.unload_lora_weights()

    return image


def add_label(image, text):
    """Add text label to image"""
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
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


def main():
    print("="*70)
    print("🎯 Endou Mamoru 最終 LoRA 完整測試")
    print("="*70)
    print(f"LoRA: {LORA_PATH}")
    print(f"輸出: {OUTPUT_DIR}")
    print(f"測試數量: {len(TEST_PROMPTS)} 個場景")
    print(f"LoRA Scale: {LORA_SCALE}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    pipe = load_pipeline()

    # Test results
    results = {
        "lora_path": str(LORA_PATH),
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "lora_scale": LORA_SCALE,
            "steps": NUM_INFERENCE_STEPS,
            "guidance": GUIDANCE_SCALE,
            "resolution": f"{WIDTH}x{HEIGHT}",
            "seed": SEED
        },
        "prompts": {}
    }

    # Generate for each prompt
    for idx, (prompt_id, prompt_data) in enumerate(TEST_PROMPTS.items(), 1):
        print(f"\n[{idx}/{len(TEST_PROMPTS)}] {prompt_data['description']}")
        print(f"Prompt: {prompt_data['prompt'][:80]}...")

        try:
            # Generate image
            image = generate_with_lora(
                pipe,
                prompt=prompt_data["prompt"],
                negative_prompt=NEGATIVE_PROMPT,
                lora_scale=LORA_SCALE,
                seed=SEED + idx  # Different seed per prompt
            )

            # Add label
            labeled_image = add_label(image.copy(), prompt_data['description'])

            # Save
            output_path = OUTPUT_DIR / f"{prompt_id}.png"
            labeled_image.save(output_path)

            print(f"✓ 已儲存: {output_path.name}")

            results["prompts"][prompt_id] = {
                "description": prompt_data["description"],
                "prompt": prompt_data["prompt"],
                "output": str(output_path),
                "status": "success"
            }

        except Exception as e:
            print(f"✗ 錯誤: {e}")
            results["prompts"][prompt_id] = {
                "description": prompt_data["description"],
                "status": "error",
                "error": str(e)
            }

    # Save results
    results_path = OUTPUT_DIR / "test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("✅ 測試完成！")
    print(f"{'='*70}")
    print(f"結果已儲存至: {OUTPUT_DIR}")
    print(f"JSON metadata: {results_path}")
    print(f"\n生成圖片:")
    for prompt_id in TEST_PROMPTS.keys():
        img_path = OUTPUT_DIR / f"{prompt_id}.png"
        if img_path.exists():
            print(f"  ✓ {img_path.name}")


if __name__ == "__main__":
    main()
