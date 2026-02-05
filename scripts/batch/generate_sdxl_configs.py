#!/usr/bin/env python3
"""
Generate SDXL Training Configs for All Characters
Based on Giulia's working SDXL config template
"""

from pathlib import Path

# Character definitions: (film, char_id, char_name, image_count, repeats, epochs)
# Strategy: Higher epochs (8/10/12), adjusted repeats to keep total steps ≤ 35,000
CHARACTERS = [
    ("luca", "alberto", "Alberto Scorfano", 509, 6, 10),
    ("luca", "giulia", "Giulia Marcovaldo", 546, 8, 8),
    ("coco", "miguel", "Miguel Rivera", 449, 7, 10),
    ("elio", "elio", "Elio Solis", 538, 8, 8),
    ("elio", "bryce", "Bryce Markwell", 201, 14, 12),
    ("elio", "caleb", "Caleb", 195, 14, 12),
    ("elio", "glordon", "Glordon", 201, 14, 12),
    ("onward", "ian_lightfoot", "Ian Lightfoot", 343, 10, 10),
    ("onward", "barley_lightfoot", "Barley Lightfoot", 254, 11, 12),
    ("up", "russell", "Russell", 243, 12, 12),
    ("orion", "orion", "Orion", 261, 13, 10),
    ("turning-red", "tyler", "Tyler", 276, 12, 10),
]

TEMPLATE = """# SDXL Character Identity LoRA Training Config
# Character: {char_name} ({film_title})
# Dataset: {image_count} images × {repeats} repeats = {steps_per_epoch} steps/epoch
# Target: {epochs} epochs ({total_steps} total steps)

[model]
pretrained_model_name_or_path = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
v2 = false
v_parameterization = false
sdxl = true

network_module = "networks.lora"
network_dim = 128
network_alpha = 96
network_dropout = 0.1
network_args = []

[paths]
train_data_dir = "/mnt/data/ai_data/datasets/3d-anime/{film}/lora_data/training_data_sdxl/{char_id}_identity"
output_dir = "/mnt/data/ai_data/models/lora_sdxl/{film}/{char_id}_identity"
output_name = "{char_id}_identity_lora_sdxl"
logging_dir = "/mnt/data/ai_data/models/lora_sdxl/{film}/{char_id}_identity/logs"
log_prefix = "{char_id}_sdxl"

[training]
optimizer_type = "AdamW8bit"
mixed_precision = "bf16"
full_bf16 = true
gradient_checkpointing = true

train_batch_size = 1
gradient_accumulation_steps = 2

learning_rate = 0.0001
text_encoder_lr = 0.00006
unet_lr = 0.0001

lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 3
lr_warmup_steps = 100

min_snr_gamma = 5.0
noise_offset = 0.05
adaptive_noise_scale = 0.0
multires_noise_iterations = 0
multires_noise_discount = 0.3

max_train_epochs = {epochs}
save_every_n_epochs = 2
save_last_n_epochs = 3

cache_latents = true
cache_latents_to_disk = false
vae_batch_size = 2

[resolution]
resolution = "1024,1024"
enable_bucket = true
min_bucket_reso = 640
max_bucket_reso = 1536
bucket_reso_steps = 64
bucket_no_upscale = false

[augmentation]
color_aug = false
flip_aug = false
random_crop = false
shuffle_caption = true
keep_tokens = 1

[sample_generation]
sample_every_n_epochs = 2
sample_prompts = "/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/prompts/lora_testing/{char_id}_identity_test.txt"
sample_sampler = "euler_a"

[performance]
persistent_data_loader_workers = true
max_data_loader_n_workers = 2
lowram = false
max_token_length = 225
seed = 42
clip_skip = 2
max_grad_norm = 1.0
"""

def main():
    project_root = Path("/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline")
    output_dir = project_root / "configs/training/character_loras_sdxl"
    output_dir.mkdir(parents=True, exist_ok=True)

    film_titles = {
        "luca": "Luca, 2021",
        "coco": "Coco, 2017",
        "elio": "Elio, 2025",
        "onward": "Onward, 2020",
        "up": "Up, 2009",
        "orion": "Orion and the Dark, 2024",
        "turning-red": "Turning Red, 2022",
    }

    for film, char_id, char_name, image_count, repeats, epochs in CHARACTERS:
        steps_per_epoch = image_count * repeats
        total_steps = steps_per_epoch * epochs
        film_title = film_titles.get(film, film.title())

        config_content = TEMPLATE.format(
            char_name=char_name,
            film_title=film_title,
            image_count=image_count,
            repeats=repeats,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            total_steps=total_steps,
            film=film,
            char_id=char_id
        )

        output_file = output_dir / f"{film}_{char_id}_identity_sdxl.toml"
        output_file.write_text(config_content)
        print(f"✓ Created: {output_file.name}")

    print(f"\n✅ Generated {len(CHARACTERS)} SDXL training configs")

if __name__ == "__main__":
    main()
