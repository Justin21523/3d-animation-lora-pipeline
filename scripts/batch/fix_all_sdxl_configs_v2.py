#!/usr/bin/env python3
"""Fix all Inazuma SDXL configs to prevent NaN loss"""

from pathlib import Path

CONFIG_DIR = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl")

CHARACTERS = [
    "endou_mamoru",    # Already fixed manually
    "gouenji_shuuya",
    "fudou_akio",
    "matsukaze_tenma",
    "inamori_asuto",
    "nosaka_yuuma",
    "utsunomiya_toramaru"
]

# Fixes to apply
FIXES = [
    # Fix 1: Lower learning rates
    {
        "old": 'learning_rate = 8e-05\nunet_lr = 8e-05\ntext_encoder_lr = 4e-05',
        "new": 'learning_rate = 5e-05\nunet_lr = 5e-05\ntext_encoder_lr = 2.5e-05'
    },
    # Fix 2: Add gradient clipping after lr_warmup_steps
    {
        "old": 'lr_warmup_steps = 100\n\n# Network architecture',
        "new": 'lr_warmup_steps = 100\n\n# Gradient clipping (CRITICAL for SDXL stability)\nmax_grad_norm = 1.0\n\n# Network architecture'
    },
    # Fix 3: Disable xformers
    {
        "old": 'xformers_memory_efficient_attention = true',
        "new": '# xformers disabled - not available in environment'
    },
    # Fix 4: Disable min_snr_gamma and noise_offset
    {
        "old": 'clip_skip = 2\nmin_snr_gamma = 5.0\nnoise_offset = 0.05',
        "new": 'clip_skip = 2\n# min_snr_gamma disabled - causes NaN loss in SDXL\n# noise_offset disabled initially for stability'
    }
]

print("=" * 70)
print("Fixing all Inazuma SDXL LoRA configs")
print("=" * 70)
print()
print("Applying fixes:")
print("  1. Learning rate: 8e-5 → 5e-5")
print("  2. Add max_grad_norm = 1.0")
print("  3. Disable xformers")
print("  4. Remove min_snr_gamma and noise_offset")
print()

for char_id in CHARACTERS:
    config_file = CONFIG_DIR / f"inazuma_{char_id}_sdxl.toml"

    if not config_file.exists():
        print(f"⚠ {config_file.name} not found, skipping")
        continue

    print(f"Processing: {char_id}...")

    content = config_file.read_text()

    # Apply all fixes
    modified = False
    for fix in FIXES:
        if fix["old"] in content:
            content = content.replace(fix["old"], fix["new"])
            modified = True

    if modified:
        config_file.write_text(content)
        print(f"  ✓ Fixed {config_file.name}")
    else:
        print(f"  → Already fixed or no changes needed")

print()
print("=" * 70)
print("✅ All configs updated!")
print("=" * 70)
print()
print("Ready for batch training with stable parameters.")
