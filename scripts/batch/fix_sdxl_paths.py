#!/usr/bin/env python3
"""Fix SDXL Config Paths - Change SD1.5 paths to SDXL paths"""

from pathlib import Path
import re

def fix_config_paths():
    config_dir = Path("configs/training/character_loras_sdxl")

    configs_to_fix = [
        "onward_barley_lightfoot_identity_sdxl.toml",
        "onward_ian_lightfoot_identity_sdxl.toml",
        "orion_orion_identity_sdxl.toml",
        "turning-red_tyler_identity_sdxl.toml",
        "up_russell_identity_sdxl.toml",
    ]

    print("=" * 70)
    print("Fixing SDXL Config Paths")
    print("=" * 70 + "\n")

    for config_name in configs_to_fix:
        config_path = config_dir / config_name

        if not config_path.exists():
            print(f"⚠️  Not found: {config_name}")
            continue

        content = config_path.read_text()

        # Replace training_data with training_data_sdxl
        new_content = content.replace(
            "/lora_data/training_data/",
            "/lora_data/training_data_sdxl/"
        )

        if content != new_content:
            config_path.write_text(new_content)
            print(f"✅ Fixed: {config_name}")
        else:
            print(f"⚠️  No changes needed: {config_name}")

    print("\n" + "=" * 70)
    print("✅ Path fixes complete")
    print("=" * 70)

if __name__ == "__main__":
    fix_config_paths()
