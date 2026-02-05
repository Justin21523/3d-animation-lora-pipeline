#!/usr/bin/env python3
"""Verify SDXL Training Configs"""

from pathlib import Path
import re

def verify_configs():
    config_dir = Path("configs/training/character_loras_sdxl")

    print("=" * 70)
    print("SDXL Training Configuration Verification")
    print("=" * 70)

    configs = sorted(config_dir.glob("*.toml"))
    print(f"\nFound {len(configs)} SDXL configs:\n")

    all_valid = True

    for config_path in configs:
        char_name = config_path.stem.replace("_sdxl", "")
        content = config_path.read_text()

        # Extract train_data_dir
        match = re.search(r'train_data_dir = "([^"]+)"', content)
        if not match:
            print(f"❌ {char_name}: No train_data_dir found")
            all_valid = False
            continue

        train_dir = Path(match.group(1))

        # Check if directory exists
        if not train_dir.exists():
            print(f"❌ {char_name}: Directory not found - {train_dir}")
            all_valid = False
            continue

        # Count captions
        caption_count = len(list(train_dir.rglob("*.txt")))

        # Check if using correct SDXL path
        if "training_data_sdxl" not in str(train_dir):
            print(f"⚠️  {char_name}: Using SD1.5 path - {train_dir}")
            all_valid = False
        else:
            print(f"✅ {char_name}: {caption_count} captions - {train_dir.name}")

    print("\n" + "=" * 70)
    if all_valid:
        print("✅ All SDXL configs are valid!")
    else:
        print("⚠️  Some configs need attention")
    print("=" * 70)

    return all_valid

if __name__ == "__main__":
    verify_configs()
