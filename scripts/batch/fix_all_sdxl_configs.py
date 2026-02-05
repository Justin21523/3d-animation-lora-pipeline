#!/usr/bin/env python3
"""
Fix ALL SDXL character configs to match proven baseline (Alberto/Miguel)
Standard: network_dim=64, network_alpha=32, network_dropout=0.0
"""

import re
import os
from pathlib import Path

# Correct parameters based on Alberto/Miguel successful configs
CORRECT_PARAMS = {
    "network_dim": "64",
    "network_alpha": "32",
    "network_dropout": "0.0",
}

def check_and_fix_config(config_path):
    """Check if config has wrong parameters and fix them"""

    with open(config_path, 'r') as f:
        content = f.read()

    # Extract current values
    dim_match = re.search(r'network_dim = (\d+)', content)
    alpha_match = re.search(r'network_alpha = (\d+)', content)
    dropout_match = re.search(r'network_dropout = ([\d.]+)', content)

    current_dim = dim_match.group(1) if dim_match else None
    current_alpha = alpha_match.group(1) if alpha_match else None
    current_dropout = dropout_match.group(1) if dropout_match else None

    needs_fix = False
    changes = []

    if current_dim != CORRECT_PARAMS["network_dim"]:
        needs_fix = True
        changes.append(f"network_dim: {current_dim} → 64")

    if current_alpha != CORRECT_PARAMS["network_alpha"]:
        needs_fix = True
        changes.append(f"network_alpha: {current_alpha} → 32")

    if current_dropout != CORRECT_PARAMS["network_dropout"]:
        needs_fix = True
        changes.append(f"network_dropout: {current_dropout} → 0.0")

    if needs_fix:
        # Apply fixes
        content = re.sub(r'network_dim = \d+', f'network_dim = 64', content)
        content = re.sub(r'network_alpha = \d+', f'network_alpha = 32', content)
        content = re.sub(r'network_dropout = [\d.]+', f'network_dropout = 0.0', content)

        with open(config_path, 'w') as f:
            f.write(content)

        return True, changes

    return False, []

def main():
    # Find all SDXL config files (exclude backups)
    config_dir = Path("/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl")
    config_files = [f for f in config_dir.glob("*.toml") if not f.name.startswith("_")]

    print("=" * 80)
    print("SDXL Configuration Verification & Fix")
    print("=" * 80)
    print(f"\nStandard: network_dim=64, network_alpha=32, network_dropout=0.0")
    print(f"Based on: Alberto & Miguel successful configs\n")

    fixed_count = 0
    correct_count = 0

    for config_file in sorted(config_files):
        needs_fix, changes = check_and_fix_config(config_file)

        if needs_fix:
            fixed_count += 1
            print(f"❌ FIXED: {config_file.name}")
            for change in changes:
                print(f"   - {change}")
        else:
            correct_count += 1
            print(f"✅ OK: {config_file.name}")

    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  ✅ Already correct: {correct_count}")
    print(f"  ❌ Fixed: {fixed_count}")
    print(f"  📊 Total configs: {len(config_files)}")
    print("=" * 80)

    if fixed_count > 0:
        print(f"\n🔧 Fixed {fixed_count} configuration file(s)")
        print("All SDXL configs now match the proven baseline!")
    else:
        print("\n✅ All configurations are already correct!")

if __name__ == "__main__":
    main()
