#!/usr/bin/env python3
"""Verify all SDXL configs have total steps ≤ 35,000"""

from pathlib import Path
import re

def verify_steps():
    config_dir = Path("configs/training/character_loras_sdxl")

    print("=" * 80)
    print("SDXL Training Steps Verification (Max: 35,000 steps)")
    print("=" * 80 + "\n")

    all_valid = True
    results = []

    for config_path in sorted(config_dir.glob("*.toml")):
        char_name = config_path.stem.replace("_sdxl", "")
        content = config_path.read_text()

        # Extract total steps from header comment
        match = re.search(r'(\d+) total steps', content)
        if match:
            total_steps = int(match.group(1))
            results.append((char_name, total_steps))

            status = "✅" if total_steps <= 35000 else "❌"
            print(f"{status} {char_name:40s} {total_steps:6d} steps")

            if total_steps > 35000:
                all_valid = False

    print("\n" + "=" * 80)
    if all_valid:
        print("✅ All configs have total steps ≤ 35,000")
    else:
        print("❌ Some configs exceed 35,000 steps - need adjustment")
    print("=" * 80)

    return all_valid

if __name__ == "__main__":
    verify_steps()
