#!/usr/bin/env python3
"""
Train All Super Wings SDXL LoRAs Sequentially

Launches SDXL LoRA training for all 9 Super Wings characters using
auto-generated TOML configs. Trains one character at a time to avoid
GPU memory conflicts.

Based on successful Pixar character LoRA results:
- Best checkpoints at Epoch 1-2
- network_dim=64, alpha=32
- AdamW8bit + BF16 + gradient checkpointing (16GB VRAM)

Author: LLMProvider Tooling
Date: 2025-12-13
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Paths
CONFIGS_DIR = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/super_wings_loras")
KOHYA_SCRIPT = Path.home() / "sd-scripts/train_network.py"
LOG_DIR = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/logs/super_wings_training")

# Characters
CHARACTERS = [
    "beard", "bello", "flip", "jerone", "jet",
    "paul", "shark", "tank", "tony"
]

# Training environment
CONDA_ENV = "ai_env"

def verify_environment() -> bool:
    """Verify training environment is ready"""
    print("Verifying training environment...")

    # Check Kohya_ss sd-scripts
    if not KOHYA_SCRIPT.exists():
        print(f"❌ Kohya_ss script not found: {KOHYA_SCRIPT}")
        print(f"   Please clone sd-scripts to: {KOHYA_SCRIPT.parent}")
        return False

    # Check configs directory
    if not CONFIGS_DIR.exists():
        print(f"❌ Configs directory not found: {CONFIGS_DIR}")
        print(f"   Please run: python scripts/batch/generate_super_wings_sdxl_configs.py")
        return False

    # Check for TOML configs
    configs = list(CONFIGS_DIR.glob("super-wings-*-sdxl.toml"))
    if not configs:
        print(f"❌ No training configs found in {CONFIGS_DIR}")
        print(f"   Please run: python scripts/batch/generate_super_wings_sdxl_configs.py")
        return False

    print(f"✅ Found {len(configs)} training configs")

    # Create log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    return True

def get_character_config(character: str) -> Optional[Path]:
    """Get TOML config path for a character"""
    config_path = CONFIGS_DIR / f"super-wings-{character}-sdxl.toml"

    if not config_path.exists():
        print(f"⚠️  Config not found for {character}: {config_path}")
        return None

    return config_path

def train_character(character: str, config_path: Path) -> bool:
    """Train SDXL LoRA for a single character"""

    print(f"\n{'=' * 70}")
    print(f"🚀 Training: {character.upper()}")
    print(f"{'=' * 70}")
    print(f"Config: {config_path.name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Prepare log file
    log_file = LOG_DIR / f"{character}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Training command
    cmd = [
        "conda", "run", "-n", CONDA_ENV,
        "python", str(KOHYA_SCRIPT),
        "--config_file", str(config_path)
    ]

    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print(f"Log: {log_file}")
    print()

    # Run training
    start_time = time.time()

    try:
        with open(log_file, 'w') as f:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output to both console and log file
            for line in process.stdout.splitlines():
                print(line)
                f.write(line + '\n')
                f.flush()

            # Check return code
            if process.returncode != 0:
                print(f"\n❌ Training failed for {character}!")
                print(f"   Check log: {log_file}")
                return False

        elapsed = time.time() - start_time
        print(f"\n✅ {character} training completed successfully!")
        print(f"   Duration: {elapsed/60:.1f} minutes")
        print(f"   Log: {log_file}")

        return True

    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user for {character}")
        return False
    except Exception as e:
        print(f"\n❌ Training error for {character}: {e}")
        print(f"   Check log: {log_file}")
        return False

def main():
    print("=" * 70)
    print("Super Wings SDXL LoRA Training - Sequential Launcher")
    print("=" * 70)
    print()

    # Verify environment
    if not verify_environment():
        sys.exit(1)

    # Get available configs
    available_characters = []
    for char in CHARACTERS:
        config = get_character_config(char)
        if config:
            available_characters.append((char, config))

    if not available_characters:
        print("❌ No characters available for training!")
        sys.exit(1)

    print(f"Characters ready for training: {len(available_characters)}")
    for char, config in available_characters:
        print(f"  - {char}: {config.name}")
    print()

    # Confirm start
    if sys.stdin.isatty():  # Only ask if interactive
        response = input(f"Start training all {len(available_characters)} characters? [y/N]: ").strip().lower()
        if response != 'y':
            print("Training cancelled.")
            sys.exit(0)

    # Train all characters sequentially
    successful = []
    failed = []

    print(f"\n{'=' * 70}")
    print("Starting Sequential Training")
    print(f"{'=' * 70}\n")

    total_start = time.time()

    for i, (character, config) in enumerate(available_characters, 1):
        print(f"\n[{i}/{len(available_characters)}] Training {character}...")

        if train_character(character, config):
            successful.append(character)
        else:
            failed.append(character)

            # Ask whether to continue after failure
            if sys.stdin.isatty() and i < len(available_characters):
                response = input(f"\n⚠️  Continue with remaining characters? [Y/n]: ").strip().lower()
                if response == 'n':
                    print("Training sequence stopped.")
                    break

    # Final summary
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"✅ Successful: {len(successful)}/{len(available_characters)}")
    for char in successful:
        output_dir = Path(f"/mnt/c/ai_models/lora_sdxl/super-wings/{char}")
        if output_dir.exists():
            checkpoints = list(output_dir.glob("*.safetensors"))
            print(f"   - {char}: {len(checkpoints)} checkpoints")

    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for char in failed:
            print(f"   - {char}")

    print(f"\nTotal duration: {total_elapsed/3600:.1f} hours")
    print(f"Logs saved to: {LOG_DIR}")

    # Next steps
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Check training logs in:", LOG_DIR)
    print("2. Review checkpoints in: /mnt/c/ai_models/lora_sdxl/super-wings/")
    print("3. Evaluate LoRAs:")
    print("   python scripts/batch/evaluate_super_wings_loras.py")
    print()

    # Save training report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_characters": len(available_characters),
        "successful": successful,
        "failed": failed,
        "duration_hours": total_elapsed / 3600,
        "log_dir": str(LOG_DIR)
    }

    report_path = LOG_DIR / "training_summary.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Training report saved: {report_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training sequence interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
