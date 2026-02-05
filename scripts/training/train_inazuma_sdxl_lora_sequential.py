#!/usr/bin/env python3
"""
Sequential SDXL LoRA training for all Inazuma Eleven characters.
Trains characters one by one with proper checkpointing.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class SDXLLoRATrainer:
    """Sequential trainer for SDXL LoRA models."""

    def __init__(
        self,
        kohya_dir: str = "/mnt/c/ai_projects/kohya_ss",
        conda_env: str = "kohya_ss",
        config_dir: str = "/mnt/data/training/lora/inazuma_eleven/configs",
    ):
        self.kohya_dir = Path(kohya_dir)
        self.conda_env = conda_env
        self.config_dir = Path(config_dir)
        self.sdxl_train_script = self.kohya_dir / "sd-scripts" / "sdxl_train_network.py"
        self.training_log = []

    def load_manifest(self) -> List[Dict]:
        """Load character manifest from configs."""
        manifest_path = self.config_dir / "manifest.json"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        return manifest

    def build_training_command(self, config_path: str) -> str:
        """Build the training command."""
        cmd = (
            f"cd {self.kohya_dir} && "
            f"conda run -n {self.conda_env} python {self.sdxl_train_script} "
            f"--config_file {config_path}"
        )
        return cmd

    def train_character(
        self,
        character_name: str,
        character_id: str,
        config_path: str,
        index: int,
        total: int,
    ) -> bool:
        """Train a single character."""

        print("\n" + "=" * 70)
        print(f"🚀 Training Character [{index}/{total}]: {character_name}")
        print("=" * 70)
        print(f"Config: {config_path}")
        print(f"Character ID: {character_id}")
        print(f"Start Time: {datetime.now().isoformat()}")
        print()

        cmd = self.build_training_command(config_path)

        try:
            # Run training
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=False,
                text=True,
                timeout=None  # No timeout for training
            )

            success = result.returncode == 0

            log_entry = {
                "character": character_name,
                "character_id": character_id,
                "config_path": config_path,
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "return_code": result.returncode,
            }

            self.training_log.append(log_entry)

            if success:
                print(f"\n✅ {character_name} training completed successfully!")
            else:
                print(f"\n❌ {character_name} training failed with code {result.returncode}")

            return success

        except subprocess.TimeoutExpired:
            print(f"\n⏱️  {character_name} training timeout")
            self.training_log.append({
                "character": character_name,
                "character_id": character_id,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": "Training timeout",
            })
            return False

        except Exception as e:
            print(f"\n❌ Error training {character_name}: {e}")
            self.training_log.append({
                "character": character_name,
                "character_id": character_id,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e),
            })
            return False

    def verify_checkpoints(self, character_id: str) -> bool:
        """Verify that checkpoints were created."""
        output_dir = Path(f"/mnt/data/training/lora/inazuma_eleven/{character_id}")

        if not output_dir.exists():
            print(f"  ⚠️  Output directory not found: {output_dir}")
            return False

        # Look for .safetensors files
        checkpoints = list(output_dir.glob("**/*.safetensors"))

        if not checkpoints:
            print(f"  ⚠️  No checkpoints found in {output_dir}")
            return False

        print(f"  ✓ Found {len(checkpoints)} checkpoint(s):")
        for ckpt in sorted(checkpoints):
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"    - {ckpt.name} ({size_mb:.1f} MB)")

        return True

    def train_all(self, characters: List[str] = None) -> Dict:
        """Train all characters sequentially."""

        manifest = self.load_manifest()

        if characters:
            # Filter manifest to specific characters
            manifest = [m for m in manifest if m["character_id"] in characters]

        print("=" * 70)
        print("🎯 SDXL LoRA Sequential Training Pipeline")
        print("=" * 70)
        print(f"Characters to train: {len(manifest)}")
        print()

        for idx, char_info in enumerate(manifest, 1):
            character_name = char_info["character"]
            character_id = char_info["character_id"]
            config_path = char_info["config_path"]

            # Train
            success = self.train_character(
                character_name=character_name,
                character_id=character_id,
                config_path=config_path,
                index=idx,
                total=len(manifest),
            )

            # Verify checkpoints
            if success:
                print(f"\n📊 Verifying {character_name} checkpoints...")
                verified = self.verify_checkpoints(character_id)

                if verified:
                    print(f"✅ {character_name} training and verification complete!")
                else:
                    print(f"⚠️  {character_name} checkpoint verification issues")
            else:
                print(f"⚠️  Skipping checkpoint verification due to training failure")

            print()

        # Final summary
        return self.generate_summary()

    def generate_summary(self) -> Dict:
        """Generate training summary."""

        successful = [log for log in self.training_log if log.get("success", False)]
        failed = [log for log in self.training_log if not log.get("success", True)]

        summary = {
            "total_trained": len(self.training_log),
            "successful": len(successful),
            "failed": len(failed),
            "timestamp": datetime.now().isoformat(),
            "training_log": self.training_log,
        }

        # Save summary
        summary_path = Path("/mnt/data/training/lora/inazuma_eleven/training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("📋 TRAINING SUMMARY")
        print("=" * 70)
        print(f"Total characters trained: {summary['total_trained']}")
        print(f"✅ Successful: {summary['successful']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"📄 Summary saved: {summary_path}")
        print("=" * 70)

        return summary


def main():
    """Main execution."""

    import argparse

    parser = argparse.ArgumentParser(description="Train SDXL LoRA models for Inazuma Eleven characters")
    parser.add_argument(
        "--characters",
        nargs="+",
        help="Specific characters to train (default: all)",
        default=None,
    )
    parser.add_argument(
        "--kohya-dir",
        default="/mnt/c/ai_projects/kohya_ss",
        help="Kohya_ss installation directory",
    )
    parser.add_argument(
        "--conda-env",
        default="kohya_ss",
        help="Conda environment name",
    )

    args = parser.parse_args()

    trainer = SDXLLoRATrainer(
        kohya_dir=args.kohya_dir,
        conda_env=args.conda_env,
    )

    summary = trainer.train_all(characters=args.characters)

    if summary["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
