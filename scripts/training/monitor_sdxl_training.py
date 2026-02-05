#!/usr/bin/env python3
"""
Monitor SDXL LoRA training progress and verify outputs.
"""

import json
import time
from pathlib import Path
from typing import Dict, List
import subprocess


class TrainingMonitor:
    """Monitor SDXL LoRA training progress."""

    def __init__(
        self,
        training_dir: str = "/mnt/data/training/lora/inazuma_eleven",
        log_file: str = "/mnt/data/training/lora/inazuma_eleven/training.log",
    ):
        self.training_dir = Path(training_dir)
        self.log_file = Path(log_file)

    def check_training_status(self) -> Dict:
        """Check current training status."""
        status = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "characters": {},
            "overall": {},
        }

        # Characters to monitor
        characters = [
            "endou_mamoru",
            "fudou_akio",
            "gouenji_shuuya",
            "inamori_asuto",
            "matsukaze_tenma",
            "nosaka_yuuma",
            "utsunomiya_toramaru",
        ]

        checkpoints_count = {}
        latest_checkpoint = {}

        for char_id in characters:
            char_dir = self.training_dir / char_id
            logs_dir = char_dir / "logs"

            # Check if training started
            if char_dir.exists():
                # Count checkpoints
                checkpoints = list(char_dir.glob("*.safetensors"))
                checkpoints_count[char_id] = len(checkpoints)

                if checkpoints:
                    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    latest_checkpoint[char_id] = {
                        "name": latest.name,
                        "size_mb": latest.stat().st_size / (1024 * 1024),
                        "mtime": time.ctime(latest.stat().st_mtime),
                    }

                    status["characters"][char_id] = {
                        "checkpoints": checkpoints_count[char_id],
                        "latest": latest_checkpoint[char_id],
                        "status": "✓ Training in progress or completed",
                    }
                else:
                    status["characters"][char_id] = {
                        "checkpoints": 0,
                        "status": "⏳ Training directory exists (waiting for first checkpoint)",
                    }
            else:
                status["characters"][char_id] = {
                    "checkpoints": 0,
                    "status": "⏸️  Not started yet",
                }

        # Overall stats
        total_checkpoints = sum(checkpoints_count.values())
        completed = sum(1 for count in checkpoints_count.values() if count > 0)

        status["overall"] = {
            "total_characters": len(characters),
            "with_checkpoints": completed,
            "total_checkpoints": total_checkpoints,
            "progress": f"{completed}/{len(characters)} characters have checkpoints",
        }

        return status

    def print_status(self, status: Dict):
        """Print formatted status."""
        print("\n" + "=" * 70)
        print("🚀 SDXL LoRA Training Monitor")
        print("=" * 70)
        print(f"Time: {status['timestamp']}")
        print()

        print("📊 Overall Progress:")
        for key, value in status["overall"].items():
            print(f"  {key}: {value}")

        print("\n📝 Per-Character Status:")
        for char_id, char_status in status["characters"].items():
            char_name = char_id.replace("_", " ").title()
            print(f"\n  {char_name} ({char_id}):")
            print(f"    Checkpoints: {char_status['checkpoints']}")
            print(f"    Status: {char_status['status']}")

            if "latest" in char_status:
                latest = char_status["latest"]
                print(f"    Latest: {latest['name']} ({latest['size_mb']:.1f} MB)")
                print(f"    Modified: {latest['mtime']}")

        print("\n" + "=" * 70)

    def wait_for_completion(self, check_interval: int = 300):
        """Wait for all training to complete."""
        print(f"\n⏳ Monitoring training (checking every {check_interval} seconds)...")
        print("Press Ctrl+C to stop monitoring\n")

        try:
            while True:
                status = self.check_training_status()
                self.print_status(status)

                # Check if all characters have completed
                if status["overall"]["with_checkpoints"] == status["overall"]["total_characters"]:
                    all_have_multiple = all(
                        status["characters"][c]["checkpoints"] >= 5
                        for c in status["characters"]
                    )

                    if all_have_multiple:
                        print("\n✅ All characters appear to have completed training!")
                        break

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user")

    def verify_all_outputs(self) -> Dict:
        """Verify all training outputs."""
        print("\n" + "=" * 70)
        print("🔍 Verifying Training Outputs")
        print("=" * 70)

        verification = {
            "total": 0,
            "with_checkpoints": 0,
            "details": {},
        }

        characters = [
            "endou_mamoru",
            "fudou_akio",
            "gouenji_shuuya",
            "inamori_asuto",
            "matsukaze_tenma",
            "nosaka_yuuma",
            "utsunomiya_toramaru",
        ]

        for char_id in characters:
            char_dir = self.training_dir / char_id
            char_name = char_id.replace("_", " ").title()

            verification["total"] += 1

            if not char_dir.exists():
                print(f"  ❌ {char_name}: Output directory not found")
                verification["details"][char_id] = {
                    "status": "missing",
                    "checkpoints": 0,
                }
                continue

            checkpoints = list(char_dir.glob("*.safetensors"))
            logs = list((char_dir / "logs").glob("**/*")) if (char_dir / "logs").exists() else []

            if checkpoints:
                verification["with_checkpoints"] += 1
                verification["details"][char_id] = {
                    "status": "✓ Complete",
                    "checkpoints": len(checkpoints),
                    "total_size_mb": sum(c.stat().st_size for c in checkpoints) / (1024 * 1024),
                    "has_logs": len(logs) > 0,
                }

                print(f"  ✅ {char_name}:")
                print(f"     Checkpoints: {len(checkpoints)}")
                print(f"     Total size: {verification['details'][char_id]['total_size_mb']:.1f} MB")
                print(f"     Logs: {'Yes' if logs else 'No'}")
            else:
                print(f"  ⏳ {char_name}: No checkpoints yet (training ongoing)")
                verification["details"][char_id] = {
                    "status": "in_progress",
                    "checkpoints": 0,
                }

        print("\n" + "=" * 70)
        print(f"Summary: {verification['with_checkpoints']}/{verification['total']} characters completed")
        print("=" * 70)

        return verification


def main():
    """Main execution."""

    import argparse

    parser = argparse.ArgumentParser(description="Monitor SDXL LoRA training progress")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for training completion",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify all outputs",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300)",
    )

    args = parser.parse_args()

    monitor = TrainingMonitor()

    if args.verify:
        monitor.verify_all_outputs()
    elif args.wait:
        monitor.wait_for_completion(check_interval=args.check_interval)
    else:
        # Single status check
        status = monitor.check_training_status()
        monitor.print_status(status)


if __name__ == "__main__":
    main()
