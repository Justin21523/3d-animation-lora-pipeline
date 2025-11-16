#!/usr/bin/env python3
"""
Checkpoint-Level Evaluation Monitor

Watches for new checkpoint files during training and automatically evaluates them.
This provides immediate feedback on each checkpoint without waiting for full iteration to complete.

Usage:
    python checkpoint_evaluator.py --watch-dir /path/to/output --character luca_human
"""

import argparse
import time
import subprocess
from pathlib import Path
from typing import Set, Dict
import sys
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class CheckpointEvaluator:
    """Monitor and evaluate checkpoints as they are created"""

    def __init__(
        self,
        watch_dir: Path,
        character: str,
        base_model_path: str,
        eval_output_base: Path,
        check_interval: int = 30
    ):
        self.watch_dir = Path(watch_dir)
        self.character = character
        self.base_model_path = base_model_path
        self.eval_output_base = Path(eval_output_base)
        self.check_interval = check_interval
        self.evaluated_checkpoints: Set[str] = set()

        # Load already evaluated checkpoints
        self._load_evaluation_history()

        print("=" * 70)
        print("CHECKPOINT-LEVEL EVALUATION MONITOR")
        print("=" * 70)
        print(f"Watching: {self.watch_dir}")
        print(f"Character: {self.character}")
        print(f"Base Model: {self.base_model_path}")
        print(f"Eval Output: {self.eval_output_base}")
        print(f"Check Interval: {self.check_interval}s")
        print("=" * 70)
        print()

    def _load_evaluation_history(self):
        """Load history of already-evaluated checkpoints"""
        history_file = self.eval_output_base / "evaluation_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                self.evaluated_checkpoints = set(history.get('evaluated', []))
                print(f"Loaded {len(self.evaluated_checkpoints)} already-evaluated checkpoints")

    def _save_evaluation_history(self):
        """Save evaluation history"""
        history_file = self.eval_output_base / "evaluation_history.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(history_file, 'w') as f:
            json.dump({
                'evaluated': list(self.evaluated_checkpoints),
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)

    def find_new_checkpoints(self) -> list[Path]:
        """Find new checkpoint files that haven't been evaluated"""
        if not self.watch_dir.exists():
            return []

        all_checkpoints = list(self.watch_dir.glob("*.safetensors"))
        new_checkpoints = [
            ckpt for ckpt in all_checkpoints
            if ckpt.name not in self.evaluated_checkpoints
        ]

        # Sort by creation time (oldest first)
        new_checkpoints.sort(key=lambda p: p.stat().st_ctime)

        return new_checkpoints

    def evaluate_checkpoint(self, checkpoint_path: Path) -> bool:
        """Evaluate a single checkpoint"""
        print("\n" + "=" * 70)
        print(f"📊 EVALUATING CHECKPOINT: {checkpoint_path.name}")
        print("=" * 70)
        print(f"Size: {checkpoint_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"Created: {datetime.fromtimestamp(checkpoint_path.stat().st_ctime).strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Output directory for this checkpoint's evaluation
        eval_dir = self.eval_output_base / checkpoint_path.stem
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Use test_lora_checkpoints.py to generate test images
        eval_script = Path(__file__).parent.parent / 'evaluation' / 'test_lora_checkpoints.py'

        if not eval_script.exists():
            print(f"❌ Evaluation script not found: {eval_script}")
            return False

        # Create temporary directory with just this checkpoint
        temp_lora_dir = eval_dir / "temp_lora"
        temp_lora_dir.mkdir(exist_ok=True)

        # Symlink the checkpoint
        temp_checkpoint = temp_lora_dir / checkpoint_path.name
        if not temp_checkpoint.exists():
            temp_checkpoint.symlink_to(checkpoint_path)

        cmd = [
            'conda', 'run', '-n', 'ai_env',
            'python', str(eval_script),
            str(temp_lora_dir),
            '--base-model', self.base_model_path,
            '--output-dir', str(eval_dir),
            '--device', 'cuda',
            '--num-variations', '4',  # 4 variations per prompt
            '--steps', '25'
        ]

        print("Running evaluation...")
        print(" ".join(cmd[:5]) + " \\")
        print("  " + " \\\n  ".join(cmd[5:]))
        print()

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )

            print("✅ Evaluation completed successfully")
            print(f"📁 Results: {eval_dir}")

            # Mark as evaluated
            self.evaluated_checkpoints.add(checkpoint_path.name)
            self._save_evaluation_history()

            # Clean up temp directory
            if temp_checkpoint.exists():
                temp_checkpoint.unlink()
            if temp_lora_dir.exists():
                temp_lora_dir.rmdir()

            return True

        except subprocess.TimeoutExpired:
            print("❌ Evaluation timeout after 30 minutes")
            return False

        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation failed: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr[:500]}")
            return False

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

    def monitor(self):
        """Main monitoring loop"""
        print("🔍 Starting monitoring loop...")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                # Find new checkpoints
                new_checkpoints = self.find_new_checkpoints()

                if new_checkpoints:
                    print(f"\n🆕 Found {len(new_checkpoints)} new checkpoint(s)")

                    for checkpoint in new_checkpoints:
                        success = self.evaluate_checkpoint(checkpoint)

                        if success:
                            print(f"✅ {checkpoint.name} evaluated successfully")
                        else:
                            print(f"⚠️  {checkpoint.name} evaluation failed")

                        # Small delay between evaluations
                        time.sleep(5)

                # Wait before next check
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("🛑 Monitoring stopped by user")
            print("=" * 70)
            print(f"Total checkpoints evaluated: {len(self.evaluated_checkpoints)}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Monitor and evaluate LoRA checkpoints as they are created"
    )
    parser.add_argument(
        '--watch-dir',
        type=Path,
        required=True,
        help="Directory to watch for new checkpoint files"
    )
    parser.add_argument(
        '--character',
        type=str,
        required=True,
        help="Character name (e.g., luca_human)"
    )
    parser.add_argument(
        '--base-model',
        type=str,
        required=True,
        help="Path to base Stable Diffusion model"
    )
    parser.add_argument(
        '--eval-output',
        type=Path,
        required=True,
        help="Base directory for evaluation outputs"
    )
    parser.add_argument(
        '--check-interval',
        type=int,
        default=30,
        help="Interval between checks in seconds (default: 30)"
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = CheckpointEvaluator(
        watch_dir=args.watch_dir,
        character=args.character,
        base_model_path=args.base_model,
        eval_output_base=args.eval_output,
        check_interval=args.check_interval
    )

    # Start monitoring
    evaluator.monitor()


if __name__ == "__main__":
    main()
