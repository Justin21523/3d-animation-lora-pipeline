#!/usr/bin/env python3
"""
Real-time Checkpoint Evaluator for Hyperparameter Optimization

Monitors trial directories for new checkpoints and evaluates them immediately.
This allows observing quality changes during training without waiting for completion.

Usage:
    python realtime_checkpoint_evaluator.py --trial-dir /path/to/trial_0001 --device cuda
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Set, Dict, Any


class RealtimeCheckpointEvaluator:
    """
    Monitors a trial directory and evaluates new checkpoints as they appear
    """

    def __init__(
        self,
        trial_dir: Path,
        device: str = "cuda",
        num_samples: int = 8,
        check_interval: int = 30,
    ):
        """
        Args:
            trial_dir: Path to trial directory to monitor
            device: Device for evaluation (cuda/cpu)
            num_samples: Number of test images to generate per checkpoint
            check_interval: Seconds between directory scans
        """
        self.trial_dir = Path(trial_dir)
        self.device = device
        self.num_samples = num_samples
        self.check_interval = check_interval

        self.evaluated_checkpoints: Set[str] = set()
        self.evaluation_results: Dict[str, Any] = {}

        # Create evaluation directory
        self.eval_base_dir = self.trial_dir / "realtime_evaluations"
        self.eval_base_dir.mkdir(exist_ok=True)

        # Load previous evaluation state if exists
        self.state_file = self.eval_base_dir / "evaluation_state.json"
        self._load_state()

    def _load_state(self):
        """Load previous evaluation state to avoid re-evaluating"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.evaluated_checkpoints = set(state.get("evaluated_checkpoints", []))
                self.evaluation_results = state.get("evaluation_results", {})
            print(f"📂 Loaded previous state: {len(self.evaluated_checkpoints)} checkpoints already evaluated")

    def _save_state(self):
        """Save evaluation state"""
        state = {
            "evaluated_checkpoints": list(self.evaluated_checkpoints),
            "evaluation_results": self.evaluation_results,
            "last_update": datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def scan_for_new_checkpoints(self) -> list[Path]:
        """Scan trial directory for new .safetensors checkpoints"""
        all_checkpoints = sorted(self.trial_dir.glob("*.safetensors"))
        new_checkpoints = [
            cp for cp in all_checkpoints
            if cp.name not in self.evaluated_checkpoints
        ]
        return new_checkpoints

    def evaluate_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Evaluate a single checkpoint using the evaluation script

        Returns:
            Dictionary of evaluation metrics
        """
        checkpoint_name = checkpoint_path.name
        print(f"\n{'='*60}")
        print(f"📊 Evaluating: {checkpoint_name}")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        # Create checkpoint-specific evaluation directory
        eval_dir = self.eval_base_dir / checkpoint_path.stem
        eval_dir.mkdir(exist_ok=True)

        # Run evaluation script
        cmd = [
            "/home/b0979/.conda/envs/kohya_ss/bin/python",
            "/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/evaluation/evaluate_single_checkpoint.py",
            "--checkpoint", str(checkpoint_path),
            "--output-dir", str(eval_dir),
            "--num-samples", str(self.num_samples),
            "--device", self.device,
        ]

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            elapsed = time.time() - start_time

            # Load metrics
            metrics_file = eval_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                # Add metadata
                metrics["checkpoint_name"] = checkpoint_name
                metrics["evaluation_time"] = elapsed
                metrics["timestamp"] = datetime.now().isoformat()

                print(f"✅ Evaluation completed in {elapsed:.1f}s")
                print(f"📈 Key metrics:")
                if "metrics" in metrics:
                    m = metrics["metrics"]
                    print(f"   - Brightness: {m.get('mean_brightness', 0):.3f}")
                    print(f"   - Contrast: {m.get('mean_contrast', 0):.3f}")
                    print(f"   - Saturation: {m.get('mean_saturation', 0):.3f}")

                # Mark as evaluated
                self.evaluated_checkpoints.add(checkpoint_name)
                self.evaluation_results[checkpoint_name] = metrics
                self._save_state()

                return metrics
            else:
                print(f"⚠️  Warning: Metrics file not found at {metrics_file}")
                return {}

        except subprocess.TimeoutExpired:
            print(f"❌ Evaluation timeout after 5 minutes")
            return {}
        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation failed with return code {e.returncode}")
            print(f"   stderr: {e.stderr[:500]}")  # First 500 chars
            return {}
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return {}

    def generate_summary_report(self):
        """Generate a summary report of all evaluations"""
        if not self.evaluation_results:
            return

        report_file = self.eval_base_dir / "evaluation_summary.json"

        # Create comparative summary
        summary = {
            "trial_dir": str(self.trial_dir),
            "total_checkpoints_evaluated": len(self.evaluated_checkpoints),
            "last_update": datetime.now().isoformat(),
            "checkpoints": []
        }

        # Sort checkpoints by name (usually correlates with training progress)
        for checkpoint_name in sorted(self.evaluated_checkpoints):
            if checkpoint_name in self.evaluation_results:
                result = self.evaluation_results[checkpoint_name]
                summary["checkpoints"].append({
                    "name": checkpoint_name,
                    "timestamp": result.get("timestamp"),
                    "metrics": result.get("metrics", {})
                })

        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📝 Summary report saved to: {report_file}")

    def run(self):
        """Main monitoring loop"""
        print(f"\n{'='*60}")
        print(f"🎯 Real-time Checkpoint Evaluator Started")
        print(f"{'='*60}")
        print(f"📁 Monitoring: {self.trial_dir}")
        print(f"⏱️  Check interval: {self.check_interval} seconds")
        print(f"🖥️  Device: {self.device}")
        print(f"🎨 Samples per checkpoint: {self.num_samples}")
        print(f"{'='*60}\n")

        try:
            while True:
                # Scan for new checkpoints
                new_checkpoints = self.scan_for_new_checkpoints()

                if new_checkpoints:
                    print(f"\n🔍 Found {len(new_checkpoints)} new checkpoint(s)")

                    for checkpoint_path in new_checkpoints:
                        self.evaluate_checkpoint(checkpoint_path)

                    # Generate updated summary
                    self.generate_summary_report()
                else:
                    # Still monitoring
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"[{timestamp}] 👀 Monitoring... (evaluated: {len(self.evaluated_checkpoints)})")

                # Wait before next scan
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print(f"🛑 Monitoring stopped by user")
            print(f"{'='*60}")
            print(f"✅ Total checkpoints evaluated: {len(self.evaluated_checkpoints)}")
            self.generate_summary_report()
            print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time checkpoint evaluator for hyperparameter optimization"
    )
    parser.add_argument(
        "--trial-dir",
        type=str,
        required=True,
        help="Path to trial directory to monitor"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for evaluation (cuda/cpu)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of test images to generate per checkpoint"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=30,
        help="Seconds between directory scans"
    )

    args = parser.parse_args()

    evaluator = RealtimeCheckpointEvaluator(
        trial_dir=args.trial_dir,
        device=args.device,
        num_samples=args.num_samples,
        check_interval=args.check_interval
    )

    evaluator.run()


if __name__ == "__main__":
    main()
