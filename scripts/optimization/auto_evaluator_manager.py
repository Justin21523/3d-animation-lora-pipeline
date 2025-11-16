#!/usr/bin/env python3
"""
Automatic Evaluator Manager for All Trials

Monitors the optimization directory and automatically launches realtime evaluators
for each new trial as they appear. Ensures every trial gets checkpoint evaluation.

Usage:
    python auto_evaluator_manager.py --optimization-dir /path/to/optimization_overnight
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Set


class AutoEvaluatorManager:
    """
    Manages realtime evaluators for all trials in an optimization run
    """

    def __init__(
        self,
        optimization_dir: Path,
        device: str = "cuda",
        num_samples: int = 8,
        check_interval: int = 30,
        evaluator_check_interval: int = 30,
    ):
        """
        Args:
            optimization_dir: Base optimization directory containing trial_* folders
            device: Device for evaluation (cuda/cpu)
            num_samples: Number of samples per checkpoint evaluation
            check_interval: Seconds between directory scans for new trials
            evaluator_check_interval: Seconds between checkpoint scans within each evaluator
        """
        self.optimization_dir = Path(optimization_dir)
        self.device = device
        self.num_samples = num_samples
        self.check_interval = check_interval
        self.evaluator_check_interval = evaluator_check_interval

        self.active_trials: Dict[str, Dict] = {}  # trial_name -> {pid, start_time}
        self.state_file = self.optimization_dir / "evaluator_manager_state.json"

        # Path to the realtime evaluator script
        self.evaluator_script = Path(__file__).parent / "realtime_checkpoint_evaluator.py"

        # Load previous state
        self._load_state()

    def _load_state(self):
        """Load previous state to track active evaluators"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.active_trials = state.get("active_trials", {})
            print(f"📂 Loaded state: {len(self.active_trials)} previously tracked trials")

    def _save_state(self):
        """Save current state"""
        state = {
            "active_trials": self.active_trials,
            "last_update": datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def scan_for_trials(self) -> list[Path]:
        """Scan optimization directory for trial_* folders"""
        trial_dirs = sorted(self.optimization_dir.glob("trial_*"))
        return [d for d in trial_dirs if d.is_dir()]

    def is_evaluator_running(self, pid: int) -> bool:
        """Check if an evaluator process is still running"""
        try:
            import os
            import signal
            # Send signal 0 to check if process exists
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def launch_evaluator(self, trial_dir: Path) -> int:
        """
        Launch a realtime evaluator for a trial

        Returns:
            PID of the launched evaluator process
        """
        trial_name = trial_dir.name
        log_file = trial_dir / "realtime_evaluation.log"

        print(f"\n{'='*60}")
        print(f"🚀 Launching evaluator for {trial_name}")
        print(f"{'='*60}")
        print(f"📁 Trial directory: {trial_dir}")
        print(f"📝 Log file: {log_file}")
        print(f"🎨 Samples per checkpoint: {self.num_samples}")
        print(f"⏱️  Check interval: {self.evaluator_check_interval}s")

        cmd = [
            "python3",
            str(self.evaluator_script),
            "--trial-dir", str(trial_dir),
            "--device", self.device,
            "--num-samples", str(self.num_samples),
            "--check-interval", str(self.evaluator_check_interval)
        ]

        # Launch in background with nohup-like behavior
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True  # Detach from parent
            )

        pid = process.pid
        print(f"✅ Evaluator launched with PID: {pid}")
        print(f"{'='*60}\n")

        return pid

    def cleanup_dead_evaluators(self):
        """Remove entries for evaluators that are no longer running"""
        dead_trials = []

        for trial_name, info in self.active_trials.items():
            pid = info.get("pid")
            if pid and not self.is_evaluator_running(pid):
                dead_trials.append(trial_name)

        for trial_name in dead_trials:
            print(f"🧹 Cleaning up dead evaluator for {trial_name} (PID: {self.active_trials[trial_name]['pid']})")
            del self.active_trials[trial_name]

        if dead_trials:
            self._save_state()

    def run(self):
        """Main management loop"""
        print(f"\n{'='*60}")
        print(f"🎯 Auto Evaluator Manager Started")
        print(f"{'='*60}")
        print(f"📁 Optimization directory: {self.optimization_dir}")
        print(f"⏱️  Check interval: {self.check_interval} seconds")
        print(f"🖥️  Device: {self.device}")
        print(f"🎨 Samples per checkpoint: {self.num_samples}")
        print(f"{'='*60}\n")

        try:
            while True:
                # Clean up dead evaluators
                self.cleanup_dead_evaluators()

                # Scan for trial directories
                trial_dirs = self.scan_for_trials()

                if not trial_dirs:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"[{timestamp}] ⏳ Waiting for trials to start...")
                else:
                    # Check each trial
                    for trial_dir in trial_dirs:
                        trial_name = trial_dir.name

                        # Check if evaluator already running
                        if trial_name in self.active_trials:
                            pid = self.active_trials[trial_name]["pid"]
                            if self.is_evaluator_running(pid):
                                continue  # Already running, skip
                            else:
                                # Was running but died, remove and relaunch
                                print(f"⚠️  Evaluator for {trial_name} (PID {pid}) died, relaunching...")
                                del self.active_trials[trial_name]

                        # Launch new evaluator
                        pid = self.launch_evaluator(trial_dir)
                        self.active_trials[trial_name] = {
                            "pid": pid,
                            "start_time": datetime.now().isoformat(),
                            "trial_dir": str(trial_dir)
                        }
                        self._save_state()

                    # Status update
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    active_count = len([t for t, info in self.active_trials.items()
                                       if self.is_evaluator_running(info["pid"])])
                    print(f"[{timestamp}] 👀 Monitoring {len(trial_dirs)} trials, "
                          f"{active_count} evaluators active")

                # Wait before next scan
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print(f"🛑 Manager stopped by user")
            print(f"{'='*60}")
            print(f"Active evaluators:")
            for trial_name, info in self.active_trials.items():
                pid = info["pid"]
                status = "✅ Running" if self.is_evaluator_running(pid) else "❌ Stopped"
                print(f"  {trial_name}: PID {pid} - {status}")
            print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Automatic evaluator manager for all optimization trials"
    )
    parser.add_argument(
        "--optimization-dir",
        type=str,
        required=True,
        help="Path to optimization directory containing trial_* folders"
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
        help="Number of test images per checkpoint"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=30,
        help="Seconds between directory scans for new trials"
    )
    parser.add_argument(
        "--evaluator-check-interval",
        type=int,
        default=30,
        help="Seconds between checkpoint scans within each evaluator"
    )

    args = parser.parse_args()

    manager = AutoEvaluatorManager(
        optimization_dir=args.optimization_dir,
        device=args.device,
        num_samples=args.num_samples,
        check_interval=args.check_interval,
        evaluator_check_interval=args.evaluator_check_interval
    )

    manager.run()


if __name__ == "__main__":
    main()
