#!/usr/bin/env python3
"""
Training Manager

Unified training management system for start/stop/restart operations.

Replaces:
- auto_train_memory_safe.sh
- restart_training.sh
- safe_restart_training.sh
- stop_training.sh

Features:
- Start training with memory monitoring
- Stop training gracefully
- Restart from last checkpoint
- VRAM usage monitoring

Author: LLMProvider Tooling
Date: 2025-11-22
"""

import argparse
import json
import psutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingManager:
    """Manage training processes"""

    def __init__(self, config_file: Path, memory_limit_gb: float = 14.0):
        self.config_file = Path(config_file)
        self.memory_limit_gb = memory_limit_gb
        self.process: Optional[subprocess.Popen] = None

    def start(self, resume_from: Optional[str] = None):
        """Start training"""
        logger.info(f"Starting training: {self.config_file}")

        cmd = [
            "conda", "run", "-n", "kohya_ss",
            "accelerate", "launch",
            "--num_cpu_threads_per_process=4",
            "train_network.py",
            "--config_file", str(self.config_file)
        ]

        if resume_from:
            cmd.extend(["--resume", resume_from])

        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Training started (PID: {self.process.pid})")

    def stop(self, graceful: bool = True):
        """Stop training"""
        if not self.process:
            logger.warning("No training process running")
            return

        if graceful:
            logger.info("Stopping training gracefully...")
            self.process.terminate()
            self.process.wait(timeout=30)
        else:
            logger.info("Force stopping training...")
            self.process.kill()

        logger.info("Training stopped")

    def restart(self, checkpoint: str = "last"):
        """Restart training from checkpoint"""
        logger.info(f"Restarting from checkpoint: {checkpoint}")
        self.stop()
        time.sleep(2)
        self.start(resume_from=checkpoint)

    def monitor_memory(self):
        """Monitor VRAM usage"""
        try:
            import GPUtil
            while self.process and self.process.poll() is None:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    used_gb = gpu.memoryUsed / 1024
                    if used_gb > self.memory_limit_gb:
                        logger.warning(f"VRAM usage high: {used_gb:.1f}GB > {self.memory_limit_gb}GB")
                time.sleep(10)
        except ImportError:
            logger.warning("GPUtil not installed, skipping memory monitoring")


def main():
    parser = argparse.ArgumentParser(description="Training Manager")
    parser.add_argument('action', choices=['start', 'stop', 'restart'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', default='last')
    parser.add_argument('--memory-limit', type=float, default=14.0)
    parser.add_argument('--graceful', action='store_true', default=True)

    args = parser.parse_args()

    manager = TrainingManager(args.config, args.memory_limit)

    if args.action == 'start':
        manager.start()
    elif args.action == 'stop':
        manager.stop(args.graceful)
    elif args.action == 'restart':
        manager.restart(args.checkpoint)

    return 0


if __name__ == '__main__':
    sys.exit(main())
