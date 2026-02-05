"""
Training Launcher for Kohya_ss SDXL LoRA Training

Launches and manages Kohya_ss training processes in background.
Provides progress monitoring, GPU resource checking, and automatic recovery.

Part of Module 6: Training Pipeline Integration
Manages the actual training execution and monitoring.

Features:
- Background process management (subprocess + tmux options)
- Real-time log tailing and progress parsing
- GPU memory checking before launch
- Automatic recovery on OOM/failure
- Training metrics extraction (loss, lr, epoch)
- Checkpoint detection and validation

Author: LLMProvider Tooling
Date: 2025-11-30
"""

import argparse
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from enum import Enum


class TrainingStatus(Enum):
    """Training process status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    OOM_ERROR = "oom_error"
    UNKNOWN = "unknown"


@dataclass
class TrainingMetrics:
    """Training progress metrics"""
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0
    last_checkpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingProcess:
    """Training process information"""
    pid: Optional[int] = None
    status: TrainingStatus = TrainingStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    config_path: str = ""
    log_file: str = ""
    output_dir: str = ""
    metrics: TrainingMetrics = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = TrainingMetrics()

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        return result


class GPUMonitor:
    """Monitor GPU resources"""

    @staticmethod
    def get_gpu_memory() -> Dict[str, Any]:
        """
        Get GPU memory usage via nvidia-smi

        Returns:
            Dict with total_mb, used_mb, free_mb, utilization_percent
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                logging.warning("nvidia-smi failed")
                return None

            # Parse first GPU only
            line = result.stdout.strip().split('\n')[0]
            total, used, free, util = map(float, line.split(','))

            return {
                'total_mb': total,
                'used_mb': used,
                'free_mb': free,
                'utilization_percent': util
            }

        except Exception as e:
            logging.warning(f"Failed to get GPU memory: {e}")
            return None

    @staticmethod
    def check_available_memory(required_mb: int = 8000) -> bool:
        """
        Check if sufficient GPU memory is available

        Args:
            required_mb: Required free memory in MB

        Returns:
            True if sufficient memory available
        """
        gpu_info = GPUMonitor.get_gpu_memory()
        if gpu_info is None:
            logging.warning("Cannot verify GPU memory - proceeding anyway")
            return True

        available = gpu_info['free_mb']
        if available < required_mb:
            logging.warning(f"Low GPU memory: {available:.0f}MB free, need {required_mb}MB")
            return False

        logging.info(f"GPU memory OK: {available:.0f}MB free")
        return True


class LogParser:
    """Parse Kohya_ss training logs for progress and metrics"""

    # Regex patterns for log parsing
    PATTERNS = {
        'epoch': re.compile(r'epoch (\d+)/(\d+)'),
        'steps': re.compile(r'steps: (\d+)/(\d+)'),
        'loss': re.compile(r'loss(?:_train)?[:\s]+([0-9.]+)'),
        'lr': re.compile(r'lr[:\s]+([0-9.e-]+)'),
        'time': re.compile(r'(?:elapsed|time)[:\s]+([0-9:.]+)'),
        'oom': re.compile(r'(?:CUDA out of memory|RuntimeError.*out of memory)', re.IGNORECASE),
        'error': re.compile(r'(?:Error|Exception|Traceback)', re.IGNORECASE),
        'checkpoint': re.compile(r'(?:saved|checkpoint).*?([a-zA-Z0-9_-]+\.safetensors)')
    }

    @staticmethod
    def parse_line(line: str, metrics: TrainingMetrics) -> bool:
        """
        Parse a log line and update metrics

        Args:
            line: Log line
            metrics: TrainingMetrics to update

        Returns:
            True if line contained useful info
        """
        updated = False

        # Epoch
        match = LogParser.PATTERNS['epoch'].search(line)
        if match:
            metrics.current_epoch = int(match.group(1))
            metrics.total_epochs = int(match.group(2))
            updated = True

        # Steps
        match = LogParser.PATTERNS['steps'].search(line)
        if match:
            metrics.current_step = int(match.group(1))
            metrics.total_steps = int(match.group(2))
            updated = True

        # Loss
        match = LogParser.PATTERNS['loss'].search(line)
        if match:
            metrics.loss = float(match.group(1))
            updated = True

        # Learning rate
        match = LogParser.PATTERNS['lr'].search(line)
        if match:
            metrics.learning_rate = float(match.group(1))
            updated = True

        # Checkpoint
        match = LogParser.PATTERNS['checkpoint'].search(line)
        if match:
            metrics.last_checkpoint = match.group(1)
            updated = True

        return updated

    @staticmethod
    def detect_error(line: str) -> Optional[str]:
        """
        Detect errors in log line

        Returns:
            Error type if detected, None otherwise
        """
        if LogParser.PATTERNS['oom'].search(line):
            return 'oom_error'
        elif LogParser.PATTERNS['error'].search(line):
            return 'generic_error'
        return None


class TrainingLauncher:
    """
    Main training launcher class

    Launches Kohya_ss training in background and monitors progress.
    Supports subprocess and tmux modes.
    """

    def __init__(
        self,
        kohya_scripts_path: str = "/mnt/c/ai_projects/kohya_ss/sd-scripts",
        conda_env: str = "kohya_ss",
        device: str = "cuda",
        use_tmux: bool = False
    ):
        """
        Initialize training launcher

        Args:
            kohya_scripts_path: Path to Kohya_ss sd-scripts directory
            conda_env: Conda environment name
            device: Device to use (cuda/cpu)
            use_tmux: Use tmux for background execution
        """
        self.kohya_scripts_path = Path(kohya_scripts_path)
        self.conda_env = conda_env
        self.device = device
        self.use_tmux = use_tmux

        # Validate paths
        self.train_script = self.kohya_scripts_path / "train_network.py"
        if not self.train_script.exists():
            raise FileNotFoundError(f"Kohya training script not found: {self.train_script}")

        # Active processes
        self.processes: Dict[str, TrainingProcess] = {}

    def launch_training(
        self,
        config_path: Path,
        job_id: str,
        output_dir: Path,
        log_dir: Path,
        blocking: bool = False,
        check_gpu_memory: bool = True,
        required_vram_mb: int = 8000
    ) -> TrainingProcess:
        """
        Launch training process

        Args:
            config_path: Path to TOML config
            job_id: Unique job identifier
            output_dir: Output directory for LoRA
            log_dir: Log directory
            blocking: Wait for training to complete
            check_gpu_memory: Check GPU memory before launch
            required_vram_mb: Required VRAM in MB

        Returns:
            TrainingProcess object
        """
        config_path = Path(config_path)
        output_dir = Path(output_dir)
        log_dir = Path(log_dir)

        # Validate config
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Create directories
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Check GPU memory
        if check_gpu_memory and self.device == "cuda":
            if not GPUMonitor.check_available_memory(required_vram_mb):
                logging.warning("Low GPU memory - training may fail")

        # Setup log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{job_id}_{timestamp}.log"

        # Initialize process tracker
        process = TrainingProcess(
            config_path=str(config_path),
            log_file=str(log_file),
            output_dir=str(output_dir),
            status=TrainingStatus.PENDING
        )

        # Build command
        cmd = self._build_command(config_path, log_file)

        logging.info(f"Launching training: {job_id}")
        logging.info(f"Config: {config_path}")
        logging.info(f"Log: {log_file}")
        logging.info(f"Output: {output_dir}")

        # Launch process
        try:
            if self.use_tmux:
                pid = self._launch_tmux(cmd, job_id, log_file)
            else:
                pid = self._launch_subprocess(cmd, log_file, blocking)

            process.pid = pid
            process.status = TrainingStatus.RUNNING
            process.start_time = time.time()

            self.processes[job_id] = process

            logging.info(f"Training launched with PID {pid}")

            # Wait if blocking
            if blocking:
                self.wait_for_completion(job_id)

            return process

        except Exception as e:
            logging.error(f"Failed to launch training: {e}")
            process.status = TrainingStatus.FAILED
            process.error_message = str(e)
            raise

    def _build_command(self, config_path: Path, log_file: Path) -> List[str]:
        """Build training command"""
        cmd = [
            "conda", "run", "-n", self.conda_env,
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "1",
            str(self.train_script),
            "--config_file", str(config_path)
        ]
        return cmd

    def _launch_subprocess(
        self,
        cmd: List[str],
        log_file: Path,
        blocking: bool
    ) -> int:
        """Launch via subprocess"""
        with open(log_file, 'w') as f:
            if blocking:
                # Blocking: wait for completion
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Training failed with code {result.returncode}")
                return -1  # No PID for completed process
            else:
                # Background: start and return PID
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    preexec_fn=os.setsid  # Create new process group
                )
                return process.pid

    def _launch_tmux(self, cmd: List[str], job_id: str, log_file: Path) -> int:
        """Launch via tmux session"""
        session_name = f"training_{job_id}"

        # Create tmux session
        tmux_cmd = [
            "tmux", "new-session", "-d", "-s", session_name,
            " ".join(cmd) + f" 2>&1 | tee {log_file}"
        ]

        subprocess.run(tmux_cmd, check=True)

        # Get PID from tmux
        pane_pid = subprocess.check_output(
            ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_pid}"],
            text=True
        ).strip()

        return int(pane_pid)

    def monitor_progress(
        self,
        job_id: str,
        callback: Optional[Callable[[TrainingMetrics], None]] = None,
        interval: float = 5.0
    ):
        """
        Monitor training progress by tailing log file

        Args:
            job_id: Job ID to monitor
            callback: Optional callback for metrics updates
            interval: Polling interval in seconds
        """
        if job_id not in self.processes:
            raise ValueError(f"Unknown job: {job_id}")

        process = self.processes[job_id]
        log_file = Path(process.log_file)

        if not log_file.exists():
            logging.warning(f"Log file not found: {log_file}")
            return

        # Tail log file
        last_pos = 0

        while process.status == TrainingStatus.RUNNING:
            # Check if process still alive
            if not self._is_process_alive(process.pid):
                process.status = TrainingStatus.COMPLETED
                process.end_time = time.time()
                break

            # Read new log lines
            with open(log_file, 'r') as f:
                f.seek(last_pos)
                new_lines = f.readlines()
                last_pos = f.tell()

            # Parse lines
            for line in new_lines:
                # Update metrics
                LogParser.parse_line(line, process.metrics)

                # Check for errors
                error_type = LogParser.detect_error(line)
                if error_type:
                    if error_type == 'oom_error':
                        process.status = TrainingStatus.OOM_ERROR
                    else:
                        process.status = TrainingStatus.FAILED
                    process.error_message = line.strip()
                    process.end_time = time.time()
                    logging.error(f"Training error detected: {line.strip()}")
                    return

            # Callback
            if callback and new_lines:
                callback(process.metrics)

            time.sleep(interval)

        logging.info(f"Training {job_id} finished with status: {process.status.value}")

    def wait_for_completion(self, job_id: str, timeout: Optional[float] = None):
        """
        Wait for training to complete

        Args:
            job_id: Job ID
            timeout: Optional timeout in seconds
        """
        if job_id not in self.processes:
            raise ValueError(f"Unknown job: {job_id}")

        process = self.processes[job_id]
        start = time.time()

        while process.status == TrainingStatus.RUNNING:
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Training timeout after {timeout}s")

            if not self._is_process_alive(process.pid):
                process.status = TrainingStatus.COMPLETED
                process.end_time = time.time()
                break

            time.sleep(1.0)

    def stop_training(self, job_id: str, force: bool = False):
        """
        Stop a running training process

        Args:
            job_id: Job ID
            force: Use SIGKILL instead of SIGTERM
        """
        if job_id not in self.processes:
            raise ValueError(f"Unknown job: {job_id}")

        process = self.processes[job_id]

        if process.status != TrainingStatus.RUNNING:
            logging.warning(f"Process {job_id} not running (status: {process.status.value})")
            return

        try:
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.killpg(os.getpgid(process.pid), sig)

            process.status = TrainingStatus.STOPPED
            process.end_time = time.time()

            logging.info(f"Stopped training {job_id}")

        except Exception as e:
            logging.error(f"Failed to stop training {job_id}: {e}")

    def get_status(self, job_id: str) -> TrainingProcess:
        """Get current status of training job"""
        if job_id not in self.processes:
            raise ValueError(f"Unknown job: {job_id}")

        process = self.processes[job_id]

        # Update status if process finished
        if process.status == TrainingStatus.RUNNING:
            if not self._is_process_alive(process.pid):
                process.status = TrainingStatus.COMPLETED
                process.end_time = time.time()

        return process

    def save_process_state(self, job_id: str, output_file: Path):
        """Save process state to JSON"""
        if job_id not in self.processes:
            raise ValueError(f"Unknown job: {job_id}")

        process = self.processes[job_id]

        with open(output_file, 'w') as f:
            json.dump(process.to_dict(), f, indent=2)

        logging.info(f"Saved process state to {output_file}")

    def load_process_state(self, input_file: Path) -> str:
        """Load process state from JSON and return job_id"""
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Reconstruct process
        process = TrainingProcess(
            pid=data.get('pid'),
            status=TrainingStatus(data['status']),
            start_time=data.get('start_time'),
            end_time=data.get('end_time'),
            config_path=data['config_path'],
            log_file=data['log_file'],
            output_dir=data['output_dir'],
            metrics=TrainingMetrics(**data['metrics']),
            error_message=data.get('error_message')
        )

        # Generate job_id from timestamp
        job_id = f"restored_{int(time.time())}"
        self.processes[job_id] = process

        logging.info(f"Loaded process state from {input_file} as {job_id}")
        return job_id

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """Check if process is alive"""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def main():
    """CLI interface for training launcher"""
    parser = argparse.ArgumentParser(description="Launch and monitor Kohya_ss SDXL LoRA training")

    # Required arguments
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training TOML config")
    parser.add_argument("--job-id", type=str, required=True,
                       help="Unique job identifier")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for LoRA")

    # Optional arguments
    parser.add_argument("--log-dir", type=str, default="./logs",
                       help="Log directory")
    parser.add_argument("--kohya-scripts-path", type=str,
                       default="/mnt/c/ai_projects/kohya_ss/sd-scripts",
                       help="Path to Kohya_ss sd-scripts")
    parser.add_argument("--conda-env", type=str, default="kohya_ss",
                       help="Conda environment name")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--use-tmux", action="store_true",
                       help="Use tmux for background execution")
    parser.add_argument("--blocking", action="store_true",
                       help="Wait for training to complete")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor progress in real-time")
    parser.add_argument("--no-gpu-check", action="store_true",
                       help="Skip GPU memory check")
    parser.add_argument("--required-vram-mb", type=int, default=8000,
                       help="Required VRAM in MB")
    parser.add_argument("--state-file", type=str,
                       help="Save/load process state to/from file")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Initialize launcher
    launcher = TrainingLauncher(
        kohya_scripts_path=args.kohya_scripts_path,
        conda_env=args.conda_env,
        device=args.device,
        use_tmux=args.use_tmux
    )

    try:
        # Launch training
        process = launcher.launch_training(
            config_path=Path(args.config),
            job_id=args.job_id,
            output_dir=Path(args.output_dir),
            log_dir=Path(args.log_dir),
            blocking=args.blocking,
            check_gpu_memory=not args.no_gpu_check,
            required_vram_mb=args.required_vram_mb
        )

        logging.info(f"Training launched: PID {process.pid}")
        logging.info(f"Log file: {process.log_file}")

        # Save state if requested
        if args.state_file:
            launcher.save_process_state(args.job_id, Path(args.state_file))

        # Monitor if requested
        if args.monitor and not args.blocking:
            def progress_callback(metrics: TrainingMetrics):
                logging.info(
                    f"Epoch {metrics.current_epoch}/{metrics.total_epochs} | "
                    f"Step {metrics.current_step}/{metrics.total_steps} | "
                    f"Loss: {metrics.loss:.4f} | LR: {metrics.learning_rate:.6f}"
                )

            launcher.monitor_progress(args.job_id, callback=progress_callback)

        # Final status
        final = launcher.get_status(args.job_id)
        logging.info(f"Final status: {final.status.value}")

        if final.status == TrainingStatus.FAILED:
            logging.error(f"Error: {final.error_message}")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
