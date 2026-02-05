#!/usr/bin/env python3
"""
Unified Training Orchestrator for LoRA Training.

This repository contains two training entry styles, and both are supported here:

1) Dataset-driven (AI_WAREHOUSE-style):
   - Generate a Kohya TOML from a prepared dataset folder and launch training.
   - Intended for quick single-character runs.

2) Config-driven (legacy queue runner):
   - Run Kohya with an existing TOML config file (single job or batch JSON).
   - Intended for long-running sequential queues and tmux-managed runs.

CLI compatibility:
  - Legacy mode CLI:
      python scripts/training/unified_training_orchestrator.py sequential \
        --config-file configs/training/xxx.toml \
        --character NAME \
        --output-dir outputs/...

  - Dataset CLI:
      python scripts/training/unified_training_orchestrator.py /path/to/dataset \
        --character-name NAME \
        --base-model sdxl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Dataset-driven training (v2)
# ============================================================================


@dataclass
class TrainingConfig:
    """Configuration for a single training job."""

    character_name: str
    dataset_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    config_path: Optional[Path] = None

    # Model
    base_model: str = "sdxl"  # "sd15" | "sdxl"
    network_dim: int = 32
    network_alpha: int = 16

    # Training
    epochs: int = 10
    batch_size: int = 1
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100

    # Checkpoints
    save_every_n_epochs: int = 2
    max_checkpoints: int = 5

    # Resources
    mixed_precision: str = "bf16"
    seed: int = 42

    # Auto-eval
    enable_auto_eval: bool = True
    eval_every_n_epochs: int = 2
    eval_prompts: Optional[List[str]] = None


@dataclass
class TrainingResult:
    """Result of a training job."""

    character_name: str
    status: str  # "completed", "failed", "stopped", "running", "timeout"
    output_dir: str
    checkpoints: List[str]
    best_checkpoint: Optional[str]
    final_loss: Optional[float]
    training_time: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BatchTrainingConfig:
    """Configuration for batch training multiple characters."""

    characters: List[TrainingConfig]
    output_root: Path
    sequential: bool = True  # Run one at a time vs parallel
    stop_on_failure: bool = False
    enable_auto_eval: bool = True
    use_tmux: bool = True
    tmux_session_prefix: str = "lora_train"


class UnifiedTrainingOrchestrator:
    """
    Dataset-driven training orchestration with optional auto-evaluation.

    Notes:
    - If `TrainingConfig.config_path` is provided, an existing TOML is used.
    - If `config_path` is not provided, `dataset_dir` is required to generate TOML.
    """

    # Common Kohya locations (varies by install/layout)
    KOHYA_PATH = Path("/mnt/c/ai_tools/kohya_ss")
    MODEL_ROOT = Path("/mnt/c/ai_models")

    def __init__(
        self,
        kohya_env: str = "kohya_ss",
        auto_evaluator=None,  # Optional AutoCheckpointEvaluator
    ):
        self.kohya_env = kohya_env
        self.auto_evaluator = auto_evaluator
        self._active_processes: Dict[str, subprocess.Popen] = {}

    def train_character_lora(
        self,
        config: TrainingConfig,
        blocking: bool = True,
        use_tmux: bool = False,
    ) -> TrainingResult:
        start_time = time.time()
        logger.info("Starting training for %s", config.character_name)

        output_dir = Path(config.output_dir) if config.output_dir else self._default_output_dir(config)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate config if not provided
        if config.config_path is None or not Path(config.config_path).exists():
            if not config.dataset_dir:
                return TrainingResult(
                    character_name=config.character_name,
                    status="failed",
                    output_dir=str(output_dir),
                    checkpoints=[],
                    best_checkpoint=None,
                    final_loss=None,
                    training_time=time.time() - start_time,
                    error_message="dataset_dir is required when config_path is not provided",
                )
            config.config_path = self._generate_config(config, output_dir)

        cmd = self._build_training_command(Path(config.config_path))

        if use_tmux:
            result = self._run_in_tmux(config.character_name, cmd, blocking)
        else:
            result = self._run_training(config.character_name, cmd, blocking)

        training_time = time.time() - start_time
        checkpoints = self._collect_checkpoints(output_dir)

        best_checkpoint = None
        if config.enable_auto_eval and checkpoints and self.auto_evaluator:
            try:
                eval_result = self.auto_evaluator.evaluate_checkpoints(
                    checkpoints,
                    config.eval_prompts or self._default_prompts(config.character_name),
                )
                best_checkpoint = getattr(eval_result, "best_checkpoint", None)
            except Exception as e:
                logger.warning("Auto-evaluation failed: %s", e)

        return TrainingResult(
            character_name=config.character_name,
            status=result["status"],
            output_dir=str(output_dir),
            checkpoints=checkpoints,
            best_checkpoint=best_checkpoint,
            final_loss=result.get("final_loss"),
            training_time=training_time,
            error_message=result.get("error"),
        )

    def batch_train(self, batch_config: BatchTrainingConfig) -> List[TrainingResult]:
        results: List[TrainingResult] = []
        logger.info("Starting batch training for %d characters", len(batch_config.characters))

        for i, char_config in enumerate(batch_config.characters):
            logger.info("Training %d/%d: %s", i + 1, len(batch_config.characters), char_config.character_name)

            if char_config.output_dir is None:
                char_config.output_dir = batch_config.output_root / char_config.character_name

            result = self.train_character_lora(
                char_config,
                blocking=batch_config.sequential,
                use_tmux=batch_config.use_tmux,
            )
            results.append(result)

            if result.status == "failed" and batch_config.stop_on_failure:
                logger.error("Training failed for %s; stopping batch", char_config.character_name)
                break

        self._save_batch_report(results, batch_config.output_root)
        return results

    def monitor_training(
        self,
        session_name: str,
        log_file: Optional[Path] = None,
    ) -> Dict:
        status: Dict[str, object] = {
            "running": False,
            "current_epoch": 0,
            "total_epochs": 0,
            "current_loss": None,
            "gpu_memory": None,
        }

        try:
            result = subprocess.run(["tmux", "has-session", "-t", session_name], capture_output=True)
            status["running"] = result.returncode == 0
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                used, total = result.stdout.strip().split(",")
                status["gpu_memory"] = f"{used.strip()}/{total.strip()} MB"
        except Exception:
            pass

        if log_file and Path(log_file).exists():
            status.update(self._parse_training_log(Path(log_file)))

        return status

    def stop_training(self, session_name: str, graceful: bool = True) -> bool:
        try:
            if graceful:
                subprocess.run(["tmux", "send-keys", "-t", session_name, "C-c"], check=False)
                time.sleep(5)
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
            logger.info("Stopped training session: %s", session_name)
            return True
        except Exception as e:
            logger.warning("Failed to stop session %s: %s", session_name, e)
            return False

    def _default_output_dir(self, config: TrainingConfig) -> Path:
        base = config.base_model
        return self.MODEL_ROOT / f"lora_{base}" / config.character_name

    def _generate_config(self, config: TrainingConfig, output_dir: Path) -> Path:
        if config.base_model == "sdxl":
            model_path = self.MODEL_ROOT / "stable-diffusion/sd_xl_base_1.0.safetensors"
            resolution = 1024
        else:
            model_path = self.MODEL_ROOT / "stable-diffusion/v1-5-pruned-emaonly.safetensors"
            resolution = 512

        trigger_word = config.character_name.lower().replace(" ", "_")
        output_name = f"{trigger_word}_lora_{config.base_model}"

        dataset_dir = Path(config.dataset_dir)  # validated by caller
        train_data_dir = dataset_dir / "images"
        logging_dir = output_dir / "logs"

        config_content = f"""# Auto-generated Kohya LoRA Training Config
# Character: {config.character_name}
# Generated: {datetime.now().isoformat()}

[model]
pretrained_model_name_or_path = "{model_path}"
v2 = false
v_parameterization = false

[train]
output_dir = "{output_dir}"
output_name = "{output_name}"
save_model_as = "safetensors"
save_precision = "fp16"
save_every_n_epochs = {config.save_every_n_epochs}
max_train_epochs = {config.epochs}
train_batch_size = {config.batch_size}
gradient_accumulation_steps = {config.gradient_accumulation}
learning_rate = {config.learning_rate}
lr_scheduler = "{config.lr_scheduler}"
lr_warmup_steps = {config.warmup_steps}
mixed_precision = "{config.mixed_precision}"
seed = {config.seed}

[network]
network_module = "networks.lora"
network_dim = {config.network_dim}
network_alpha = {config.network_alpha}

[dataset]
train_data_dir = "{train_data_dir}"
resolution = {resolution}
enable_bucket = true
min_bucket_reso = {resolution // 2}
max_bucket_reso = {int(resolution * 1.5)}
bucket_reso_steps = 64
caption_extension = ".txt"
shuffle_caption = true
keep_tokens = 1

[optimizer]
optimizer_type = "AdamW8bit"
optimizer_args = []

[logging]
logging_dir = "{logging_dir}"
log_prefix = "{trigger_word}"
"""

        toml_path = output_dir / f"{config.character_name}_train.toml"
        toml_path.parent.mkdir(parents=True, exist_ok=True)
        toml_path.write_text(config_content, encoding="utf-8")
        return toml_path

    def _resolve_train_network(self) -> Path:
        candidates = [
            self.KOHYA_PATH / "train_network.py",
            self.KOHYA_PATH / "sd-scripts" / "train_network.py",
            self.KOHYA_PATH / "sd_scripts" / "train_network.py",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Could not find train_network.py under: {self.KOHYA_PATH}")

    def _build_training_command(self, config_path: Path) -> List[str]:
        train_script = self._resolve_train_network()
        return [
            "conda",
            "run",
            "-n",
            self.kohya_env,
            "accelerate",
            "launch",
            "--num_cpu_threads_per_process",
            "2",
            str(train_script),
            "--config_file",
            str(config_path),
        ]

    def _run_training(self, name: str, cmd: List[str], blocking: bool) -> Dict:
        try:
            if blocking:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600 * 24,  # 24 hour timeout
                )
                return {
                    "status": "completed" if result.returncode == 0 else "failed",
                    "error": result.stderr if result.returncode != 0 else None,
                }

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._active_processes[name] = process
            return {"status": "running"}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Training exceeded 24 hour limit"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _run_in_tmux(self, name: str, cmd: List[str], blocking: bool) -> Dict:
        session_name = f"lora_train_{name}"
        cmd_str = " ".join(cmd)

        try:
            subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)
            subprocess.run(["tmux", "new-session", "-d", "-s", session_name, cmd_str], check=True)
            logger.info("Started TMUX session: %s", session_name)

            if not blocking:
                return {"status": "running", "session": session_name}

            while True:
                result = subprocess.run(["tmux", "has-session", "-t", session_name], capture_output=True)
                if result.returncode != 0:
                    break
                time.sleep(60)

            return {"status": "completed"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _collect_checkpoints(self, output_dir: Path) -> List[str]:
        checkpoints: List[Path] = []
        for pattern in ("*.safetensors", "*.pt", "*.ckpt"):
            checkpoints.extend(output_dir.glob(pattern))
        return sorted(str(p) for p in checkpoints)

    def _parse_training_log(self, log_file: Path) -> Dict:
        info: Dict[str, object] = {}
        try:
            content = log_file.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")
            for line in reversed(lines[-100:]):
                if "epoch" in line.lower():
                    match = re.search(r"epoch\\s*(\\d+)/(\\d+)", line, re.IGNORECASE)
                    if match:
                        info["current_epoch"] = int(match.group(1))
                        info["total_epochs"] = int(match.group(2))
                if "loss" in line.lower():
                    match = re.search(r"loss[:\\s]+([0-9.]+)", line, re.IGNORECASE)
                    if match:
                        info["current_loss"] = float(match.group(1))
        except Exception:
            pass
        return info

    def _default_prompts(self, character_name: str) -> List[str]:
        trigger = character_name.lower().replace(" ", "_")
        return [
            f"{trigger}, portrait, front view, neutral expression",
            f"{trigger}, full body, standing pose",
            f"{trigger}, three-quarter view, smiling",
            f"{trigger}, close-up face, detailed",
            f"{trigger}, action pose, dynamic",
        ]

    def _save_batch_report(self, results: List[TrainingResult], output_dir: Path) -> None:
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_characters": len(results),
            "completed": sum(1 for r in results if r.status == "completed"),
            "failed": sum(1 for r in results if r.status == "failed"),
            "results": [r.to_dict() for r in results],
        }

        report_path = output_dir / "batch_training_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Saved batch report: %s", report_path)


def quick_train(
    character_name: str,
    dataset_dir: Union[str, Path],
    base_model: str = "sdxl",
    epochs: int = 10,
) -> TrainingResult:
    dataset_dir = Path(dataset_dir)
    output_dir = Path("/mnt/c/ai_models") / f"lora_{base_model}" / character_name

    config = TrainingConfig(
        character_name=character_name,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        base_model=base_model,
        epochs=epochs,
    )

    orchestrator = UnifiedTrainingOrchestrator()
    return orchestrator.train_character_lora(config, use_tmux=True)


# ============================================================================
# Config-driven queue runner (legacy)
# ============================================================================


@dataclass
class TrainingJob:
    """Single training job definition (config-file driven)."""

    job_id: str
    config_file: Path
    character_name: str
    output_dir: Path
    priority: int = 0

    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    checkpoint_path: Optional[Path] = None

    def duration_seconds(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "config_file": str(self.config_file),
            "character_name": self.character_name,
            "output_dir": str(self.output_dir),
            "priority": self.priority,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds(),
            "error_message": self.error_message,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
        }


class TrainingOrchestrator:
    """
    Legacy training orchestration system (config-file driven).

    Supports:
    - sequential: Train characters one by one
    - iterative: Placeholder for train → evaluate → improve loops
    """

    def __init__(
        self,
        mode: str = "sequential",
        kohya_scripts_dir: Optional[Path] = None,
        conda_env: str = "kohya_ss",
        use_tmux: bool = False,
        tmux_session: str = "lora_training",
    ):
        self.mode = mode
        self.kohya_scripts_dir = kohya_scripts_dir or self._find_kohya_scripts()
        self.conda_env = conda_env
        self.use_tmux = use_tmux
        self.tmux_session = tmux_session

        self.jobs: List[TrainingJob] = []
        self.current_job: Optional[TrainingJob] = None

        logger.info("TrainingOrchestrator initialized (mode=%s, kohya=%s)", mode, self.kohya_scripts_dir)

    def _find_kohya_scripts(self) -> Path:
        candidates = [
            Path("/mnt/c/ai_tools/kohya_ss/sd-scripts"),
            Path("/mnt/c/ai_tools/kohya_ss/sd_scripts"),
            Path("/mnt/c/AI_LLM_projects/kohya_ss/sd-scripts"),
            Path.home() / "kohya_ss/sd-scripts",
            Path.cwd().parent / "kohya_ss/sd-scripts",
        ]
        for path in candidates:
            if path.exists() and (path / "train_network.py").exists():
                return path
        raise FileNotFoundError("Could not find kohya_ss/sd-scripts. Please specify with --kohya-scripts-dir")

    def add_job(self, job: TrainingJob) -> None:
        self.jobs.append(job)
        logger.info("Added job: %s (%s)", job.job_id, job.character_name)

    def add_jobs_from_config(self, batch_config: Path) -> None:
        with open(batch_config, encoding="utf-8") as f:
            config = json.load(f)

        for i, job_config in enumerate(config.get("jobs", [])):
            job = TrainingJob(
                job_id=f"job_{i:03d}",
                config_file=Path(job_config["config_file"]),
                character_name=job_config["character_name"],
                output_dir=Path(job_config["output_dir"]),
                priority=job_config.get("priority", 0),
            )
            self.add_job(job)

        logger.info("Loaded %d jobs from %s", len(self.jobs), batch_config)

    def run(self) -> Dict:
        if not self.jobs:
            logger.warning("No jobs to execute")
            return {"status": "no_jobs", "jobs": []}

        self.jobs.sort(key=lambda j: j.priority, reverse=True)
        logger.info("Starting execution of %d jobs in %s mode", len(self.jobs), self.mode)

        if self.mode == "sequential":
            return self._run_sequential()
        if self.mode == "iterative":
            return self._run_iterative()
        raise ValueError(f"Unknown mode: {self.mode}")

    def _run_sequential(self) -> Dict:
        results: List[Dict] = []

        for i, job in enumerate(self.jobs, 1):
            logger.info("=" * 70)
            logger.info("Job %d/%d: %s", i, len(self.jobs), job.character_name)
            logger.info("=" * 70)

            self.current_job = job
            self._execute_job(job)
            results.append(job.to_dict())

        return {
            "mode": "sequential",
            "total_jobs": len(self.jobs),
            "completed": sum(1 for j in self.jobs if j.status == "completed"),
            "failed": sum(1 for j in self.jobs if j.status == "failed"),
            "jobs": results,
        }

    def _run_iterative(self) -> Dict:
        if len(self.jobs) > 1:
            logger.warning("Iterative mode works best with a single character; using the first job only.")

        job = self.jobs[0]
        iterations: List[Dict] = []

        for iteration in range(1, 6):  # Max 5 iterations
            logger.info("=" * 70)
            logger.info("Iteration %d: %s", iteration, job.character_name)
            logger.info("=" * 70)

            self.current_job = job
            success = self._execute_job(job)

            iteration_result = job.to_dict()
            iteration_result["iteration"] = iteration
            iterations.append(iteration_result)

            if not success:
                logger.error("Iteration %d failed", iteration)
                break

            # Placeholder for evaluation-driven parameter adjustment
            break

        return {
            "mode": "iterative",
            "character": job.character_name,
            "iterations": iterations,
            "total_iterations": len(iterations),
        }

    def _execute_job(self, job: TrainingJob) -> bool:
        job.status = "running"
        job.start_time = datetime.now()

        try:
            if not job.config_file.exists():
                raise FileNotFoundError(f"Config file not found: {job.config_file}")

            job.output_dir.mkdir(parents=True, exist_ok=True)
            train_script = self.kohya_scripts_dir / "train_network.py"

            cmd = [
                "conda",
                "run",
                "-n",
                self.conda_env,
                "accelerate",
                "launch",
                "--num_cpu_threads_per_process=4",
                str(train_script),
                "--config_file",
                str(job.config_file),
            ]

            if self.use_tmux:
                success = self._run_in_tmux(cmd)
            else:
                success = self._run_direct(cmd, job)

            job.status = "completed" if success else "failed"
            job.end_time = datetime.now()
            return success

        except Exception as e:
            job.status = "failed"
            job.end_time = datetime.now()
            job.error_message = str(e)
            logger.error("Job failed: %s", e, exc_info=True)
            return False

    def _run_direct(self, cmd: List[str], job: TrainingJob) -> bool:
        log_file = job.output_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=self.kohya_scripts_dir,
            )
            returncode = process.wait()

        logger.info("Training log: %s", log_file)
        return returncode == 0

    def _run_in_tmux(self, cmd: List[str]) -> bool:
        subprocess.run(["tmux", "new-session", "-d", "-s", self.tmux_session], check=False)
        subprocess.run(["tmux", "send-keys", "-t", self.tmux_session, " ".join(cmd), "C-m"], check=False)
        logger.info("Training started in tmux session: %s (attach: tmux attach -t %s)", self.tmux_session, self.tmux_session)
        return True

    def save_results(self, results: Dict, output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Results saved to: %s", output_file)


# ============================================================================
# CLI
# ============================================================================


def _run_legacy_mode_cli(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Unified Training Orchestrator (legacy mode CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("mode", choices=["sequential", "iterative"], help="Training mode")

    job_group = parser.add_mutually_exclusive_group(required=True)
    job_group.add_argument("--config-file", type=Path, help="Single training config file (TOML)")
    job_group.add_argument("--batch-config", type=Path, help="Batch configuration file (JSON)")

    parser.add_argument("--character", type=str, help="Character name (required with --config-file)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (required with --config-file)")

    parser.add_argument("--kohya-scripts-dir", type=Path, help="Path to kohya_ss/sd-scripts directory")
    parser.add_argument("--conda-env", type=str, default="kohya_ss", help="Conda environment name")

    parser.add_argument("--use-tmux", action="store_true", help="Run training in tmux session")
    parser.add_argument("--tmux-session", type=str, default="lora_training", help="Tmux session name")

    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("training_results.json"),
        help="Output file for results (default: training_results.json)",
    )

    args = parser.parse_args(argv)

    if args.config_file and (not args.character or not args.output_dir):
        parser.error("--config-file requires --character and --output-dir")

    orchestrator = TrainingOrchestrator(
        mode=args.mode,
        kohya_scripts_dir=args.kohya_scripts_dir,
        conda_env=args.conda_env,
        use_tmux=args.use_tmux,
        tmux_session=args.tmux_session,
    )

    if args.batch_config:
        orchestrator.add_jobs_from_config(args.batch_config)
    else:
        orchestrator.add_job(
            TrainingJob(
                job_id="single_job",
                config_file=args.config_file,
                character_name=args.character,
                output_dir=args.output_dir,
            )
        )

    try:
        results = orchestrator.run()
        orchestrator.save_results(results, Path(args.results_file))

        failed = results.get("failed", 0)
        return 1 if failed and failed > 0 else 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        return 1


def _run_dataset_cli(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Train LoRA from a prepared dataset folder")
    parser.add_argument("dataset_dir", help="Path to prepared training dataset")
    parser.add_argument("--character-name", "-n", required=True, help="Character name")
    parser.add_argument("--output-dir", "-o", help="Output directory for checkpoints")
    parser.add_argument("--base-model", choices=["sd15", "sdxl"], default="sdxl", help="Base model type")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--network-dim", type=int, default=32, help="LoRA network dimension")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--no-tmux", action="store_true", help="Run without TMUX")
    parser.add_argument("--no-eval", action="store_true", help="Skip auto-evaluation")

    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        output_dir = Path("/mnt/c/ai_models") / f"lora_{args.base_model}" / args.character_name

    config = TrainingConfig(
        character_name=args.character_name,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        network_dim=args.network_dim,
        learning_rate=args.learning_rate,
        enable_auto_eval=not args.no_eval,
    )

    orchestrator = UnifiedTrainingOrchestrator()
    result = orchestrator.train_character_lora(config, blocking=True, use_tmux=not args.no_tmux)

    print("\nTraining Result:")
    print(f"  Character: {result.character_name}")
    print(f"  Status: {result.status}")
    print(f"  Output: {result.output_dir}")
    print(f"  Checkpoints: {len(result.checkpoints)}")
    if result.best_checkpoint:
        print(f"  Best: {result.best_checkpoint}")
    print(f"  Time: {result.training_time:.1f}s")
    if result.error_message:
        print(f"  Error: {result.error_message}")

    return 0 if result.status == "completed" else 1


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if argv and argv[0] in {"sequential", "iterative"}:
        return _run_legacy_mode_cli(argv)
    return _run_dataset_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
