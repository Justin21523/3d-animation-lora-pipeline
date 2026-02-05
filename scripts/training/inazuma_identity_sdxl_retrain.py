#!/usr/bin/env python3
"""
Sequential retrain for Inazuma Eleven SDXL identity LoRAs (UNet-only).

This launcher:
- uses Kohya sd-scripts `sdxl_train_network.py`
- runs 7 characters sequentially
- writes all outputs under: /mnt/data/training/lora/inazuma_eleven/

Usage:
  python -m scripts.training.inazuma_identity_sdxl_retrain --clean-output-root
  python -m scripts.training.inazuma_identity_sdxl_retrain --characters endou_mamoru,nosaka_yuuma
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_ROOT = Path("/mnt/data/training/lora/inazuma_eleven")
DEFAULT_KOHYA_ROOT = Path("/mnt/c/ai_tools/kohya_ss")
DEFAULT_CONDA_ENV = "kohya_ss"

CONFIG_DIR = REPO_ROOT / "configs" / "training" / "inazuma_eleven" / "identity_sdxl"

CHARACTER_IDS: List[str] = [
    "endou_mamoru",
    "fudou_akio",
    "gouenji_shuuya",
    "inamori_asuto",
    "matsukaze_tenma",
    "nosaka_yuuma",
    "utsunomiya_toramaru",
]


@dataclass(frozen=True)
class TrainJob:
    character_id: str
    config_path: Path

    @property
    def output_dir(self) -> Path:
        return DEFAULT_OUTPUT_ROOT / self.character_id


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_conda_env_python(conda_env: str) -> Optional[Path]:
    conda_base = os.environ.get("CONDA_PREFIX")
    # If we're inside some conda env already, CONDA_PREFIX points to it, not base.
    # Try common install base.
    candidates = [
        Path("/home/justin/miniconda3/envs") / conda_env / "bin" / "python",
        Path.home() / "miniconda3" / "envs" / conda_env / "bin" / "python",
        Path("/opt/conda/envs") / conda_env / "bin" / "python",
    ]
    if conda_base:
        base_guess = Path(conda_base).parents[1]  # .../envs/<env> -> .../
        candidates.insert(0, base_guess / "envs" / conda_env / "bin" / "python")

    for p in candidates:
        if p.exists():
            return p
    return None


def _ensure_exists(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")


def _parse_characters_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    characters = [c.strip() for c in value.split(",") if c.strip()]
    if not characters:
        return None
    return characters


def _build_jobs(selected_character_ids: Optional[Iterable[str]]) -> List[TrainJob]:
    selected = list(selected_character_ids) if selected_character_ids else CHARACTER_IDS

    jobs: List[TrainJob] = []
    for character_id in selected:
        config_path = CONFIG_DIR / f"train_{character_id}.toml"
        _ensure_exists(config_path, "Train config")
        jobs.append(TrainJob(character_id=character_id, config_path=config_path))
    return jobs


def _safe_clean_output_root(output_root: Path) -> None:
    # guardrails: only allow deleting the expected directory (or a subdir of it)
    expected = DEFAULT_OUTPUT_ROOT.resolve()
    candidate = output_root.resolve()
    if candidate != expected:
        raise ValueError(f"Refusing to clean unexpected output_root: {candidate} (expected exactly {expected})")

    if candidate.exists():
        archive_root = Path("/mnt/data/_archive_deleted/training_lora")
        archive_root.mkdir(parents=True, exist_ok=True)
        archive_path = archive_root / f"inazuma_eleven_{_now_tag()}"
        shutil.move(str(candidate), str(archive_path))

    candidate.mkdir(parents=True, exist_ok=True)


def _run_one(
    *,
    job: TrainJob,
    output_root: Path,
    kohya_root: Path,
    conda_env: str,
    dry_run: bool,
) -> dict:
    output_dir = output_root / job.character_id
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{job.character_id}_{_now_tag()}.log"

    wrapper = REPO_ROOT / "scripts" / "training" / "sdxl_train_safe_checkpointing.py"
    _ensure_exists(wrapper, "Safe training wrapper")

    python_exe = _resolve_conda_env_python(conda_env)
    if python_exe is None:
        cmd = [
            "conda",
            "run",
            "-n",
            conda_env,
            "python",
            "-u",
            str(wrapper),
            "--config_file",
            str(job.config_path),
        ]
    else:
        cmd = [
            str(python_exe),
            "-u",
            str(wrapper),
            "--config_file",
            str(job.config_path),
        ]

    record = {
        "character_id": job.character_id,
        "config_path": str(job.config_path),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "cmd": cmd,
        "start_time": datetime.now().isoformat(),
        "status": "dry_run" if dry_run else "running",
        "return_code": None,
        "duration_sec": None,
    }

    if dry_run:
        return record

    start = time.time()
    env = os.environ.copy()
    env["KOHYA_ROOT"] = str(kohya_root)

    with log_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )

    record["return_code"] = proc.returncode
    record["duration_sec"] = round(time.time() - start, 3)
    record["end_time"] = datetime.now().isoformat()
    record["status"] = "success" if proc.returncode == 0 else "failed"
    return record


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--characters", type=str, default=None, help="Comma-separated character IDs to train")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--kohya-root", type=Path, default=DEFAULT_KOHYA_ROOT)
    parser.add_argument("--conda-env", type=str, default=DEFAULT_CONDA_ENV)
    parser.add_argument("--clean-output-root", action="store_true", help="DELETE /mnt/data/training/lora/inazuma_eleven")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    _ensure_exists(CONFIG_DIR, "Config directory")
    _ensure_exists(args.kohya_root / "sd-scripts", "Kohya sd-scripts directory")

    selected = _parse_characters_arg(args.characters)
    jobs = _build_jobs(selected)

    if args.clean_output_root and not args.dry_run:
        _safe_clean_output_root(args.output_root)
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)

    run_tag = _now_tag()
    summary_path = args.output_root / f"training_summary_{run_tag}.json"

    results: List[dict] = []
    for i, job in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] training: {job.character_id}")
        result = _run_one(
            job=job,
            output_root=args.output_root,
            kohya_root=args.kohya_root,
            conda_env=args.conda_env,
            dry_run=args.dry_run,
        )
        results.append(result)

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_tag": run_tag,
                    "created_at": datetime.now().isoformat(),
                    "output_root": str(args.output_root),
                    "kohya_root": str(args.kohya_root),
                    "conda_env": args.conda_env,
                    "jobs": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        if not args.dry_run and result["status"] != "success":
            print(f"FAILED: {job.character_id} (see {result['log_path']})", file=sys.stderr)
            return 1

    print(f"Done. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
