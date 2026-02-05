#!/usr/bin/env python3
"""
Sequential trainer for pair-interaction SDXL LoRAs.

Runs kohya_ss sdxl_train_network.py one config at a time (GPU-safe).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def _load_progress(path: Path) -> Dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "started_at": datetime.now().isoformat(),
        "completed": [],
        "failed": [],
        "current": None,
        "last_updated": None,
    }


def _save_progress(path: Path, progress: Dict) -> None:
    progress["last_updated"] = datetime.now().isoformat()
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config-dir",
        type=Path,
        default=Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/pair_loras_sdxl"),
    )
    ap.add_argument(
        "--sd-scripts",
        type=Path,
        default=Path("/mnt/c/ai_projects/kohya_ss/sd-scripts"),
    )
    ap.add_argument("--start-from", default=None, help="Start from config filename (without .toml also ok)")
    ap.add_argument("--num-cpu-threads", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--progress-file", type=Path, default=Path("/tmp/pair_lora_training_progress.json"))
    args = ap.parse_args()

    cfgs = sorted(args.config_dir.glob("*.toml"))
    if not cfgs:
        raise SystemExit(f"No configs found in {args.config_dir}")

    progress = _load_progress(args.progress_file)
    completed = set(progress.get("completed", []))

    # Apply start-from
    if args.start_from:
        start_key = args.start_from
        if not start_key.endswith(".toml"):
            start_key += ".toml"
        found = False
        filtered: List[Path] = []
        for c in cfgs:
            if c.name == start_key:
                found = True
            if found:
                filtered.append(c)
        if not found:
            raise SystemExit(f"--start-from not found: {args.start_from}")
        cfgs = filtered

    # Skip completed
    cfgs = [c for c in cfgs if c.stem not in completed and c.name not in completed]

    print(f"Configs to train: {len(cfgs)}")
    print(f"Progress file: {args.progress_file}")

    for idx, config_path in enumerate(cfgs, start=1):
        name = config_path.stem
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(cfgs)}] Training: {name}")
        print("=" * 80)
        print(f"Config: {config_path}")

        if args.dry_run:
            print("DRY RUN")
            continue

        progress["current"] = name
        _save_progress(args.progress_file, progress)

        cmd = [
            "conda",
            "run",
            "-n",
            "kohya_ss",
            "accelerate",
            "launch",
            "--mixed_precision",
            "bf16",
            "--num_cpu_threads_per_process",
            str(int(args.num_cpu_threads)),
            str(args.sd_scripts / "sdxl_train_network.py"),
            "--config_file",
            str(config_path),
        ]

        start = time.time()
        try:
            subprocess.run(cmd, check=True)
            elapsed_h = (time.time() - start) / 3600.0
            print(f"✅ Done: {name} ({elapsed_h:.2f}h)")
            progress["completed"].append(name)
        except subprocess.CalledProcessError as e:
            elapsed_h = (time.time() - start) / 3600.0
            print(f"❌ Failed: {name} ({elapsed_h:.2f}h): {e}")
            progress["failed"].append({"name": name, "error": str(e), "time": datetime.now().isoformat()})
        finally:
            progress["current"] = None
            _save_progress(args.progress_file, progress)

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

