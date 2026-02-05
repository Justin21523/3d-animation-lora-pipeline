#!/usr/bin/env python3
"""
Train pair-interaction SDXL LoRAs sequentially and run a "full" evaluation after each pair.

Training:
  conda run -n kohya_ss accelerate launch sdxl_train_network.py --config_file <pair_config.toml>

Mid-training samples:
  Handled by Kohya via the config's [sample_generation] section.

Post-training full evaluation:
  Uses scripts/evaluation/sdxl_multi_lora_compositor.py with:
    - identity LoRA A (BEST checkpoint)
    - identity LoRA B (BEST checkpoint)
    - trained pair-interaction LoRA checkpoint (epoch1 + epoch2)
  and low interaction weights to test generalization on varied backgrounds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BASE_MODEL = Path("/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors")
BEST_IDENTITY_DIR = Path("/mnt/c/ai_models/lora_sdxl/BEST_CHECKPOINTS_COLLECTION")
PROJECT_ROOT = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline")


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _pair_from_config_name(stem: str) -> str:
    # pair_<A>__<B>_interaction_sdxl
    if stem.startswith("pair_") and stem.endswith("_interaction_sdxl"):
        return stem[len("pair_") : -len("_interaction_sdxl")]
    raise ValueError(f"Unexpected config name: {stem}")


def _best_identity_lora(character: str) -> Path:
    cand = BEST_IDENTITY_DIR / f"BEST_{character}_lora_sdxl.safetensors"
    if not cand.exists():
        raise FileNotFoundError(f"Missing identity LoRA for {character}: {cand}")
    return cand


def _latest_checkpoint(output_dir: Path, output_name: str) -> Path:
    cands = sorted(output_dir.glob(f"{output_name}-*.safetensors"))
    if not cands:
        # Fallback to any safetensors
        cands = sorted(output_dir.glob("*.safetensors"))
    if not cands:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    # Choose by name sort (epoch numbers are zero-padded)
    return cands[-1]


def _run_training(config_path: Path, num_cpu_threads: int) -> None:
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
        str(int(num_cpu_threads)),
        "/mnt/c/ai_projects/kohya_ss/sd-scripts/sdxl_train_network.py",
        "--config_file",
        str(config_path),
    ]
    subprocess.run(cmd, check=True)


def _run_full_eval(
    pair: str,
    pair_checkpoint: Path,
    prompts_file: Path,
    out_dir: Path,
    weight_a: float,
    weight_b: float,
    weight_pair: float,
    steps: int,
    guidance: float,
    num_samples: int,
) -> None:
    a, b = pair.split("__", 1)
    id_a = _best_identity_lora(a)
    id_b = _best_identity_lora(b)

    compositor = PROJECT_ROOT / "scripts/evaluation/sdxl_multi_lora_compositor.py"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep lora-names away from prompt tokens so smart selection loads all.
    loras = [str(id_a), str(id_b), str(pair_checkpoint)]
    names = ["loraA", "loraB", "loraC"]
    combos = [f"{weight_a},{weight_b},{weight_pair}"]
    # Use ai_env python directly (diffusers installed there)
    py = "/home/justin/miniconda3/envs/ai_env/bin/python"
    cmd = [
        py,
        str(compositor),
        "--loras",
        *loras,
        "--lora-names",
        *names,
        "--weight-combos",
        *combos,
        "--prompts-file",
        str(prompts_file),
        "--negative-prompt",
        "multiple people, crowd, extra limbs, extra fingers, bad anatomy, deformed, ugly, low quality, blurry, watermark, text",
        "--base-model",
        str(BASE_MODEL),
        "--output-dir",
        str(out_dir),
        "--num-samples",
        str(int(num_samples)),
        "--steps",
        str(int(steps)),
        "--guidance-scale",
        str(float(guidance)),
        "--seed-start",
        "42",
        "--width",
        "1024",
        "--height",
        "1024",
        "--device",
        "cuda",
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", type=Path, default=Path("configs/training/pair_loras_sdxl"))
    ap.add_argument("--start-from", default=None)
    ap.add_argument("--num-cpu-threads", type=int, default=32)
    ap.add_argument("--progress-file", type=Path, default=Path("/tmp/pair_lora_training_progress.json"))
    ap.add_argument("--log-dir", type=Path, default=Path("logs/pair_training"))

    ap.add_argument(
        "--pairs-root",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/interaction_pairs_by_pair_20260117_210900"),
        help="Used only to locate per-pair prompt files by name",
    )
    ap.add_argument(
        "--prompts-root",
        type=Path,
        default=Path("prompts/lora_testing/pairs"),
    )
    ap.add_argument(
        "--eval-root",
        type=Path,
        default=Path("outputs/evaluation/pair_interactions"),
    )
    ap.add_argument("--eval-samples", type=int, default=1)
    ap.add_argument("--eval-steps", type=int, default=30)
    ap.add_argument("--eval-guidance", type=float, default=7.5)
    ap.add_argument("--identity-weight-a", type=float, default=1.2)
    ap.add_argument("--identity-weight-b", type=float, default=0.7)
    ap.add_argument("--pair-weight", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    progress = _load_progress(args.progress_file)
    completed = set(progress.get("completed", []))

    configs = sorted(args.config_dir.glob("pair_*_interaction_sdxl.toml"))
    if not configs:
        raise SystemExit(f"No pair configs found in {args.config_dir}")

    if args.start_from:
        start_key = args.start_from
        if not start_key.endswith(".toml"):
            start_key += ".toml"
        found = False
        filtered: List[Path] = []
        for c in configs:
            if c.name == start_key:
                found = True
            if found:
                filtered.append(c)
        if not found:
            raise SystemExit(f"--start-from not found: {args.start_from}")
        configs = filtered

    # Skip completed
    configs = [c for c in configs if c.stem not in completed and c.name not in completed]
    print(f"Pairs to train: {len(configs)}")

    for idx, config_path in enumerate(configs, start=1):
        name = config_path.stem
        pair = _pair_from_config_name(name)

        log_file = args.log_dir / f"{name}_{_now_stamp()}.log"
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(configs)}] TRAIN: {pair}")
        print(f"Config: {config_path}")
        print(f"Log: {log_file}")
        print("=" * 80)

        if args.dry_run:
            continue

        progress["current"] = name
        _save_progress(args.progress_file, progress)

        # Run training with stdout/stderr tee'd to a log file
        start = time.time()
        try:
            with log_file.open("a", encoding="utf-8") as lf:
                lf.write(f"START {datetime.now().isoformat()}\n")
                lf.flush()
                proc = subprocess.run(
                    [
                        "bash",
                        "-lc",
                        f"conda run -n kohya_ss accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process {int(args.num_cpu_threads)} "
                        f"/mnt/c/ai_projects/kohya_ss/sd-scripts/sdxl_train_network.py --config_file '{config_path}'",
                    ],
                    check=True,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )

            elapsed_h = (time.time() - start) / 3600.0
            print(f"✅ Training done: {pair} ({elapsed_h:.2f}h)")

            # Read output_dir/output_name from TOML via grep (fast + robust)
            output_dir = Path(
                subprocess.check_output(["bash", "-lc", f"grep '^output_dir' '{config_path}' | sed 's/.*\"\\(.*\\)\".*/\\1/'"]).decode().strip()
            )
            output_name = (
                subprocess.check_output(["bash", "-lc", f"grep '^output_name' '{config_path}' | sed 's/.*\"\\(.*\\)\".*/\\1/'"]).decode().strip()
            )
            max_epochs = int(
                subprocess.check_output(
                    ["bash", "-lc", f"grep '^max_train_epochs' '{config_path}' | awk '{{print $3}}'"]
                ).decode().strip()
            )

            prompts_file = args.prompts_root / f"{pair}.txt"
            if not prompts_file.exists():
                raise FileNotFoundError(f"Missing prompts file: {prompts_file}")

            # Full eval for each epoch checkpoint (epoch1 + epoch2).
            # This provides per-epoch results without requiring mid-training hooks.
            for epoch in range(1, max_epochs + 1):
                ckpt_name = f"{output_name}-{epoch:06d}.safetensors"
                ckpt = output_dir / ckpt_name
                if not ckpt.exists():
                    # fallback: try latest if naming differs
                    ckpt = _latest_checkpoint(output_dir, output_name)

                eval_out = args.eval_root / pair / f"epoch{epoch}" / f"{_now_stamp()}_{ckpt.stem}"
                print(f"🧪 Full eval (epoch {epoch}) -> {eval_out}")
                _run_full_eval(
                    pair=pair,
                    pair_checkpoint=ckpt,
                    prompts_file=prompts_file,
                    out_dir=eval_out,
                    weight_a=float(args.identity_weight_a),
                    weight_b=float(args.identity_weight_b),
                    weight_pair=float(args.pair_weight),
                    steps=int(args.eval_steps),
                    guidance=float(args.eval_guidance),
                    num_samples=int(args.eval_samples),
                )

            progress["completed"].append(name)
        except Exception as e:
            print(f"❌ Failed: {pair}: {e}")
            progress["failed"].append({"name": name, "pair": pair, "error": str(e), "time": datetime.now().isoformat()})
        finally:
            progress["current"] = None
            _save_progress(args.progress_file, progress)

    print("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
