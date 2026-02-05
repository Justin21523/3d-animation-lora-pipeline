#!/usr/bin/env python3
"""
Generate SDXL pair-interaction LoRA TOML configs from a template.

Assumes training data is prepared in Kohya repeat folder format:
  {train_root}/{A}__{B}/{repeats}_{concept}/*.png + *.txt
and config should point train_data_dir to:
  {train_root}/{A}__{B}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List


def _iter_pairs(train_root: Path) -> Iterable[str]:
    for d in sorted([p for p in train_root.iterdir() if p.is_dir()]):
        # Expect at least one repeat_* concept folder
        if any(child.is_dir() and "_" in child.name for child in d.iterdir()):
            yield d.name


def _render_template(template: str, mapping: Dict[str, str]) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", type=Path, required=True)
    ap.add_argument("--template", type=Path, default=Path("configs/training/pair_lora_sdxl_template.toml"))
    ap.add_argument("--output-dir", type=Path, default=Path("configs/training/pair_loras_sdxl"))
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/data/ai_data/models/lora_sdxl/interaction_pairs"),
        help="Root output directory for trained LoRAs",
    )
    ap.add_argument(
        "--sample-prompts-root",
        type=Path,
        default=Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/prompts/lora_testing/pairs"),
        help="Root directory that contains <pair>.txt test prompts",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tmpl = args.template.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for pair in _iter_pairs(args.train_root):
        train_data_dir = args.train_root / pair
        out_dir = args.output_root / pair
        log_dir = out_dir / "logs"
        sample_prompts_file = args.sample_prompts_root / f"{pair}.txt"

        config_name = f"pair_{pair}_interaction_sdxl.toml"
        out_path = args.output_dir / config_name
        if out_path.exists() and not args.overwrite:
            continue

        mapping = {
            "TRAIN_DATA_DIR": str(train_data_dir),
            "OUTPUT_DIR": str(out_dir),
            "OUTPUT_NAME": f"pair_{pair}_interaction_lora_sdxl",
            "LOGGING_DIR": str(log_dir),
            "LOG_PREFIX": f"pair_{pair}",
            "SAMPLE_PROMPTS_FILE": str(sample_prompts_file),
        }
        out_path.write_text(_render_template(tmpl, mapping), encoding="utf-8")
        written.append(out_path)

    print(f"Wrote {len(written)} configs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
