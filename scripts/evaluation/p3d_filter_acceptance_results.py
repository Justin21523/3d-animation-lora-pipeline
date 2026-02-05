#!/usr/bin/env python3
"""
Filter acceptance_report.csv into accepted/rejected prompt lists.

Input:
  - acceptance_report.csv produced by scripts/evaluation/p3d_pair_acceptance_report.py

Outputs:
  - accepted_prompts.txt
  - rejected_prompts.txt
  - accepted.tsv / rejected.tsv (index + paths + key metrics)

This is intended for "few but high-quality" selection by default.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def safe_int(value: str) -> int:
    try:
        return int(value)
    except Exception:
        return -1


@dataclass(frozen=True)
class Row:
    index: int
    image_path: str
    prompt_path: str
    sharpness: float
    luma_mean: float
    luma_std: float
    sat_mean: float
    overexposed_frac: float
    underexposed_frac: float
    flags: str

    @staticmethod
    def from_csv(d: dict[str, str]) -> "Row":
        return Row(
            index=safe_int(d.get("index", "")),
            image_path=d.get("image_path", ""),
            prompt_path=d.get("prompt_path", ""),
            sharpness=safe_float(d.get("sharpness", "")),
            luma_mean=safe_float(d.get("luma_mean", "")),
            luma_std=safe_float(d.get("luma_std", "")),
            sat_mean=safe_float(d.get("sat_mean", "")),
            overexposed_frac=safe_float(d.get("overexposed_frac", "")),
            underexposed_frac=safe_float(d.get("underexposed_frac", "")),
            flags=d.get("flags", "").strip(),
        )


def load_rows(report_csv: Path) -> List[Row]:
    with report_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = [Row.from_csv(d) for d in r]
    rows = [x for x in rows if x.index >= 0]
    rows.sort(key=lambda x: x.index)
    return rows


def passes_strict(
    row: Row,
    *,
    min_sharpness: float,
    min_luma_std: float,
    min_luma_mean: float,
    max_luma_mean: float,
    max_overexposed_frac: float,
    max_underexposed_frac: float,
    require_no_flags: bool,
) -> bool:
    if require_no_flags and row.flags:
        return False
    if row.sharpness < min_sharpness:
        return False
    if row.luma_std < min_luma_std:
        return False
    if row.luma_mean < min_luma_mean:
        return False
    if row.luma_mean > max_luma_mean:
        return False
    if row.overexposed_frac > max_overexposed_frac:
        return False
    if row.underexposed_frac > max_underexposed_frac:
        return False
    return True


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-csv", required=True, help="Path to acceptance_report.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory for accepted/rejected lists")

    # Strict defaults: aim for "few but high-quality"
    ap.add_argument("--min-sharpness", type=float, default=120.0)
    ap.add_argument("--min-luma-std", type=float, default=0.10)
    ap.add_argument("--min-luma-mean", type=float, default=0.20)
    ap.add_argument("--max-luma-mean", type=float, default=0.90)
    ap.add_argument("--max-overexposed-frac", type=float, default=0.03)
    ap.add_argument("--max-underexposed-frac", type=float, default=0.03)
    ap.add_argument("--require-no-flags", type=int, default=1, help="1=require empty flags column")

    ap.add_argument("--limit-accept", type=int, default=0, help="0=no limit; otherwise cap accepted count")
    ap.add_argument("--limit-reject", type=int, default=0, help="0=no limit; otherwise cap rejected count")
    args = ap.parse_args()

    report_csv = Path(args.report_csv)
    if not report_csv.exists():
        raise SystemExit(f"report not found: {report_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(report_csv)
    if not rows:
        raise SystemExit(f"no rows in report: {report_csv}")

    accepted: List[Row] = []
    rejected: List[Row] = []
    for row in rows:
        ok = passes_strict(
            row,
            min_sharpness=args.min_sharpness,
            min_luma_std=args.min_luma_std,
            min_luma_mean=args.min_luma_mean,
            max_luma_mean=args.max_luma_mean,
            max_overexposed_frac=args.max_overexposed_frac,
            max_underexposed_frac=args.max_underexposed_frac,
            require_no_flags=bool(args.require_no_flags),
        )
        if ok:
            accepted.append(row)
        else:
            rejected.append(row)

    if args.limit_accept and args.limit_accept > 0:
        accepted = accepted[: args.limit_accept]
    if args.limit_reject and args.limit_reject > 0:
        rejected = rejected[: args.limit_reject]

    def prompt_for(row: Row) -> str:
        if row.prompt_path and Path(row.prompt_path).exists():
            return read_text(Path(row.prompt_path))
        # Fallback: try to derive prompt path from image path
        img = Path(row.image_path)
        maybe = img.with_suffix(".txt")
        if maybe.exists():
            return read_text(maybe)
        return ""

    accepted_prompts = [prompt_for(r) for r in accepted]
    rejected_prompts = [prompt_for(r) for r in rejected]
    accepted_prompts = [p for p in accepted_prompts if p]
    rejected_prompts = [p for p in rejected_prompts if p]

    write_lines(out_dir / "accepted_prompts.txt", accepted_prompts)
    write_lines(out_dir / "rejected_prompts.txt", rejected_prompts)

    def write_tsv(path: Path, items: List[Row]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(
                [
                    "index",
                    "sharpness",
                    "luma_mean",
                    "luma_std",
                    "sat_mean",
                    "overexposed_frac",
                    "underexposed_frac",
                    "flags",
                    "image_path",
                    "prompt_path",
                ]
            )
            for r in items:
                w.writerow(
                    [
                        r.index,
                        f"{r.sharpness:.4f}",
                        f"{r.luma_mean:.6f}",
                        f"{r.luma_std:.6f}",
                        f"{r.sat_mean:.6f}",
                        f"{r.overexposed_frac:.6f}",
                        f"{r.underexposed_frac:.6f}",
                        r.flags,
                        r.image_path,
                        r.prompt_path,
                    ]
                )

    write_tsv(out_dir / "accepted.tsv", accepted)
    write_tsv(out_dir / "rejected.tsv", rejected)

    summary = {
        "report_csv": str(report_csv),
        "rows": len(rows),
        "accepted": len(accepted_prompts),
        "rejected": len(rejected_prompts),
        "thresholds": {
            "min_sharpness": args.min_sharpness,
            "min_luma_std": args.min_luma_std,
            "min_luma_mean": args.min_luma_mean,
            "max_luma_mean": args.max_luma_mean,
            "max_overexposed_frac": args.max_overexposed_frac,
            "max_underexposed_frac": args.max_underexposed_frac,
            "require_no_flags": bool(args.require_no_flags),
        },
    }
    (out_dir / "filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

