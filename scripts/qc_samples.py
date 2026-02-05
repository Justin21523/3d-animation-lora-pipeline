#!/usr/bin/env python
"""
Quick QC script: prints counts and example paths for generated assets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick QC for generated assets.")
    parser.add_argument("--metadata", type=str, nargs="+", required=True, help="List of parquet/csv metadata files to inspect.")
    parser.add_argument("--head", type=int, default=5, help="Number of sample rows to print per file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for meta in args.metadata:
        path = Path(meta)
        if not path.exists():
            print(f"[WARN] {path} missing.")
            continue
        try:
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] Failed to read {path}: {exc}")
            continue
        print(f"\n=== {path} ===")
        print(f"rows: {len(df)}")
        print(df.head(args.head).to_string(index=False))


if __name__ == "__main__":
    main()
