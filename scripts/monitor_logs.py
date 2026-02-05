#!/usr/bin/env python
"""
Monitor recent log files (tail).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tail recent logs in logs/ directory.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory containing logs.")
    parser.add_argument("--count", type=int, default=5, help="Number of recent log files to show.")
    parser.add_argument("--lines", type=int, default=20, help="Number of lines to tail from each file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"[WARN] Log directory {log_dir} does not exist.")
        return

    logs: List[Path] = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    logs = logs[: args.count]
    if not logs:
        print("[INFO] No log files found.")
        return

    for log_file in logs:
        print(f"\n=== {log_file} ===")
        try:
            with log_file.open("r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                tail = lines[-args.lines :]
                for line in tail:
                    print(line.rstrip())
        except Exception as exc:
            print(f"[WARN] Failed to read {log_file}: {exc}")


if __name__ == "__main__":
    main()
