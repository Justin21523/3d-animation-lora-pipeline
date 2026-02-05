#!/usr/bin/env python
"""
Simple loss monitor: parse log files for 'loss=' patterns and print recent values.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor loss values from logs.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory containing logs.")
    parser.add_argument("--pattern", type=str, default="loss=", help="Substring to search for.")
    parser.add_argument("--count", type=int, default=5, help="Number of recent log files to scan.")
    parser.add_argument("--tail", type=int, default=200, help="Number of lines from end of file to scan.")
    return parser.parse_args()


def extract_losses(text: List[str], pattern: str) -> List[Tuple[str, float]]:
    losses = []
    for line in text:
        if pattern in line:
            matches = re.findall(r"loss=([0-9]*\\.?[0-9]+)", line)
            for m in matches:
                try:
                    losses.append((line.strip(), float(m)))
                except ValueError:
                    continue
    return losses


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"[WARN] Log directory {log_dir} does not exist.")
        return

    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[: args.count]
    if not logs:
        print("[INFO] No log files found.")
        return

    for log_file in logs:
        print(f"\n=== {log_file} ===")
        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        tail_lines = lines[-args.tail :]
        losses = extract_losses(tail_lines, args.pattern)
        if not losses:
            print("No loss values found.")
            continue
        for line, val in losses[-10:]:
            print(line)
        last_val = losses[-1][1]
        print(f"Last loss: {last_val}")


if __name__ == "__main__":
    main()
