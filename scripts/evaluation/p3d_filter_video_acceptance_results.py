#!/usr/bin/env python3
"""
Filter video QA results into accepted/rejected lists and (optionally) export captions.

This is a simple utility to avoid manually reviewing hundreds/thousands of clips.
It uses the `flags` column produced by `p3d_video_acceptance_report.py`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-csv", required=True, help="Path to video_acceptance_report.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--captions-dir", default="", help="Optional captions dir (<id>.txt) to copy for accepted")
    ap.add_argument("--max-accepted", type=int, default=0, help="0=all")
    args = ap.parse_args()

    report_csv = Path(args.report_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted_ids: list[str] = []
    rejected_ids: list[str] = []

    with report_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            vid_id = (r.get("video_id") or "").strip()
            flags = (r.get("flags") or "").strip()
            if not vid_id:
                continue
            if flags:
                rejected_ids.append(vid_id)
            else:
                accepted_ids.append(vid_id)

    if args.max_accepted and args.max_accepted > 0:
        accepted_ids = accepted_ids[: args.max_accepted]

    (out_dir / "accepted_ids.txt").write_text("\n".join(accepted_ids) + "\n", encoding="utf-8")
    (out_dir / "rejected_ids.txt").write_text("\n".join(rejected_ids) + "\n", encoding="utf-8")

    if args.captions_dir:
        captions_dir = Path(args.captions_dir)
        out_caps = out_dir / "captions"
        out_caps.mkdir(parents=True, exist_ok=True)
        for vid_id in accepted_ids:
            src = captions_dir / f"{vid_id}.txt"
            if not src.exists():
                continue
            dst = out_caps / src.name
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    print(
        {
            "accepted": len(accepted_ids),
            "rejected": len(rejected_ids),
            "out_dir": str(out_dir),
        }
    )


if __name__ == "__main__":
    main()

