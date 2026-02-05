#!/usr/bin/env python3
"""
Export an AnimateDiff run folder (videos + manifest) into a training-ready dataset layout.

Output layout:
  <out_dir>/
    videos/<id>.mp4
    captions/<id>.txt
    metadata.jsonl               (one json per line: {"file_name": "...", "text": "..."})
    manifest.tsv                 (copied from run)
    negative.txt                 (optional)
    README.md

This stays under outputs/ by default (large artifacts are intentionally not versioned).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="AnimateDiff run dir (contains videos/manifest.tsv)")
    ap.add_argument("--captions-dir", required=True, help="Caption dir (contains <id>.txt)")
    ap.add_argument("--out-dir", required=True, help="Output dataset directory")
    ap.add_argument("--negative", default="", help="Negative prompt file to copy (optional)")
    ap.add_argument(
        "--accepted-ids",
        default="",
        help="Optional newline-separated list of ids to export (others skipped)",
    )
    ap.add_argument("--copy", type=int, default=1, help="1=copy files, 0=hardlink when possible")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    videos_dir = run_dir / "videos"
    manifest_path = videos_dir / "manifest.tsv"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.tsv not found: {manifest_path}")

    captions_dir = Path(args.captions_dir)
    if not captions_dir.exists():
        raise SystemExit(f"captions dir not found: {captions_dir}")

    out_dir = Path(args.out_dir)
    out_videos = out_dir / "videos"
    out_caps = out_dir / "captions"
    out_videos.mkdir(parents=True, exist_ok=True)
    out_caps.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)

    accepted: set[str] | None = None
    if args.accepted_ids:
        p = Path(args.accepted_ids)
        if not p.exists():
            raise SystemExit(f"accepted ids file not found: {p}")
        accepted = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}

    exported = 0
    meta_path = out_dir / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as mf:
        for r in rows:
            vid_id = r["id"]
            if accepted is not None and vid_id not in accepted:
                continue
            src_vid = Path(r["output_path"])
            if not src_vid.exists():
                continue
            src_cap = captions_dir / f"{vid_id}.txt"
            if not src_cap.exists():
                continue

            dst_vid = out_videos / f"{vid_id}.mp4"
            dst_cap = out_caps / f"{vid_id}.txt"

            if args.copy:
                shutil.copy2(src_vid, dst_vid)
                shutil.copy2(src_cap, dst_cap)
            else:
                try:
                    if dst_vid.exists():
                        dst_vid.unlink()
                    if dst_cap.exists():
                        dst_cap.unlink()
                    dst_vid.hardlink_to(src_vid)
                    dst_cap.hardlink_to(src_cap)
                except Exception:
                    shutil.copy2(src_vid, dst_vid)
                    shutil.copy2(src_cap, dst_cap)

            cap_text = _read_text(dst_cap)
            # Wan2.1 (DiffSynth-Studio) trainer expects `video` as the data key (see --data_file_keys "video").
            mf.write(
                json.dumps(
                    {
                        "file_name": f"videos/{vid_id}.mp4",
                        "video": f"videos/{vid_id}.mp4",
                        "prompt": cap_text,
                        "text": cap_text,
                        "caption": cap_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            exported += 1

    shutil.copy2(manifest_path, out_dir / "manifest.tsv")
    if args.negative:
        neg = Path(args.negative)
        if neg.exists():
            shutil.copy2(neg, out_dir / "negative.txt")

    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# P3D Pair Videos Dataset",
                "",
                "Contents:",
                "- `videos/`: mp4 clips (AnimateDiff)",
                "- `captions/`: one caption per video (same id)",
                "- `metadata.jsonl`: jsonl mapping used by many trainers",
                "- `manifest.tsv`: original batch manifest",
                "- `negative.txt`: optional negative prompt used at generation time",
                "",
                "Notes:",
                "- This dataset is generated under `outputs/` and is not meant to be committed.",
                "- If your trainer expects a different schema, use `metadata.jsonl` as the source of truth.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {"exported": exported, "out_dir": str(out_dir), "filtered": bool(args.accepted_ids)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
