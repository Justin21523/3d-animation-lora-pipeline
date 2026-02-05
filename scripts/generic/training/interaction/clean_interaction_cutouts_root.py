#!/usr/bin/env python3
"""
Clean interaction cutout root directory.

Currently the pipeline expects:
  /mnt/data/.../interaction_cutouts/<character>/*.png (+ optional *.json)

Some older runs wrote outputs under:
  /mnt/data/.../interaction_cutouts/<character>/cutouts/*.png

This tool merges that legacy subfolder into the character folder, moving any
conflicting duplicates into a timestamped trash folder.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
from pathlib import Path
from typing import Optional


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_same_file(a: Path, b: Path) -> bool:
    try:
        if a.stat().st_size != b.stat().st_size:
            return False
        # Hash only on same-size collisions
        return _sha256(a) == _sha256(b)
    except Exception:
        return False


def _unique_trash_path(trash_dir: Path, name: str) -> Path:
    p = trash_dir / name
    if not p.exists():
        return p
    stem = Path(name).stem
    suf = Path(name).suffix
    for i in range(1, 10_000):
        cand = trash_dir / f"{stem}__dup{i}{suf}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not find unique trash path for {name}")


def merge_legacy_cutouts(
    cutout_root: Path,
    legacy_subdir_name: str,
    trash_root: Path,
    dry_run: bool,
) -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_trash = trash_root / f"interaction_cutouts_cleanup_{stamp}"

    moved = 0
    deduped = 0
    trashed = 0

    for char_dir in sorted([p for p in cutout_root.iterdir() if p.is_dir()]):
        legacy_dir = char_dir / legacy_subdir_name
        if not legacy_dir.exists() or not legacy_dir.is_dir():
            continue

        char_trash = run_trash / char_dir.name / legacy_subdir_name
        char_trash.mkdir(parents=True, exist_ok=True)

        for src in sorted(legacy_dir.iterdir()):
            if not src.is_file():
                continue
            if src.suffix.lower() not in {".png", ".json"}:
                continue

            dst = char_dir / src.name
            if not dst.exists():
                if not dry_run:
                    shutil.move(str(src), str(dst))
                moved += 1
                continue

            if _is_same_file(src, dst):
                # keep dst, discard src
                if not dry_run:
                    src.unlink(missing_ok=True)
                deduped += 1
                continue

            # Conflict: keep dst, move src to trash
            trash_path = _unique_trash_path(char_trash, src.name)
            if not dry_run:
                shutil.move(str(src), str(trash_path))
            trashed += 1

        # Remove legacy dir if empty
        try:
            if not any(legacy_dir.iterdir()):
                if not dry_run:
                    legacy_dir.rmdir()
        except Exception:
            pass

    print(f"Legacy merge complete. moved={moved} deduped={deduped} trashed={trashed}")
    if moved or deduped or trashed:
        print(f"Trash (if any): {run_trash}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cutout-root",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/interaction_cutouts"),
    )
    ap.add_argument("--legacy-subdir-name", default="cutouts")
    ap.add_argument(
        "--trash-root",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/_trash"),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    merge_legacy_cutouts(
        cutout_root=args.cutout_root,
        legacy_subdir_name=args.legacy_subdir_name,
        trash_root=args.trash_root,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

