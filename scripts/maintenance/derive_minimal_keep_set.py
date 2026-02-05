#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]


ENTRYPOINTS = [
    # A (3D CLI)
    Path("scripts/core/pipeline/__main__.py"),
    # B (2D CLI)
    Path("scripts/run_pipeline.py"),
    # C (Wan2.1)
    Path("scripts/batch/prepare_wan21_dataset_from_p3d_image_acceptance.py"),
    Path("scripts/batch/launch_prepare_wan21_dataset_from_p3d_images_qc_mild_nohup.sh"),
    Path("scripts/batch/launch_wan21_lora_train_p3d_pairs_from_images_nohup.sh"),
    Path("scripts/batch/launch_wan21_sft_cache_p3d_pairs_from_images_cpu_nohup.sh"),
    Path("scripts/batch/launch_wan21_sft_train_from_cache_p3d_pairs_from_images_nohup.sh"),
    # D (SDXL/Kohya)
    Path("scripts/training/unified_training_orchestrator.py"),
    Path("scripts/training/training_manager.py"),
    Path("scripts/training/training_monitor.py"),
    Path("scripts/training/monitor_sdxl_training.py"),
    Path("scripts/training/generate_sdxl_lora_configs.py"),
    Path("scripts/train_lora_sd.py"),
    Path("scripts/train_controlnet_pose.py"),
    Path("start_training_with_log.sh"),
    Path("monitor_training_progress.sh"),
    Path("resume_luca_training.sh"),
    Path("safe_view_training.sh"),
    Path("check_progress.sh"),
]

ALWAYS_KEEP_FILES = [
    # Repo meta / primary docs
    Path("README.md"),
    Path("README_2D.md"),
    Path("AGENTS.md"),
    # Minimal docs to support A/B/C/D
    Path("docs/guides/quick_start.md"),
    Path("docs/setup"),
    Path("docs/3d-training/guides/WAN21_TWO_STAGE_SFT_GUIDE.md"),
    # Maintenance utilities
    Path("scripts/maintenance/cleanup_repo_artifacts.sh"),
]


PY_IMPORT_REPO_PREFIXES = ("scripts.", "anime_pipeline.")


@dataclass(frozen=True)
class Reason:
    kind: str
    detail: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _is_within_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT)
        return True
    except Exception:
        return False


def _normalize_rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def _module_to_path(module: str) -> Optional[Path]:
    # Only resolve local modules under known prefixes to avoid over-keeping.
    if not module.startswith(PY_IMPORT_REPO_PREFIXES):
        return None
    rel = Path(*module.split("."))
    py_file = PROJECT_ROOT / f"{rel}.py"
    if py_file.exists():
        return py_file
    pkg_init = PROJECT_ROOT / rel / "__init__.py"
    if pkg_init.exists():
        return pkg_init
    return None


def _path_to_module(path: Path) -> Optional[str]:
    """
    Best-effort convert a repo path to a Python module path, e.g.
      scripts/core/pipeline/__main__.py -> scripts.core.pipeline.__main__
      scripts/core/pipeline/__init__.py -> scripts.core.pipeline
    """
    if not _is_within_repo(path) or path.suffix.lower() != ".py":
        return None
    rel = path.resolve().relative_to(PROJECT_ROOT)
    parts = list(rel.parts)
    if not parts:
        return None
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = Path(parts[-1]).stem
    if not parts:
        return None
    return ".".join(parts)


def _resolve_relative_import(cur_file: Path, node: ast.ImportFrom) -> list[str]:
    """
    Resolve ImportFrom with relative levels into absolute module names.
    Returns a list of module strings to consider.
    """
    cur_mod = _path_to_module(cur_file)
    if not cur_mod:
        return []

    # Determine current package (without the module leaf)
    cur_pkg_parts = cur_mod.split(".")
    if cur_file.name != "__init__.py" and cur_pkg_parts:
        cur_pkg_parts = cur_pkg_parts[:-1]

    level = int(getattr(node, "level", 0) or 0)
    if level <= 0:
        return [node.module] if node.module else []

    if level > len(cur_pkg_parts):
        return []
    base_parts = cur_pkg_parts[: len(cur_pkg_parts) - level + 1]

    resolved: list[str] = []
    if node.module:
        resolved.append(".".join(base_parts + node.module.split(".")))
    else:
        # from . import x
        for alias in node.names:
            resolved.append(".".join(base_parts + [alias.name]))
    return resolved


def _parse_python_imports(path: Path) -> set[tuple[str, str]]:
    """
    Return a set of (imported_module, via) pairs where via is a short description.
    """
    text = _read_text(path)
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return set()
    out: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add((alias.name, "import"))
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_relative_import(path, node)
            for mod in resolved:
                if mod:
                    out.add((mod, "from"))
        elif isinstance(node, ast.Call):
            # importlib.import_module("x.y")
            fn = node.func
            fn_name = None
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                fn_name = f"{fn.value.id}.{fn.attr}"
            if fn_name == "importlib.import_module" and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    out.add((arg0.value, "importlib.import_module"))
    return out


_PATH_TOKEN_RE = re.compile(
    r"""
    (?:
        (?P<q>["'])
        (?P<p1>
            (?:scripts|anime_pipeline|configs|docs|prompts)/[^"' \t\r\n]+
        )
        (?P=q)
    )
    |
    (?P<p2>
        (?:scripts|anime_pipeline|configs|docs|prompts)/[^ \t\r\n]+
    )
    """,
    re.VERBOSE,
)

_PY_CONSTRUCTED_SCRIPT_LINE_RE = re.compile(r""".*['"].*\.(?:py|sh|bash)['"].*""")
_QUOTED_TOKEN_RE = re.compile(r"""['"]([^'"]+)['"]""")
_SCRIPT_ROOT_TOKENS = {"generic", "segmentation", "training", "evaluation", "setup", "utils", "core", "batch", "pipelines", "optimization"}


def _extract_repo_paths_from_text(text: str) -> set[str]:
    hits: set[str] = set()
    for m in _PATH_TOKEN_RE.finditer(text):
        p = m.group("p1") or m.group("p2")
        if not p:
            continue
        # Trim trailing punctuation that commonly appears in docs snippets.
        p = p.rstrip(").,;:\"'")
        hits.add(p)
    return hits


def _parse_shell_references(path: Path) -> set[str]:
    text = _read_text(path)
    return _extract_repo_paths_from_text(text)

def _extract_constructed_scripts_from_python(text: str) -> set[str]:
    """
    Heuristic: catch subprocess target paths built like:
      Path(__file__).parent.parent.parent / 'generic' / 'video' / 'tool.py'
    and convert to repo path:
      scripts/generic/video/tool.py
    """
    hits: set[str] = set()
    for line in text.splitlines():
        if "Path(__file__)" not in line and "Path(" not in line:
            continue
        if " / " not in line:
            continue
        if not _PY_CONSTRUCTED_SCRIPT_LINE_RE.match(line):
            continue
        tokens = _QUOTED_TOKEN_RE.findall(line)
        if not tokens:
            continue
        if tokens[0] not in _SCRIPT_ROOT_TOKENS:
            continue
        if not (tokens[-1].endswith(".py") or tokens[-1].endswith(".sh") or tokens[-1].endswith(".bash")):
            continue
        candidate = Path("scripts")
        for t in tokens:
            candidate = candidate / t
        hits.add(str(candidate).replace("\\", "/"))
    return hits


def _iter_files(root: Path, exts: Optional[set[str]] = None) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if exts is not None and p.suffix.lower() not in exts:
            continue
        yield p


def main() -> int:
    # Validate entrypoints exist; if some don't, keep going (repo variants).
    existing_entrypoints: list[Path] = []
    missing_entrypoints: list[str] = []
    for ep in ENTRYPOINTS:
        abs_ep = PROJECT_ROOT / ep
        if abs_ep.exists():
            existing_entrypoints.append(abs_ep)
        else:
            missing_entrypoints.append(str(ep))

    keep: dict[str, list[Reason]] = {}

    def mark_keep(path: Path, reason: Reason):
        if not path.exists():
            return
        if not _is_within_repo(path):
            return
        rel = _normalize_rel(path)
        keep.setdefault(rel, []).append(reason)

    # Always keep minimal roots required for A/B/C/D to work.
    always_keep_dirs = [
        "scripts",
        "anime_pipeline",
        "configs",
        "requirements",
        "tests",
    ]
    for d in always_keep_dirs:
        p = PROJECT_ROOT / d
        if p.exists():
            mark_keep(p, Reason("dir_required", d))

    for k in ALWAYS_KEEP_FILES:
        abs_k = (PROJECT_ROOT / k)
        if abs_k.exists():
            mark_keep(abs_k, Reason("always_keep", str(k).replace("\\", "/")))

    for ep in existing_entrypoints:
        mark_keep(ep, Reason("entrypoint", _normalize_rel(ep)))

    # BFS over python imports and textual path references.
    queue: list[tuple[Path, Reason]] = [(ep, Reason("entrypoint", _normalize_rel(ep))) for ep in existing_entrypoints]
    seen: set[str] = set(_normalize_rel(ep) for ep in existing_entrypoints)

    # Seed from configs: keep configs themselves (convention-based) and any repo-local paths referenced.
    configs_root = PROJECT_ROOT / "configs"
    if configs_root.exists():
        for cfg in _iter_files(configs_root, exts={".yaml", ".yml", ".toml", ".json", ".jsonl"}):
            mark_keep(cfg, Reason("config_seed", _normalize_rel(cfg)))
            cfg_text = _read_text(cfg)
            for ref in _extract_repo_paths_from_text(cfg_text):
                ref_path = PROJECT_ROOT / ref
                if ref_path.exists():
                    mark_keep(ref_path, Reason("config_ref", f"{_normalize_rel(cfg)} -> {ref}"))
                    dep_rel = _normalize_rel(ref_path)
                    if dep_rel not in seen and ref_path.suffix.lower() in {".py", ".sh"}:
                        seen.add(dep_rel)
                        queue.append((ref_path, Reason("config_ref", f"{_normalize_rel(cfg)} -> {ref}")))

    while queue:
        cur, from_reason = queue.pop(0)
        if not cur.exists():
            continue
        rel_cur = _normalize_rel(cur)

        if cur.suffix.lower() == ".py":
            cur_text = _read_text(cur)
            for module, via in _parse_python_imports(cur):
                dep_path = _module_to_path(module)
                if dep_path is None:
                    continue
                dep_rel = _normalize_rel(dep_path)
                mark_keep(dep_path, Reason("py_import", f"{rel_cur} -> {via} {module}"))
                if dep_rel not in seen:
                    seen.add(dep_rel)
                    queue.append((dep_path, Reason("py_import", f"{rel_cur} -> {module}")))

            # Also scan for repo path strings inside python files (configs/prompts/scripts).
            for ref in _extract_repo_paths_from_text(cur_text):
                ref_path = (PROJECT_ROOT / ref).resolve()
                if not _is_within_repo(ref_path):
                    continue
                if ref_path.exists():
                    mark_keep(ref_path, Reason("text_path", f"{rel_cur} -> {ref}"))
                    dep_rel = _normalize_rel(ref_path)
                    if dep_rel not in seen and ref_path.suffix.lower() in {".py", ".sh"}:
                        seen.add(dep_rel)
                        queue.append((ref_path, Reason("text_path", f"{rel_cur} -> {ref}")))

            # Heuristic for constructed paths like scripts/generic/.../tool.py
            for ref in _extract_constructed_scripts_from_python(cur_text):
                ref_path = PROJECT_ROOT / ref
                if ref_path.exists():
                    mark_keep(ref_path, Reason("constructed_path", f"{rel_cur} -> {ref}"))
                    dep_rel = _normalize_rel(ref_path)
                    if dep_rel not in seen and ref_path.suffix.lower() in {".py", ".sh"}:
                        seen.add(dep_rel)
                        queue.append((ref_path, Reason("constructed_path", f"{rel_cur} -> {ref}")))

        elif cur.suffix.lower() in {".sh", ".bash"}:
            for ref in _parse_shell_references(cur):
                ref_path = PROJECT_ROOT / ref
                if ref_path.exists():
                    mark_keep(ref_path, Reason("shell_ref", f"{rel_cur} -> {ref}"))
                    dep_rel = _normalize_rel(ref_path)
                    if dep_rel not in seen and ref_path.suffix.lower() in {".py", ".sh"}:
                        seen.add(dep_rel)
                        queue.append((ref_path, Reason("shell_ref", f"{rel_cur} -> {ref}")))
        else:
            # Markdown/config/etc: scan for local paths that should be kept
            text = _read_text(cur)
            for ref in _extract_repo_paths_from_text(text):
                ref_path = PROJECT_ROOT / ref
                if ref_path.exists():
                    mark_keep(ref_path, Reason("text_path", f"{rel_cur} -> {ref}"))

    # Expand kept directories into all their files only when the directory itself is explicitly marked.
    # For scripts/anime_pipeline we will keep only reachable files, but we must keep package __init__.py chain.
    # We'll add parents for every kept file.
    keep_files = set(keep.keys())
    for rel in list(keep_files):
        p = PROJECT_ROOT / rel
        if p.is_file():
            for parent in p.parents:
                if parent == PROJECT_ROOT:
                    break
                init = parent / "__init__.py"
                if init.exists():
                    mark_keep(init, Reason("pkg_init", f"parent of {rel}"))

    # Candidate deletes: scripts/** and docs/** and prompts/** that are not in keep set.
    scopes = [
        ("scripts", {".py", ".sh", ".bash", ".md"}),
        ("docs", {".md"}),
        ("prompts", None),
    ]
    delete_candidates: list[str] = []
    for scope, exts in scopes:
        scope_root = PROJECT_ROOT / scope
        if not scope_root.exists():
            continue
        for p in _iter_files(scope_root, exts=exts):
            rel = _normalize_rel(p)
            if rel not in keep:
                delete_candidates.append(rel)

    report = {
        "project_root": str(PROJECT_ROOT),
        "entrypoints_existing": [_normalize_rel(p) for p in existing_entrypoints],
        "entrypoints_missing": missing_entrypoints,
        "keep": {k: [r.__dict__ for r in v] for k, v in sorted(keep.items())},
        "delete_candidates": sorted(delete_candidates),
    }

    out_dir = PROJECT_ROOT / "outputs" / "maintenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "minimal_keep_set_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
