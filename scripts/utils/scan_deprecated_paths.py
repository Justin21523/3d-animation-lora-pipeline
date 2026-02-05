#!/usr/bin/env python3
"""
Scan entire codebase for deprecated AI_DATA path references.

CPU-ONLY TOOL - No GPU required, uses 32-thread parallel processing.

This tool scans all Python and YAML files in the project for deprecated
path patterns that should be migrated to AI_WAREHOUSE 3.0 paths.

Usage:
    python scripts/utils/scan_deprecated_paths.py
    python scripts/utils/scan_deprecated_paths.py --output report.json
    python scripts/utils/scan_deprecated_paths.py --include-docs

Author: Justin Lu
Date: 2025-12-02
"""

import os
import sys
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, field
import argparse
import json
import time

# CPU-only - no torch/cuda imports


@dataclass
class Finding:
    """A single deprecated path finding."""
    file_path: str
    line_number: int
    line_content: str
    pattern_matched: str


@dataclass
class ScanResult:
    """Result of scanning a single file."""
    file_path: str
    findings: List[Finding] = field(default_factory=list)
    error: str = None


# Deprecated path patterns (regex)
DEPRECATED_PATTERNS = [
    r"/mnt/data/ai_data",
    r"~/ai_data",
    r"\$HOME/datasets",
    r"\$HOME/ai_data",
    r"/home/\w+/ai_data",
    r"ai_data/datasets",
    r"ai_data/training_data",
    r"ai_data/models",
]

# Compiled regex patterns
COMPILED_PATTERNS = [(p, re.compile(p)) for p in DEPRECATED_PATTERNS]

# File extensions to scan
SCAN_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".sh",
    ".bash",
}

# Documentation extensions (optional)
DOC_EXTENSIONS = {
    ".md",
    ".rst",
    ".txt",
}

# Directories to skip
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    "*.egg-info",
}


def should_skip_directory(dir_name: str) -> bool:
    """Check if directory should be skipped."""
    return dir_name in SKIP_DIRS or dir_name.startswith(".")


def scan_file(file_path: Path) -> ScanResult:
    """
    Scan a single file for deprecated patterns.

    Returns:
        ScanResult with any findings
    """
    findings = []

    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            for pattern_str, pattern in COMPILED_PATTERNS:
                if pattern.search(line):
                    # Skip if it's a comment about deprecated paths (documentation)
                    stripped = line.strip()
                    if stripped.startswith('#') and 'deprecated' in stripped.lower():
                        continue

                    findings.append(Finding(
                        file_path=str(file_path),
                        line_number=line_num,
                        line_content=line.strip()[:150],
                        pattern_matched=pattern_str
                    ))
                    break  # One finding per line is enough

    except Exception as e:
        return ScanResult(
            file_path=str(file_path),
            error=str(e)[:100]
        )

    return ScanResult(
        file_path=str(file_path),
        findings=findings
    )


def collect_files(
    project_root: Path,
    include_docs: bool = False,
    extra_extensions: Set[str] = None
) -> List[Path]:
    """
    Collect all files to scan.

    Args:
        project_root: Root directory to scan
        include_docs: Whether to include documentation files
        extra_extensions: Additional file extensions to scan

    Returns:
        List of file paths to scan
    """
    extensions = SCAN_EXTENSIONS.copy()
    if include_docs:
        extensions.update(DOC_EXTENSIONS)
    if extra_extensions:
        extensions.update(extra_extensions)

    files = []

    for root, dirs, filenames in os.walk(project_root):
        # Filter out directories to skip
        dirs[:] = [d for d in dirs if not should_skip_directory(d)]

        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext in extensions:
                files.append(Path(root) / filename)

    return files


def scan_all_files(
    project_root: Path,
    num_threads: int = 32,
    include_docs: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Scan all files in project for deprecated paths.

    Args:
        project_root: Root directory to scan
        num_threads: Number of threads for parallel processing
        include_docs: Include documentation files
        verbose: Print progress

    Returns:
        Scan summary dictionary
    """
    # Collect files
    files = collect_files(project_root, include_docs)

    print(f"Scanning {len(files)} files with {num_threads} threads...")
    start_time = time.time()

    all_findings: List[Finding] = []
    files_with_issues: Set[str] = set()
    errors: List[str] = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(scan_file, f): f for f in files}

        for future in as_completed(futures):
            result = future.result()

            if result.error:
                errors.append(f"{result.file_path}: {result.error}")
            elif result.findings:
                all_findings.extend(result.findings)
                files_with_issues.add(result.file_path)

                if verbose:
                    print(f"  ⚠️  {Path(result.file_path).name}: {len(result.findings)} finding(s)")

    elapsed = time.time() - start_time

    # Group findings by pattern
    by_pattern: Dict[str, int] = {}
    for f in all_findings:
        by_pattern[f.pattern_matched] = by_pattern.get(f.pattern_matched, 0) + 1

    # Group findings by file
    by_file: Dict[str, List[Dict]] = {}
    for f in all_findings:
        if f.file_path not in by_file:
            by_file[f.file_path] = []
        by_file[f.file_path].append({
            "line": f.line_number,
            "content": f.line_content,
            "pattern": f.pattern_matched
        })

    summary = {
        "total_files_scanned": len(files),
        "files_with_deprecated_paths": len(files_with_issues),
        "total_findings": len(all_findings),
        "findings_by_pattern": by_pattern,
        "scan_errors": len(errors),
        "elapsed_seconds": round(elapsed, 2),
        "findings_by_file": by_file,
        "errors": errors[:10]  # Limit errors in output
    }

    return summary


def print_summary(summary: Dict, verbose: bool = False) -> None:
    """Print scan summary."""
    print("\n" + "=" * 70)
    print("Deprecated Path Scan Results")
    print("=" * 70)
    print(f"\nTotal files scanned: {summary['total_files_scanned']}")
    print(f"Files with deprecated paths: {summary['files_with_deprecated_paths']}")
    print(f"Total findings: {summary['total_findings']}")
    print(f"Scan errors: {summary['scan_errors']}")
    print(f"Time elapsed: {summary['elapsed_seconds']}s")

    if summary['findings_by_pattern']:
        print("\n" + "-" * 70)
        print("Findings by pattern:")
        print("-" * 70)
        for pattern, count in sorted(summary['findings_by_pattern'].items(),
                                     key=lambda x: x[1], reverse=True):
            print(f"  {count:4d}  {pattern}")

    if summary['total_findings'] > 0:
        print("\n" + "-" * 70)
        print("Files with deprecated paths:")
        print("-" * 70)

        for file_path, findings in sorted(summary['findings_by_file'].items()):
            # Make path relative for readability
            try:
                rel_path = Path(file_path).relative_to(Path.cwd())
            except ValueError:
                rel_path = file_path

            print(f"\n❌ {rel_path}")
            for f in findings[:5]:  # Limit to 5 per file
                print(f"   Line {f['line']}: {f['content'][:80]}")
            if len(findings) > 5:
                print(f"   ... and {len(findings) - 5} more")

    print("\n" + "=" * 70)

    if summary['total_findings'] == 0:
        print("✅ No deprecated paths found!")
    else:
        print(f"❌ Found {summary['total_findings']} deprecated path reference(s)")
        print("\nTo fix these issues:")
        print("  1. Replace /mnt/data/ai_data/ with AI_WAREHOUSE 3.0 paths")
        print("  2. Use config variables: ${roots.datasets}, ${roots.models}")
        print("  3. See docs/setup/AI_WAREHOUSE_3.0.md for migration guide")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Scan codebase for deprecated AI_DATA paths"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Root directory to scan (default: auto-detect)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Number of threads for parallel processing (default: 32)"
    )
    parser.add_argument(
        "--include-docs",
        action="store_true",
        help="Include documentation files (.md, .rst, .txt)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save scan report to JSON file"
    )

    args = parser.parse_args()

    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        # Auto-detect from script location
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent

    if not project_root.exists():
        print(f"Error: Project directory not found: {project_root}")
        sys.exit(1)

    print(f"Project root: {project_root}")
    print(f"Using {args.threads} threads")
    if args.include_docs:
        print("Including documentation files")

    # Run scan
    summary = scan_all_files(
        project_root,
        args.threads,
        args.include_docs,
        args.verbose
    )

    # Print summary
    print_summary(summary, args.verbose)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nReport saved to: {output_path}")

    # Exit with error code if findings
    sys.exit(0 if summary['total_findings'] == 0 else 1)


if __name__ == "__main__":
    main()
