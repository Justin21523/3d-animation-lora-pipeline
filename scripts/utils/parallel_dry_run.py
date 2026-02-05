#!/usr/bin/env python3
"""
Parallel dry-run testing for all project configurations.

CPU-ONLY TOOL - No GPU required, uses 32-thread parallel processing.

This tool tests configuration loading for all projects in parallel,
ensuring configs are syntactically correct and can be merged properly.

Usage:
    python scripts/utils/parallel_dry_run.py
    python scripts/utils/parallel_dry_run.py --project simpsons
    python scripts/utils/parallel_dry_run.py --output results.json

Author: Justin Lu
Date: 2025-12-02
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import argparse
import json
import time
import traceback

# Add parent directory to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# CPU-only imports
try:
    from anime_pipeline.config import (
        load_config,
        get_config,
        merge_configs,
        validate_no_deprecated_paths
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


@dataclass
class ConfigTestResult:
    """Result of testing a configuration."""
    config_name: str
    config_type: str
    load_success: bool
    merge_success: bool
    deprecated_warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


def test_config_loading(config_path: Path, config_type: str) -> ConfigTestResult:
    """
    Test loading a single configuration file.

    Args:
        config_path: Path to config file
        config_type: Type of config (global, project, character, stage)

    Returns:
        ConfigTestResult
    """
    config_name = config_path.stem
    result = ConfigTestResult(
        config_name=config_name,
        config_type=config_type,
        load_success=False,
        merge_success=False
    )

    start = time.time()

    try:
        # Test basic loading
        config = load_config(config_name, config_type=config_type, use_cache=False)
        result.load_success = True

        # Test for deprecated paths
        try:
            warnings = validate_no_deprecated_paths(config)
            result.deprecated_warnings = warnings
        except Exception as e:
            result.deprecated_warnings = [f"Validation error: {str(e)[:100]}"]

        # Test merging (if project config)
        if config_type == "project":
            try:
                merged = get_config(project=config_name)
                result.merge_success = True
            except Exception as e:
                result.errors.append(f"Merge error: {str(e)[:100]}")
        else:
            result.merge_success = True  # N/A for non-project configs

    except FileNotFoundError as e:
        result.errors.append(f"File not found: {str(e)[:100]}")
    except Exception as e:
        result.errors.append(f"Load error: {str(e)[:100]}")
        # Add traceback for debugging
        result.errors.append(traceback.format_exc()[:300])

    result.elapsed_ms = (time.time() - start) * 1000
    return result


def collect_config_files(config_root: Path) -> List[Tuple[Path, str]]:
    """
    Collect all configuration files with their types.

    Returns:
        List of (path, config_type) tuples
    """
    configs = []

    # Global configs
    global_dir = config_root / "global"
    if global_dir.exists():
        for f in global_dir.glob("*.yaml"):
            configs.append((f, "global"))

    # Project configs
    projects_dir = config_root / "projects"
    if projects_dir.exists():
        for f in projects_dir.glob("*.yaml"):
            if f.name != "_template.yaml":  # Skip template
                configs.append((f, "project"))

    # Character configs
    characters_dir = config_root / "characters"
    if characters_dir.exists():
        for f in characters_dir.glob("*.yaml"):
            configs.append((f, "character"))

    # Stage configs (including subdirectories)
    stages_dir = config_root / "stages"
    if stages_dir.exists():
        for f in stages_dir.rglob("*.yaml"):
            configs.append((f, "stage"))

    # Training configs
    training_dir = config_root / "training"
    if training_dir.exists():
        for f in training_dir.glob("*.yaml"):
            configs.append((f, "training"))

    # Animation configs
    animation_dir = config_root / "animation"
    if animation_dir.exists():
        for f in animation_dir.glob("*.yaml"):
            configs.append((f, "animation"))

    return configs


def run_parallel_tests(
    config_root: Path,
    num_threads: int = 32,
    verbose: bool = False,
    config_filter: Optional[str] = None
) -> Dict:
    """
    Run configuration tests in parallel.

    Args:
        config_root: Root directory for configurations
        num_threads: Number of threads
        verbose: Print detailed output
        config_filter: Only test configs matching this name

    Returns:
        Test summary dictionary
    """
    if not IMPORT_SUCCESS:
        return {
            "error": f"Failed to import anime_pipeline.config: {IMPORT_ERROR}",
            "total_tests": 0,
            "passed": 0,
            "failed": 0
        }

    # Collect configs
    configs = collect_config_files(config_root)

    # Filter if specified
    if config_filter:
        configs = [(p, t) for p, t in configs if config_filter in p.name]

    if not configs:
        return {
            "error": "No configuration files found",
            "total_tests": 0,
            "passed": 0,
            "failed": 0
        }

    print(f"Testing {len(configs)} configuration files with {num_threads} threads...")
    start_time = time.time()

    results: List[ConfigTestResult] = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(test_config_loading, p, t): (p, t) for p, t in configs}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if verbose:
                status = "✅" if result.load_success and result.merge_success else "❌"
                print(f"  {status} [{result.config_type}] {result.config_name} ({result.elapsed_ms:.1f}ms)")

    elapsed = time.time() - start_time

    # Compile summary
    passed = sum(1 for r in results if r.load_success and r.merge_success and not r.errors)
    failed = len(results) - passed
    with_warnings = sum(1 for r in results if r.deprecated_warnings)

    # Group by type
    by_type: Dict[str, Dict[str, int]] = {}
    for r in results:
        if r.config_type not in by_type:
            by_type[r.config_type] = {"total": 0, "passed": 0, "failed": 0}
        by_type[r.config_type]["total"] += 1
        if r.load_success and r.merge_success and not r.errors:
            by_type[r.config_type]["passed"] += 1
        else:
            by_type[r.config_type]["failed"] += 1

    summary = {
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "with_deprecated_warnings": with_warnings,
        "elapsed_seconds": round(elapsed, 2),
        "by_type": by_type,
        "results": [
            {
                "name": r.config_name,
                "type": r.config_type,
                "load_success": r.load_success,
                "merge_success": r.merge_success,
                "deprecated_warnings": r.deprecated_warnings,
                "errors": r.errors,
                "elapsed_ms": round(r.elapsed_ms, 2)
            }
            for r in sorted(results, key=lambda x: (x.config_type, x.config_name))
        ]
    }

    return summary


def print_summary(summary: Dict, verbose: bool = False) -> None:
    """Print test summary."""
    if "error" in summary:
        print(f"\n❌ Error: {summary['error']}")
        return

    print("\n" + "=" * 70)
    print("Parallel Configuration Dry-Run Results")
    print("=" * 70)

    print(f"\nTotal tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ✅")
    print(f"Failed: {summary['failed']} ❌")
    print(f"With deprecated warnings: {summary['with_deprecated_warnings']} ⚠️")
    print(f"Time elapsed: {summary['elapsed_seconds']}s")

    # By type breakdown
    print("\n" + "-" * 70)
    print("Results by config type:")
    print("-" * 70)
    print(f"{'Type':<20} {'Total':>10} {'Passed':>10} {'Failed':>10}")
    print("-" * 50)

    for config_type, stats in sorted(summary['by_type'].items()):
        print(f"{config_type:<20} {stats['total']:>10} {stats['passed']:>10} {stats['failed']:>10}")

    # Failed configs
    failed_results = [r for r in summary['results'] if not r['load_success'] or r['errors']]
    if failed_results:
        print("\n" + "-" * 70)
        print("Failed configurations:")
        print("-" * 70)

        for r in failed_results:
            print(f"\n❌ [{r['type']}] {r['name']}")
            for err in r['errors']:
                # Truncate long errors
                err_lines = err.split('\n')
                for line in err_lines[:3]:
                    print(f"   {line[:100]}")

    # Warnings
    warning_results = [r for r in summary['results'] if r['deprecated_warnings']]
    if warning_results and verbose:
        print("\n" + "-" * 70)
        print("Deprecated path warnings:")
        print("-" * 70)

        for r in warning_results:
            print(f"\n⚠️  [{r['type']}] {r['name']}")
            for w in r['deprecated_warnings'][:3]:
                print(f"   {w[:100]}")

    print("\n" + "=" * 70)
    if summary['failed'] == 0:
        print("✅ All configuration tests passed!")
    else:
        print(f"❌ {summary['failed']} configuration(s) failed")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel dry-run testing for configurations"
    )
    parser.add_argument(
        "--config-root",
        type=str,
        default=None,
        help="Root directory for configurations (default: auto-detect)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Number of threads for parallel processing (default: 32)"
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Only test configs matching this name"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save test results to JSON file"
    )

    args = parser.parse_args()

    # Determine config root
    if args.config_root:
        config_root = Path(args.config_root)
    else:
        config_root = project_root / "configs"

    if not config_root.exists():
        print(f"Error: Config directory not found: {config_root}")
        sys.exit(1)

    print(f"Config root: {config_root}")
    print(f"Using {args.threads} threads")

    # Run tests
    summary = run_parallel_tests(
        config_root,
        args.threads,
        args.verbose,
        args.filter
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

    # Exit with error code if failures
    sys.exit(0 if summary.get('failed', 1) == 0 else 1)


if __name__ == "__main__":
    main()
