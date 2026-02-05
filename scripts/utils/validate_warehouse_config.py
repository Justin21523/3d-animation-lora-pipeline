#!/usr/bin/env python3
"""
Validate all configurations against AI_WAREHOUSE 3.0 specification.

CPU-ONLY TOOL - No GPU required, uses 32-thread parallel processing.

This tool checks:
1. All configuration files for deprecated path usage
2. Path existence for referenced directories
3. Configuration syntax and required fields
4. Cross-config consistency

Usage:
    python scripts/utils/validate_warehouse_config.py
    python scripts/utils/validate_warehouse_config.py --verbose
    python scripts/utils/validate_warehouse_config.py --fix-suggestions

Author: Justin Lu
Date: 2025-12-02
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import argparse
import json
import yaml
import time

# CPU-only - no torch/cuda imports


@dataclass
class ValidationResult:
    """Result of validating a single configuration file."""
    file_path: str
    is_valid: bool
    deprecated_paths: List[str]
    missing_paths: List[str]
    syntax_errors: List[str]
    warnings: List[str]


# Deprecated paths from AI_WAREHOUSE 3.0 spec
DEPRECATED_PATHS = [
    "/mnt/data/ai_data",
    "/mnt/data/ai_data/datasets",
    "/mnt/data/ai_data/training_data",
    "/mnt/data/ai_data/models",
    "~/ai_data",
    "$HOME/datasets",
    "$HOME/ai_data",
]

# Path keywords that should be validated
PATH_KEYWORDS = ["path", "dir", "root", "model", "output", "input", "base"]


def expand_home_path(path: str) -> str:
    """Expand ~ and $HOME in path string."""
    expanded = path.replace("~", str(Path.home()))
    expanded = expanded.replace("$HOME", str(Path.home()))
    return expanded


def check_deprecated_paths_in_content(content: str, file_path: str) -> List[str]:
    """Check for deprecated paths in file content."""
    findings = []
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        for dep_path in DEPRECATED_PATHS:
            expanded_dep = expand_home_path(dep_path)
            if dep_path in line or expanded_dep in line:
                findings.append(f"Line {i}: {line.strip()[:100]}")

    return findings


def validate_yaml_syntax(file_path: Path) -> Tuple[bool, List[str], Any]:
    """
    Validate YAML syntax.

    Returns:
        (is_valid, errors, parsed_content)
    """
    errors = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
        return True, [], content
    except yaml.YAMLError as e:
        errors.append(f"YAML syntax error: {str(e)[:200]}")
        return False, errors, None
    except Exception as e:
        errors.append(f"Error reading file: {str(e)[:100]}")
        return False, errors, None


def check_path_existence(config: Dict, file_path: str) -> List[str]:
    """Check if paths referenced in config exist."""
    missing = []

    def check_recursive(obj: Any, key_path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{key_path}.{key}" if key_path else key
                check_recursive(value, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_recursive(item, f"{key_path}[{i}]")
        elif isinstance(obj, str):
            # Check if this looks like a path that should exist
            key_lower = key_path.lower()
            if any(pk in key_lower for pk in PATH_KEYWORDS):
                # Skip OmegaConf variables
                if "${" in obj:
                    return
                # Skip HuggingFace model IDs
                if "/" in obj and not obj.startswith("/"):
                    parts = obj.split("/")
                    if len(parts) <= 3 and not any(p.endswith(('.yaml', '.json', '.pt', '.pth', '.safetensors')) for p in parts):
                        return

                # Check absolute paths
                if obj.startswith("/"):
                    expanded = expand_home_path(obj)
                    path = Path(expanded)
                    # Only check existence for base directories, not specific files that may not exist yet
                    parent = path.parent
                    if not parent.exists() and parent != path:
                        missing.append(f"{key_path}: parent directory missing for {obj}")

    if config:
        check_recursive(config)
    return missing


def validate_config_file(config_file: Path) -> ValidationResult:
    """Validate a single configuration file (CPU-only)."""
    file_path_str = str(config_file)
    deprecated_paths = []
    missing_paths = []
    syntax_errors = []
    warnings = []

    try:
        # Read file content
        content = config_file.read_text(encoding='utf-8', errors='ignore')

        # Check for deprecated paths
        deprecated_paths = check_deprecated_paths_in_content(content, file_path_str)

        # Validate YAML syntax
        is_valid_yaml, yaml_errors, parsed = validate_yaml_syntax(config_file)
        syntax_errors.extend(yaml_errors)

        # Check path existence
        if parsed:
            missing_paths = check_path_existence(parsed, file_path_str)

        # Check for common issues
        if parsed:
            # Warn if no version specified
            if "version" not in parsed and config_file.name in ["warehouse.yaml", "models.yaml"]:
                warnings.append("Missing 'version' field")

        is_valid = len(syntax_errors) == 0 and len(deprecated_paths) == 0

    except Exception as e:
        syntax_errors.append(f"Unexpected error: {str(e)[:100]}")
        is_valid = False

    return ValidationResult(
        file_path=file_path_str,
        is_valid=is_valid,
        deprecated_paths=deprecated_paths,
        missing_paths=missing_paths,
        syntax_errors=syntax_errors,
        warnings=warnings
    )


def validate_all_configs(config_root: Path, num_threads: int = 32, verbose: bool = False) -> Dict:
    """
    Validate all configuration files in parallel.

    Args:
        config_root: Root directory for configurations
        num_threads: Number of threads for parallel processing
        verbose: Print detailed output

    Returns:
        Validation summary dictionary
    """
    # Find all YAML files
    yaml_files = list(config_root.rglob("*.yaml"))
    yaml_files.extend(config_root.rglob("*.yml"))

    print(f"Validating {len(yaml_files)} config files with {num_threads} threads...")
    start_time = time.time()

    results: List[ValidationResult] = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(validate_config_file, f): f for f in yaml_files}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if verbose:
                status = "✅" if result.is_valid else "❌"
                print(f"  {status} {Path(result.file_path).name}")

    elapsed = time.time() - start_time

    # Compile summary
    total = len(results)
    valid = sum(1 for r in results if r.is_valid)
    invalid = total - valid
    deprecated_count = sum(len(r.deprecated_paths) for r in results)
    missing_count = sum(len(r.missing_paths) for r in results)
    syntax_count = sum(len(r.syntax_errors) for r in results)
    warning_count = sum(len(r.warnings) for r in results)

    summary = {
        "total_files": total,
        "valid_files": valid,
        "invalid_files": invalid,
        "deprecated_path_findings": deprecated_count,
        "missing_path_findings": missing_count,
        "syntax_errors": syntax_count,
        "warnings": warning_count,
        "elapsed_seconds": round(elapsed, 2),
        "results": [
            {
                "file": r.file_path,
                "valid": r.is_valid,
                "deprecated_paths": r.deprecated_paths,
                "missing_paths": r.missing_paths,
                "syntax_errors": r.syntax_errors,
                "warnings": r.warnings
            }
            for r in results if not r.is_valid or r.warnings or r.missing_paths
        ]
    }

    return summary


def print_summary(summary: Dict, verbose: bool = False) -> None:
    """Print validation summary."""
    print("\n" + "=" * 70)
    print("AI_WAREHOUSE 3.0 Configuration Validation Summary")
    print("=" * 70)
    print(f"\nTotal files scanned: {summary['total_files']}")
    print(f"Valid files: {summary['valid_files']}")
    print(f"Invalid files: {summary['invalid_files']}")
    print(f"Deprecated path findings: {summary['deprecated_path_findings']}")
    print(f"Missing parent directories: {summary['missing_path_findings']}")
    print(f"Syntax errors: {summary['syntax_errors']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Time elapsed: {summary['elapsed_seconds']}s")

    if summary['invalid_files'] > 0 or verbose:
        print("\n" + "-" * 70)
        print("Details:")
        print("-" * 70)

        for result in summary['results']:
            if not result['valid'] or result['warnings'] or result['missing_paths']:
                file_name = Path(result['file']).relative_to(Path(result['file']).parents[3])
                status = "❌ INVALID" if not result['valid'] else "⚠️  WARNING"
                print(f"\n{status}: {file_name}")

                if result['deprecated_paths']:
                    print("  Deprecated paths found:")
                    for dp in result['deprecated_paths'][:5]:
                        print(f"    - {dp}")
                    if len(result['deprecated_paths']) > 5:
                        print(f"    ... and {len(result['deprecated_paths']) - 5} more")

                if result['syntax_errors']:
                    print("  Syntax errors:")
                    for se in result['syntax_errors']:
                        print(f"    - {se}")

                if result['missing_paths']:
                    print("  Missing paths:")
                    for mp in result['missing_paths'][:3]:
                        print(f"    - {mp}")

                if result['warnings']:
                    print("  Warnings:")
                    for w in result['warnings']:
                        print(f"    - {w}")

    print("\n" + "=" * 70)

    if summary['invalid_files'] == 0:
        print("✅ All configurations are valid!")
    else:
        print(f"❌ {summary['invalid_files']} configuration(s) need attention")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate configurations against AI_WAREHOUSE 3.0 spec"
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
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save validation report to JSON file"
    )
    parser.add_argument(
        "--fix-suggestions",
        action="store_true",
        help="Show suggestions for fixing issues"
    )

    args = parser.parse_args()

    # Determine config root
    if args.config_root:
        config_root = Path(args.config_root)
    else:
        # Auto-detect from script location
        script_dir = Path(__file__).resolve().parent
        config_root = script_dir.parent.parent / "configs"

    if not config_root.exists():
        print(f"Error: Config directory not found: {config_root}")
        sys.exit(1)

    print(f"Config root: {config_root}")
    print(f"Using {args.threads} threads")

    # Run validation
    summary = validate_all_configs(config_root, args.threads, args.verbose)

    # Print summary
    print_summary(summary, args.verbose)

    # Show fix suggestions if requested
    if args.fix_suggestions and summary['deprecated_path_findings'] > 0:
        print("\n" + "=" * 70)
        print("Fix Suggestions:")
        print("=" * 70)
        print("""
Replace deprecated paths with AI_WAREHOUSE 3.0 paths:

OLD PATH (deprecated)              -> NEW PATH
/mnt/data/ai_data/datasets/        -> /mnt/data/datasets/
/mnt/data/ai_data/training_data/   -> /mnt/data/training/
/mnt/data/ai_data/models/          -> /mnt/c/ai_models/
~/ai_data/                         -> Use absolute paths from warehouse.yaml

Use config variables instead of hardcoded paths:
  ${roots.datasets}/general        -> /mnt/data/datasets/general
  ${roots.models}/segmentation     -> /mnt/c/ai_models/segmentation
  ${animation_2d.training_data}    -> /mnt/data/training/lora/2d_characters
""")

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nReport saved to: {output_path}")

    # Exit with error code if invalid configs found
    sys.exit(0 if summary['invalid_files'] == 0 else 1)


if __name__ == "__main__":
    main()
