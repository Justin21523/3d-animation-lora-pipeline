#!/usr/bin/env python3
"""
Path Utilities for Pipeline Scripts

Provides consistent path handling and directory operations across all pipeline components.

Features:
- Safe directory creation with permission handling
- Path validation and normalization
- Relative-to-absolute path conversion
- Common path queries and operations

Usage:
    from scripts.core.utils.path_utils import ensure_dir, validate_path, get_project_root

    # Ensure directory exists
    ensure_dir("outputs/results")

    # Validate paths
    if validate_path("/path/to/file"):
        process_file()

    # Get project root
    root = get_project_root()

Author: AI Pipeline
Date: 2025-01-20
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional, List


def ensure_dir(
    path: Union[str, Path],
    mode: int = 0o755,
    parents: bool = True,
    exist_ok: bool = True
) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Thread-safe directory creation with proper permissions.

    Args:
        path: Directory path to create
        mode: Permission mode (default: 0o755 = rwxr-xr-x)
        parents: Create parent directories if needed (default: True)
        exist_ok: Don't raise error if directory already exists (default: True)

    Returns:
        Path object for the created/existing directory

    Raises:
        PermissionError: If insufficient permissions to create directory
        FileExistsError: If path exists and is a file (not directory)

    Examples:
        # Basic usage
        output_dir = ensure_dir("outputs/results")

        # Create nested directories
        data_dir = ensure_dir("data/processed/train/images")

        # Custom permissions (read-write-execute for owner only)
        secure_dir = ensure_dir("secrets/", mode=0o700)
    """
    path = Path(path)

    # Check if path exists and is a file
    if path.exists() and not path.is_dir():
        raise FileExistsError(
            f"Path exists but is not a directory: {path}"
        )

    # Create directory with specified permissions
    try:
        path.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)
    except PermissionError as e:
        raise PermissionError(
            f"Insufficient permissions to create directory: {path}"
        ) from e

    return path


def validate_path(
    path: Union[str, Path],
    must_exist: bool = False,
    file_ok: bool = True,
    dir_ok: bool = True,
    readable: bool = True,
    writable: bool = False
) -> bool:
    """
    Validate path against specified criteria.

    Args:
        path: Path to validate
        must_exist: Path must exist (default: False)
        file_ok: Files are acceptable (default: True)
        dir_ok: Directories are acceptable (default: True)
        readable: Path must be readable (default: True)
        writable: Path must be writable (default: False)

    Returns:
        True if path is valid, False otherwise

    Examples:
        # Check if input file exists and is readable
        if validate_path("input.txt", must_exist=True, dir_ok=False):
            process_file("input.txt")

        # Check if output directory is writable
        if validate_path("outputs/", must_exist=True, file_ok=False, writable=True):
            save_results()
    """
    path = Path(path)

    # Check existence
    if must_exist and not path.exists():
        return False

    if path.exists():
        # Check type
        if path.is_file() and not file_ok:
            return False
        if path.is_dir() and not dir_ok:
            return False

        # Check permissions
        if readable and not os.access(path, os.R_OK):
            return False
        if writable and not os.access(path, os.W_OK):
            return False

    return True


def get_project_root() -> Path:
    """
    Get project root directory.

    Searches upward from current file location to find project root
    (directory containing .git folder or specific marker files).

    Returns:
        Path to project root directory

    Raises:
        FileNotFoundError: If project root cannot be determined

    Example:
        root = get_project_root()
        config_path = root / "configs" / "global_config.yaml"
    """
    # Start from this file's location
    current = Path(__file__).resolve()

    # Search upward for project markers
    for parent in [current, *current.parents]:
        # Check for .git directory (git repository root)
        if (parent / ".git").exists():
            return parent

        # Check for common project marker files
        markers = ["setup.py", "pyproject.toml", "README.md"]
        for marker in markers:
            if (parent / marker).exists():
                return parent

    # If no marker found, use 3 levels up from scripts/core/utils/
    # (assuming standard project structure)
    return current.parents[2]


def make_absolute(
    path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    Convert path to absolute path.

    If path is already absolute, return as-is.
    If path is relative, make absolute relative to base_dir (or cwd).

    Args:
        path: Path to convert
        base_dir: Base directory for relative paths (default: current working directory)

    Returns:
        Absolute Path object

    Examples:
        # Convert relative to absolute (using cwd)
        abs_path = make_absolute("data/train.txt")

        # Convert relative to absolute (using custom base)
        abs_path = make_absolute("images/", base_dir="/mnt/data/datasets")
    """
    path = Path(path)

    if path.is_absolute():
        return path

    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    return (base_dir / path).resolve()


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False,
    files_only: bool = True
) -> List[Path]:
    """
    List files in directory matching pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern (default: "*" matches all)
        recursive: Search subdirectories recursively (default: False)
        files_only: Return only files, not directories (default: True)

    Returns:
        List of Path objects matching criteria

    Examples:
        # List all images in directory
        images = list_files("data/images", pattern="*.jpg")

        # List all Python files recursively
        py_files = list_files("scripts/", pattern="*.py", recursive=True)

        # List all items (including directories)
        all_items = list_files("outputs/", files_only=False)
    """
    directory = Path(directory)

    if not directory.exists():
        return []

    if recursive:
        matches = directory.rglob(pattern)
    else:
        matches = directory.glob(pattern)

    if files_only:
        return sorted([p for p in matches if p.is_file()])
    else:
        return sorted(list(matches))


def get_size_mb(path: Union[str, Path]) -> float:
    """
    Get file or directory size in megabytes.

    For directories, recursively calculates total size of all files.

    Args:
        path: File or directory path

    Returns:
        Size in megabytes (MB)

    Examples:
        # Get file size
        size = get_size_mb("model.safetensors")
        print(f"Model size: {size:.2f} MB")

        # Get directory size
        total_size = get_size_mb("outputs/lora_checkpoints/")
        print(f"Total checkpoints: {total_size:.2f} MB")
    """
    path = Path(path)

    if not path.exists():
        return 0.0

    if path.is_file():
        return path.stat().st_size / (1024 * 1024)

    # Directory: sum all file sizes recursively
    total_size = sum(
        f.stat().st_size
        for f in path.rglob("*")
        if f.is_file()
    )

    return total_size / (1024 * 1024)


def safe_filename(name: str, replacement: str = "_") -> str:
    """
    Convert string to safe filename by removing/replacing invalid characters.

    Args:
        name: Original filename or string
        replacement: Character to replace invalid chars with (default: "_")

    Returns:
        Safe filename string

    Examples:
        # Sanitize user input for filename
        safe_name = safe_filename("My Image: Test (v2).jpg")
        # Returns: "My_Image_Test_v2_.jpg"

        # Remove invalid characters
        safe_name = safe_filename("file/with\\slashes", replacement="-")
        # Returns: "file-with-slashes"
    """
    # Characters not allowed in filenames on most systems
    invalid_chars = '<>:"/\\|?*'

    # Replace invalid characters
    for char in invalid_chars:
        name = name.replace(char, replacement)

    # Remove leading/trailing whitespace and dots
    name = name.strip(). strip('.')

    # Ensure not empty
    if not name:
        name = "unnamed"

    return name


def rel_to_abs_paths(config: dict, base_dir: Union[str, Path]) -> dict:
    """
    Recursively convert relative paths in config dict to absolute paths.

    Useful for loading YAML/JSON configs with relative paths and converting them
    to absolute paths relative to config file location.

    Args:
        config: Configuration dictionary (may contain nested dicts)
        base_dir: Base directory for relative paths

    Returns:
        Modified config dict with absolute paths

    Examples:
        # Load config and convert paths
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        config = rel_to_abs_paths(config, base_dir=Path("config.yaml").parent)
    """
    base_dir = Path(base_dir)

    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively process nested dicts
            config[key] = rel_to_abs_paths(value, base_dir)
        elif isinstance(value, str):
            # Check if value looks like a path
            if "/" in value or "\\" in value:
                path = Path(value)
                if not path.is_absolute():
                    config[key] = str(make_absolute(path, base_dir))

    return config


def find_files_by_ext(
    directory: Union[str, Path],
    extensions: Union[str, List[str]],
    recursive: bool = True
) -> List[Path]:
    """
    Find all files with specified extension(s).

    Args:
        directory: Directory to search
        extensions: File extension(s) (with or without leading dot)
        recursive: Search subdirectories recursively (default: True)

    Returns:
        List of matching file paths

    Examples:
        # Find all images
        images = find_files_by_ext("data/", extensions=[".jpg", ".png", ".jpeg"])

        # Find Python scripts
        scripts = find_files_by_ext("scripts/", extensions="py")

        # Non-recursive search
        local_images = find_files_by_ext("./", extensions=".jpg", recursive=False)
    """
    directory = Path(directory)

    if isinstance(extensions, str):
        extensions = [extensions]

    # Normalize extensions (ensure leading dot)
    extensions = [
        ext if ext.startswith('.') else f'.{ext}'
        for ext in extensions
    ]

    # Find matching files
    matching_files = []

    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    for path in directory.glob(pattern):
        if path.is_file() and path.suffix.lower() in extensions:
            matching_files.append(path)

    return sorted(matching_files)


# Convenience functions for common path operations

def is_image_file(path: Union[str, Path]) -> bool:
    """Check if path is an image file (jpg, jpeg, png, webp, bmp)."""
    image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    return Path(path).suffix.lower() in image_exts


def is_video_file(path: Union[str, Path]) -> bool:
    """Check if path is a video file (mp4, avi, mov, mkv, webm)."""
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    return Path(path).suffix.lower() in video_exts


def is_json_file(path: Union[str, Path]) -> bool:
    """Check if path is a JSON file."""
    return Path(path).suffix.lower() == '.json'


# Example usage
if __name__ == "__main__":
    # Demo directory operations
    print("=== Path Utilities Demo ===\n")

    # 1. Ensure directory exists
    print("1. Create directory:")
    test_dir = ensure_dir("temp/test_outputs/demo")
    print(f"   Created: {test_dir}")

    # 2. Validate paths
    print("\n2. Validate paths:")
    print(f"   Test dir exists: {validate_path(test_dir, must_exist=True)}")
    print(f"   Test dir writable: {validate_path(test_dir, writable=True)}")

    # 3. Get project root
    print("\n3. Project root:")
    root = get_project_root()
    print(f"   Root: {root}")

    # 4. Make absolute paths
    print("\n4. Absolute paths:")
    abs_path = make_absolute("data/train.txt", base_dir=root)
    print(f"   Relative: data/train.txt")
    print(f"   Absolute: {abs_path}")

    # 5. Safe filenames
    print("\n5. Safe filenames:")
    unsafe = "My File: Test (v2) <draft>.txt"
    safe = safe_filename(unsafe)
    print(f"   Unsafe: {unsafe}")
    print(f"   Safe:   {safe}")

    # 6. List files
    print("\n6. List Python files:")
    py_files = list_files(root / "scripts", pattern="*.py", recursive=False)
    print(f"   Found {len(py_files)} Python files")
    for f in py_files[:3]:
        print(f"   - {f.name}")

    # Clean up test directory
    import shutil
    if (Path("temp") / "test_outputs").exists():
        shutil.rmtree("temp/test_outputs")
        print("\n[Demo cleanup complete]")
