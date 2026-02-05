#!/usr/bin/env python3
"""
Unified Configuration Loader for 2D Animation LoRA Pipeline

Provides centralized configuration management using OmegaConf for hierarchical
YAML configuration files. Supports:
- Global pipeline settings
- Stage-specific configurations
- Character definitions
- Project overrides
- Animation-specific configs (AnimateDiff, ControlNet)
- Automatic path resolution
- Configuration merging and validation
- 2D/3D parameter conversion

Usage:
    from anime_pipeline.config.omega_loader import load_config, get_config

    # Load global pipeline config
    config = load_config("pipeline")

    # Load stage-specific config
    seg_config = load_config("toonout", config_type="stage")

    # Load character config
    char_config = load_config("homer", config_type="character")

    # Load project config with overrides
    project_config = load_config("simpsons", config_type="project")

    # Get merged configuration for a project
    full_config = get_config(project="simpsons", character="homer")

Author: Adapted from 3D pipeline by Justin Lu
Date: 2025-01-XX
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

from omegaconf import OmegaConf, DictConfig


# Default paths - ADAPTED FOR 2D PIPELINE STRUCTURE
REPO_ROOT = Path(__file__).resolve().parents[2]  # 2 levels up from anime_pipeline/config/
CONFIG_ROOT = REPO_ROOT / "configs"

# Configuration directory structure - ADDED animation directory for 2D-specific configs
CONFIG_DIRS = {
    "global": CONFIG_ROOT / "global",
    "stage": CONFIG_ROOT / "stages",
    "character": CONFIG_ROOT / "characters",
    "project": CONFIG_ROOT / "projects",
    "training": CONFIG_ROOT / "training",
    "animation": CONFIG_ROOT / "animation",  # AnimateDiff, ControlNet configs
}

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Unified configuration loader with hierarchical merging and path resolution.
    """

    def __init__(self, config_root: Optional[Path] = None):
        """
        Initialize configuration loader.

        Args:
            config_root: Root directory for configurations (default: repo/configs/)
        """
        self.config_root = Path(config_root) if config_root else CONFIG_ROOT
        self.config_dirs = {
            key: self.config_root / subdir.relative_to(CONFIG_ROOT)
            for key, subdir in CONFIG_DIRS.items()
        }
        self._cache: Dict[str, DictConfig] = {}

    def load(self,
             name: str,
             config_type: str = "global",
             use_cache: bool = True,
             resolve_variables: bool = False,
             resolve_env: bool = False) -> DictConfig:
        """
        Load configuration file by name and type.

        Args:
            name: Configuration name (without .yaml extension)
            config_type: Type of config - "global", "stage", "character", "project", "training", "animation"
            use_cache: Whether to use cached config if available
            resolve_variables: Whether to resolve OmegaConf variables (default: False for merging later)

        Returns:
            OmegaConf DictConfig object

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If config_type is invalid
        """
        if config_type not in self.config_dirs:
            raise ValueError(f"Invalid config_type: {config_type}. "
                           f"Must be one of: {list(self.config_dirs.keys())}")

        # Allow loading arbitrary YAML files by path (useful for tests and one-off configs)
        name_path = Path(name)
        if name_path.exists() and name_path.is_file():
            cache_key = f"path:{name_path.resolve()}:{resolve_variables}:{resolve_env}"

            if use_cache and cache_key in self._cache:
                logger.debug(f"Using cached config: {cache_key}")
                return self._cache[cache_key]

            logger.info(f"Loading config: {name_path}")
            config = OmegaConf.load(name_path)

            if resolve_env:
                config = self._expand_env_placeholders(config)

            if resolve_variables:
                try:
                    config = self._resolve_paths(config)
                except Exception as e:
                    # Allow configs with external placeholders (e.g., `${SOME_ENV}`) to load
                    # without hard-failing resolution. Prefer `${oc.env:VAR}` for true env expansion.
                    logger.debug(f"Skipping OmegaConf resolution for {name_path}: {e}")

            if use_cache:
                self._cache[cache_key] = config

            return config

        cache_key = f"{config_type}:{name}:{resolve_variables}:{resolve_env}"

        # Return cached config if available
        if use_cache and cache_key in self._cache:
            logger.debug(f"Using cached config: {cache_key}")
            return self._cache[cache_key]

        # Find config file
        config_dir = self.config_dirs[config_type]

        # For stage configs, check subdirectories
        if config_type == "stage":
            config_file = self._find_stage_config(config_dir, name)
        else:
            config_file = config_dir / f"{name}.yaml"

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}\n"
                f"Looking in: {config_dir}"
            )

        # Load and parse YAML
        logger.info(f"Loading config: {config_file}")
        config = OmegaConf.load(config_file)

        if resolve_env:
            config = self._expand_env_placeholders(config)

        # Optionally resolve variables and paths (only if requested)
        if resolve_variables:
            config = self._resolve_paths(config)

        # Cache config
        if use_cache:
            self._cache[cache_key] = config

        return config

    def _expand_env_placeholders(self, config: DictConfig) -> DictConfig:
        """
        Expand `${ENV_VAR}` placeholders using the current environment.

        - Only expands ALL-CAPS keys (A-Z0-9_) to avoid clobbering normal OmegaConf
          interpolations like `${model_root}`.
        - If an env var is missing, the placeholder is escaped as `\\${ENV_VAR}` so it
          becomes a literal `${ENV_VAR}` string (and doesn't raise on access).
        """
        import os
        import re

        pattern = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")

        def replace_in_str(value: str) -> str:
            def repl(match: re.Match) -> str:
                var = match.group(1)
                env_val = os.getenv(var)
                if env_val is None:
                    return r"\\${" + var + "}"
                return env_val

            return pattern.sub(repl, value)

        def walk(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: walk(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [walk(v) for v in obj]
            if isinstance(obj, str):
                return replace_in_str(obj)
            return obj

        container = OmegaConf.to_container(config, resolve=False)
        processed = walk(container)
        return OmegaConf.create(processed)

    def _find_stage_config(self, stage_dir: Path, name: str) -> Path:
        """
        Find stage configuration file in subdirectories.

        Args:
            stage_dir: Stage configurations directory
            name: Configuration name

        Returns:
            Path to configuration file
        """
        # Check direct file
        direct_file = stage_dir / f"{name}.yaml"
        if direct_file.exists():
            return direct_file

        # Check subdirectories (e.g., stages/segmentation/toonout.yaml)
        for subdir in stage_dir.iterdir():
            if subdir.is_dir():
                config_file = subdir / f"{name}.yaml"
                if config_file.exists():
                    return config_file

        # Return direct file path for error message
        return direct_file

    def _resolve_paths(self, config: DictConfig) -> DictConfig:
        """
        Resolve relative paths to absolute paths.

        Args:
            config: Configuration object

        Returns:
            Configuration with resolved paths
        """
        # Resolve OmegaConf variables first
        OmegaConf.resolve(config)

        # Convert to container for easier manipulation
        config_dict = OmegaConf.to_container(config, resolve=True)

        # Recursively resolve path-like strings
        resolved = self._resolve_paths_recursive(config_dict)

        # Convert back to DictConfig
        return OmegaConf.create(resolved)

    def _resolve_paths_recursive(self, obj: Any, parent_key: str = "") -> Any:
        """
        Recursively resolve path-like values to absolute paths.

        Args:
            obj: Object to process (dict, list, or value)
            parent_key: Parent key name for context

        Returns:
            Object with resolved paths
        """
        if isinstance(obj, dict):
            return {
                key: self._resolve_paths_recursive(value, key)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._resolve_paths_recursive(item, parent_key) for item in obj]
        elif isinstance(obj, str):
            # Resolve paths if key suggests it's a path
            if any(path_key in parent_key.lower() for path_key in
                   ["path", "dir", "root", "file", "model"]):
                return self._resolve_single_path(obj)

        return obj

    def _resolve_single_path(self, path_str: str) -> str:
        """
        Resolve a single path string to absolute path.

        Args:
            path_str: Path string (may be relative or absolute)

        Returns:
            Absolute path string
        """
        # Skip if already absolute
        if path_str.startswith("/"):
            return path_str

        # Skip HuggingFace model IDs (organization/model format without file extensions)
        if "/" in path_str:
            # Check if it looks like a HuggingFace model ID:
            # - Contains exactly one "/" (or maybe more for submodules)
            # - Doesn't end with file extensions
            # - Doesn't start with common path prefixes
            if not path_str.startswith(("./", "../", "data/", "models/", "outputs/", "configs/")):
                # Likely a HuggingFace model ID or similar identifier
                # Examples: "openai/clip-vit-large-patch14", "Salesforce/blip2-opt-2.7b"
                parts = path_str.split("/")
                if len(parts) <= 3 and not any(part.endswith((".py", ".yaml", ".json", ".toml", ".safetensors", ".pt", ".pth"))
                                               for part in parts):
                    return path_str

        # Skip simple identifiers without slashes
        if "/" not in path_str:
            return path_str

        # Skip URLs
        if "://" in path_str or path_str.startswith("http"):
            return path_str

        # Convert to Path and resolve
        path = Path(path_str)

        # If relative, make it relative to repo root
        if not path.is_absolute():
            path = REPO_ROOT / path

        return str(path.resolve())

    def merge_configs(self, *configs: DictConfig) -> DictConfig:
        """
        Merge multiple configurations with priority order.

        Later configs override earlier configs.

        Args:
            *configs: Configuration objects to merge

        Returns:
            Merged configuration
        """
        if not configs:
            return OmegaConf.create()

        merged = configs[0]
        for config in configs[1:]:
            merged = OmegaConf.merge(merged, config)

        return merged

    def get_full_config(self,
                       project: Optional[str] = None,
                       character: Optional[str] = None,
                       stage: Optional[str] = None) -> DictConfig:
        """
        Get merged configuration combining global, project, character, and stage configs.

        Merge priority (later overrides earlier):
        1. Global pipeline config
        2. Global models config
        3. Stage config (if specified)
        4. Character config (if specified)
        5. Project config (if specified)

        Variables are resolved AFTER merging, allowing cross-config references.

        Args:
            project: Project name (e.g., "simpsons")
            character: Character name (e.g., "homer")
            stage: Stage name (e.g., "toonout")

        Returns:
            Merged configuration with resolved variables
        """
        configs_to_merge = []

        # 1. Load global configs (without resolving variables)
        try:
            pipeline_config = self.load("pipeline", config_type="global", resolve_variables=False)
            configs_to_merge.append(pipeline_config)
        except FileNotFoundError:
            logger.warning("Global pipeline.yaml not found, skipping")

        try:
            models_config = self.load("models", config_type="global", resolve_variables=False)
            configs_to_merge.append(models_config)
        except FileNotFoundError:
            logger.warning("Global models.yaml not found, skipping")

        # 2. Load stage config
        if stage:
            try:
                stage_config = self.load(stage, config_type="stage", resolve_variables=False)
                configs_to_merge.append(stage_config)
            except FileNotFoundError:
                logger.warning(f"Stage config '{stage}' not found, skipping")

        # 3. Load character config
        if character:
            try:
                char_config = self.load(character, config_type="character", resolve_variables=False)
                configs_to_merge.append(char_config)
            except FileNotFoundError:
                logger.warning(f"Character config '{character}' not found, skipping")

        # 4. Load project config (highest priority)
        if project:
            try:
                project_config = self.load(project, config_type="project", resolve_variables=False)
                configs_to_merge.append(project_config)
            except FileNotFoundError:
                logger.warning(f"Project config '{project}' not found, skipping")

        # Merge all configs
        if not configs_to_merge:
            logger.warning("No configuration files loaded, returning empty config")
            return OmegaConf.create()

        merged = self.merge_configs(*configs_to_merge)

        # Resolve variables AFTER merging (allows cross-config references)
        merged = self._resolve_paths(merged)

        return merged

    def clear_cache(self):
        """Clear configuration cache."""
        self._cache.clear()
        logger.debug("Configuration cache cleared")


# Global loader instance
_global_loader = ConfigLoader()


def load_config(name: str,
               config_type: str = "global",
               use_cache: bool = True,
               resolve_variables: bool = True,
               resolve_env: bool = False) -> DictConfig:
    """
    Load configuration file using global loader.

    Args:
        name: Configuration name (without .yaml extension)
        config_type: Type of config - "global", "stage", "character", "project", "training", "animation"
        use_cache: Whether to use cached config if available
        resolve_variables: Whether to resolve OmegaConf variables (default: True for standalone loading)

    Returns:
        OmegaConf DictConfig object

    Example:
        >>> config = load_config("pipeline")
        >>> seg_config = load_config("toonout", config_type="stage")
        >>> # Load without resolving variables (for manual merging later)
        >>> raw_config = load_config("simpsons", config_type="project", resolve_variables=False)
    """
    return _global_loader.load(
        name,
        config_type=config_type,
        use_cache=use_cache,
        resolve_variables=resolve_variables,
        resolve_env=resolve_env,
    )


def get_config(project: Optional[str] = None,
              character: Optional[str] = None,
              stage: Optional[str] = None) -> DictConfig:
    """
    Get merged configuration using global loader.

    Args:
        project: Project name (e.g., "simpsons")
        character: Character name (e.g., "homer")
        stage: Stage name (e.g., "toonout")

    Returns:
        Merged configuration

    Example:
        >>> config = get_config(project="simpsons", character="homer", stage="toonout")
    """
    return _global_loader.get_full_config(project=project, character=character, stage=stage)


def merge_configs(*configs: Union[DictConfig, Dict]) -> DictConfig:
    """
    Merge multiple configurations.

    Args:
        *configs: Configuration objects to merge

    Returns:
        Merged configuration

    Example:
        >>> merged = merge_configs(global_config, project_config, custom_config)
    """
    # Convert dicts to DictConfig
    omega_configs = [
        OmegaConf.create(c) if isinstance(c, dict) else c
        for c in configs
    ]
    return _global_loader.merge_configs(*omega_configs)


def save_config(config: Union[DictConfig, Dict],
               output_path: Union[str, Path]):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration to save
        output_path: Output file path

    Example:
        >>> save_config(config, "outputs/run_config.yaml")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, dict):
        config = OmegaConf.create(config)

    with open(output_path, 'w') as f:
        OmegaConf.save(config, f)

    logger.info(f"Configuration saved to: {output_path}")


def validate_config(config: DictConfig, required_keys: List[str]) -> bool:
    """
    Validate configuration has required keys.

    Args:
        config: Configuration to validate
        required_keys: List of required key paths (e.g., ["paths.warehouse_root", "models.clip"])

    Returns:
        True if valid, raises ValueError if invalid

    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = []

    for key_path in required_keys:
        keys = key_path.split(".")
        current = config

        for key in keys:
            if key not in current:
                missing_keys.append(key_path)
                break
            current = current[key]

    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    return True


# ============================================================================
# AI_WAREHOUSE 3.0 SUPPORT FUNCTIONS
# ============================================================================

def load_warehouse_config(use_cache: bool = True) -> DictConfig:
    """
    Load AI_WAREHOUSE 3.0 configuration.

    Returns:
        OmegaConf DictConfig with warehouse paths and settings
    """
    return _global_loader.load("warehouse", config_type="global", use_cache=use_cache,
                               resolve_variables=True)


def load_models_config(use_cache: bool = True) -> DictConfig:
    """
    Load centralized models configuration.

    Returns:
        OmegaConf DictConfig with model paths and settings
    """
    return _global_loader.load("models", config_type="global", use_cache=use_cache,
                               resolve_variables=True)


def inject_environment_variables(config: Optional[DictConfig] = None) -> None:
    """
    Set environment variables from warehouse config.

    This sets HF_HOME, TRANSFORMERS_CACHE, TORCH_HOME, etc. based on
    the AI_WAREHOUSE 3.0 cache configuration.

    Args:
        config: Optional warehouse config. If None, loads from warehouse.yaml
    """
    import os

    if config is None:
        try:
            config = load_warehouse_config()
        except FileNotFoundError:
            logger.warning("warehouse.yaml not found, skipping environment variable injection")
            return

    # Get env_vars section
    env_vars = config.get("env_vars", {})

    for var_name, var_value in env_vars.items():
        if var_value:
            # Resolve any remaining variables
            resolved_value = str(var_value)
            os.environ[var_name] = resolved_value
            logger.debug(f"Set environment variable: {var_name}={resolved_value}")


def validate_no_deprecated_paths(config: DictConfig) -> List[str]:
    """
    Check configuration for deprecated path usage.

    Scans all string values in the config for deprecated paths defined
    in warehouse.yaml's deprecated section.

    Args:
        config: Configuration to validate

    Returns:
        List of warning messages for any deprecated paths found
    """
    warnings = []

    # Load warehouse config to get deprecated paths
    try:
        warehouse_config = load_warehouse_config()
        deprecated_paths = warehouse_config.get("deprecated", [])
    except FileNotFoundError:
        logger.warning("warehouse.yaml not found, cannot validate deprecated paths")
        return []

    # Flatten config and check all string values
    def check_recursive(obj: Any, path: str = "") -> None:
        if isinstance(obj, DictConfig) or isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                check_recursive(value, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_recursive(item, f"{path}[{i}]")
        elif isinstance(obj, str):
            for dep_path in deprecated_paths:
                # Expand home directory
                expanded_dep = dep_path.replace("~", str(Path.home()))
                expanded_dep = expanded_dep.replace("$HOME", str(Path.home()))

                if expanded_dep in obj or dep_path in obj:
                    warnings.append(f"Deprecated path in '{path}': {obj}")

    check_recursive(config)
    return warnings


def get_warehouse_path(path_key: str, animation_type: str = "2d") -> Optional[str]:
    """
    Get a specific path from warehouse configuration.

    Args:
        path_key: Key in warehouse config (e.g., "datasets", "training_data", "lora_outputs")
        animation_type: "2d" or "3d" to select animation-specific paths

    Returns:
        Resolved path string or None if not found

    Example:
        >>> datasets_path = get_warehouse_path("datasets", "2d")
        "/mnt/data/datasets/general"
    """
    try:
        config = load_warehouse_config()
    except FileNotFoundError:
        logger.warning("warehouse.yaml not found")
        return None

    # Check animation-specific paths first
    animation_key = f"animation_{animation_type}"
    animation_config = config.get(animation_key, {})

    if path_key in animation_config:
        return str(animation_config[path_key])

    # Fall back to roots
    roots = config.get("roots", {})
    if path_key in roots:
        return str(roots[path_key])

    return None


def get_model_path(model_type: str, model_name: str) -> Optional[str]:
    """
    Get a specific model path from models configuration.

    Args:
        model_type: Model category (e.g., "segmentation", "detection", "vlm")
        model_name: Specific model name (e.g., "sam2_hiera_large", "yolov11n")

    Returns:
        Model path string or HuggingFace ID, or None if not found

    Example:
        >>> sam2_path = get_model_path("segmentation", "sam2_hiera_large")
        "/mnt/c/ai_models/segmentation/sam2_hiera_large.pt"
    """
    try:
        config = load_models_config()
    except FileNotFoundError:
        logger.warning("models.yaml not found")
        return None

    model_category = config.get(model_type, {})
    model_info = model_category.get(model_name, {})

    if isinstance(model_info, dict):
        # Return path or hf_id
        return model_info.get("path") or model_info.get("hf_id")
    elif isinstance(model_info, str):
        return model_info

    return None


def get_default_model(task: str) -> Optional[str]:
    """
    Get the default model for a task from models configuration.

    Args:
        task: Task name (e.g., "segmentation", "detection", "vlm")

    Returns:
        Default model name for the task, or None if not found

    Example:
        >>> default_seg = get_default_model("segmentation")
        "toonout"
    """
    try:
        config = load_models_config()
    except FileNotFoundError:
        logger.warning("models.yaml not found")
        return None

    defaults = config.get("defaults", {})
    return defaults.get(task)


def setup_warehouse_environment() -> None:
    """
    Set up the full AI_WAREHOUSE 3.0 environment.

    This:
    1. Loads warehouse configuration
    2. Sets environment variables for caches
    3. Validates no deprecated paths are in use
    4. Logs any warnings
    """
    logger.info("Setting up AI_WAREHOUSE 3.0 environment...")

    try:
        # Load and validate warehouse config
        warehouse_config = load_warehouse_config()

        # Set environment variables
        inject_environment_variables(warehouse_config)

        # Log configuration version
        version = warehouse_config.get("version", "unknown")
        logger.info(f"AI_WAREHOUSE version: {version}")

        # Check for deprecated paths in the warehouse config itself
        warnings = validate_no_deprecated_paths(warehouse_config)
        for warning in warnings:
            logger.warning(warning)

        logger.info("AI_WAREHOUSE 3.0 environment setup complete")

    except FileNotFoundError:
        logger.warning("AI_WAREHOUSE 3.0 configuration not found. Using default paths.")
    except Exception as e:
        logger.error(f"Error setting up warehouse environment: {e}")


def main():
    """Test configuration loader."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Configuration Loader")
    parser.add_argument("--name", type=str, help="Config name (required unless using --project/--character/--stage)")
    parser.add_argument("--type", type=str, default="global",
                       choices=["global", "stage", "character", "project", "training", "animation"],
                       help="Config type")
    parser.add_argument("--project", type=str, help="Project name for merged config")
    parser.add_argument("--character", type=str, help="Character name for merged config")
    parser.add_argument("--stage", type=str, help="Stage name for merged config")
    parser.add_argument("--output", type=str, help="Save merged config to file")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Validate arguments
    if not args.name and not (args.project or args.character or args.stage):
        parser.error("Either --name or at least one of --project/--character/--stage is required")

    # Load configuration
    if args.project or args.character or args.stage:
        print(f"\n=== Loading Merged Configuration ===")
        if args.project:
            print(f"Project: {args.project}")
        if args.character:
            print(f"Character: {args.character}")
        if args.stage:
            print(f"Stage: {args.stage}")
        print()
        config = get_config(project=args.project, character=args.character, stage=args.stage)
    else:
        print(f"\n=== Loading {args.type.upper()} Configuration: {args.name} ===")
        config = load_config(args.name, config_type=args.type)

    # Print configuration
    print(OmegaConf.to_yaml(config))

    # Save if requested
    if args.output:
        save_config(config, args.output)
        print(f"\nConfiguration saved to: {args.output}")


if __name__ == "__main__":
    main()
