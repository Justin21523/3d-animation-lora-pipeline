#!/usr/bin/env python3
"""
Parameter Converter for 2D/3D Animation Pipelines

Automatically converts parameters between 2D Western animation and 3D Pixar-style
animation based on the parameter mapping table.

Features:
- Automatic parameter scaling (numeric values)
- Direct replacement (non-numeric values like strings, booleans)
- Parameter validation for animation style
- Detailed conversion logging

Author: Adapted from 3D pipeline
Date: 2025-01-XX
"""

import logging
from pathlib import Path
from typing import Dict, Any, Literal, Optional, List

from omegaconf import OmegaConf, DictConfig


AnimationStyle = Literal["2d", "3d"]

logger = logging.getLogger(__name__)


class ParameterConverter:
    """
    Convert parameters between 2D and 3D animation defaults.

    Uses parameter mapping table to automatically adapt pipeline parameters
    for different animation styles.
    """

    def __init__(self, mapping_path: Optional[Path] = None):
        """
        Initialize parameter converter.

        Args:
            mapping_path: Path to param_mapping.yaml (default: configs/global/param_mapping.yaml)
        """
        if mapping_path is None:
            # Default to configs/global/param_mapping.yaml
            mapping_path = Path(__file__).parents[2] / "configs/global/param_mapping.yaml"

        if not mapping_path.exists():
            raise FileNotFoundError(
                f"Parameter mapping file not found: {mapping_path}\n"
                f"Please ensure configs/global/param_mapping.yaml exists."
            )

        self.mapping = OmegaConf.load(mapping_path)
        self.mapping_path = mapping_path
        logger.info(f"Loaded parameter mapping from: {mapping_path}")

    def convert_config(
        self,
        config: Dict[str, Any],
        source_style: AnimationStyle,
        target_style: AnimationStyle,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert configuration from source style to target style.

        Args:
            config: Configuration dictionary to convert
            source_style: Source animation style ("2d" or "3d")
            target_style: Target animation style ("2d" or "3d")
            overrides: Optional manual parameter overrides (highest priority)

        Returns:
            Converted configuration dictionary

        Example:
            >>> converter = ParameterConverter()
            >>> # Convert 3D defaults to 2D
            >>> config_2d = converter.convert_config(
            ...     {"segmentation": {"alpha_threshold": 0.15}},
            ...     source_style="3d",
            ...     target_style="2d"
            ... )
            >>> # Result: {"segmentation": {"alpha_threshold": 0.25}}
        """
        if source_style == target_style:
            logger.info(f"Source and target styles are the same ({source_style}), no conversion needed")
            return config.copy()

        logger.info(f"Converting config from {source_style} to {target_style}")
        converted = self._deep_copy_dict(config)

        # Track conversions for logging
        conversions: List[str] = []

        # Convert each category
        for category in self.mapping.keys():
            if category in ["version"]:  # Skip metadata
                continue

            if category not in converted:
                continue

            param_mappings = self.mapping[category]

            for param, mapping in param_mappings.items():
                # Skip non-parameter fields (description, rationale)
                if param in ["description", "rationale"]:
                    continue

                # Handle nested parameters (e.g., color_jitter.brightness)
                if "." in param:
                    self._convert_nested_param(
                        converted[category], param, mapping,
                        source_style, target_style, conversions
                    )
                else:
                    self._convert_flat_param(
                        converted[category], param, mapping,
                        source_style, target_style, conversions
                    )

        # Apply overrides (highest priority)
        if overrides:
            logger.info(f"Applying manual overrides: {list(overrides.keys())}")
            converted = self._apply_overrides(converted, overrides)

        # Log all conversions
        if conversions:
            logger.info(f"Applied {len(conversions)} parameter conversions:")
            for conversion in conversions:
                logger.debug(f"  {conversion}")

        return converted

    def _convert_flat_param(
        self,
        category_config: Dict,
        param: str,
        mapping: Dict,
        source_style: str,
        target_style: str,
        conversions: List[str]
    ):
        """Convert a flat parameter (not nested)."""
        if param not in category_config:
            return

        source_key = f"{source_style}_default"
        target_key = f"{target_style}_default"

        if source_key not in mapping or target_key not in mapping:
            return

        source_val = mapping[source_key]
        target_val = mapping[target_key]
        current_val = category_config[param]

        # Convert based on type
        if isinstance(target_val, (int, float)) and isinstance(current_val, (int, float)):
            # Scale numeric values proportionally
            if source_val != 0:
                scale = target_val / source_val
                new_val = current_val * scale
                category_config[param] = type(target_val)(new_val)  # Preserve int/float type
            else:
                category_config[param] = target_val

            conversions.append(
                f"{param}: {current_val} → {category_config[param]} "
                f"(scale={target_val/source_val if source_val != 0 else 'N/A'})"
            )
        else:
            # Direct replacement for non-numeric values
            category_config[param] = target_val
            conversions.append(f"{param}: {current_val} → {target_val}")

    def _convert_nested_param(
        self,
        category_config: Dict,
        param_path: str,
        mapping: Dict,
        source_style: str,
        target_style: str,
        conversions: List[str]
    ):
        """Convert a nested parameter (e.g., color_jitter.brightness)."""
        keys = param_path.split(".")

        # Navigate to parent dict
        current = category_config
        for key in keys[:-1]:
            if key not in current:
                return
            current = current[key]

        # Convert the final key
        final_key = keys[-1]
        if final_key not in current:
            return

        source_key = f"{source_style}_default"
        target_key = f"{target_style}_default"

        if source_key not in mapping or target_key not in mapping:
            return

        target_val = mapping[target_key]
        current_val = current[final_key]

        # Direct replacement (nested params are usually simple types)
        current[final_key] = target_val
        conversions.append(f"{param_path}: {current_val} → {target_val}")

    def _deep_copy_dict(self, d: Dict) -> Dict:
        """Deep copy a dictionary."""
        import copy
        return copy.deepcopy(d)

    def _apply_overrides(self, config: Dict, overrides: Dict) -> Dict:
        """Apply manual overrides to config (supports nested keys with dot notation)."""
        for key, value in overrides.items():
            if "." in key:
                # Handle nested keys like "segmentation.alpha_threshold"
                keys = key.split(".")
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value
        return config

    def validate_parameters(
        self,
        config: Dict,
        mode: AnimationStyle
    ) -> bool:
        """
        Validate parameter combinations for animation style.

        Args:
            config: Configuration to validate
            mode: Animation style ("2d" or "3d")

        Returns:
            True if valid

        Raises:
            ValueError: If invalid parameter combinations detected
        """
        invalid_rules = self._get_validation_rules(mode)

        errors = []
        for rule in invalid_rules:
            condition = rule["condition"]
            error_msg = rule["message"]
            suggestion = rule["suggestion"]

            if self._eval_condition(condition, config):
                errors.append(f"{error_msg}\n  Suggestion: {suggestion}")

        if errors:
            error_summary = "\n".join([f"  - {err}" for err in errors])
            raise ValueError(
                f"Invalid parameters for {mode} mode:\n{error_summary}"
            )

        logger.info(f"Parameter validation passed for {mode} mode")
        return True

    def _get_validation_rules(self, mode: AnimationStyle) -> List[Dict]:
        """Get validation rules for animation style."""
        if mode == "2d":
            return [
                {
                    "condition": "segmentation.alpha_threshold < 0.20",
                    "message": "alpha_threshold too low for 2D hard edges",
                    "suggestion": "Use >= 0.25 for 2D animation"
                },
                {
                    "condition": "clustering.min_cluster_size < 15",
                    "message": "min_cluster_size too small for 2D variation",
                    "suggestion": "Use >= 20 for 2D animation"
                },
                {
                    "condition": "clustering.min_samples < 3",
                    "message": "min_samples too small for 2D style variance",
                    "suggestion": "Use >= 4 for 2D animation"
                },
            ]
        else:  # 3d
            return [
                {
                    "condition": "segmentation.alpha_threshold > 0.20",
                    "message": "alpha_threshold too high for 3D soft edges",
                    "suggestion": "Use <= 0.15 for 3D animation"
                },
                {
                    "condition": "augmentation.color_jitter.enabled == True",
                    "message": "color_jitter breaks 3D PBR materials",
                    "suggestion": "Disable color_jitter for 3D animation"
                },
                {
                    "condition": "clustering.min_cluster_size > 15",
                    "message": "min_cluster_size unnecessarily large for consistent 3D models",
                    "suggestion": "Use 10-15 for 3D animation"
                },
            ]

    def _eval_condition(self, condition: str, config: Dict) -> bool:
        """
        Evaluate a validation condition.

        Args:
            condition: Condition string (e.g., "segmentation.alpha_threshold < 0.20")
            config: Configuration dictionary

        Returns:
            True if condition matches (validation fails), False otherwise
        """
        try:
            # Parse condition
            if " < " in condition:
                key, threshold = condition.split(" < ")
                value = self._get_nested_value(config, key.strip())
                return value is not None and value < float(threshold.strip())
            elif " > " in condition:
                key, threshold = condition.split(" > ")
                value = self._get_nested_value(config, key.strip())
                return value is not None and value > float(threshold.strip())
            elif " == " in condition:
                key, expected = condition.split(" == ")
                value = self._get_nested_value(config, key.strip())
                expected = expected.strip()
                # Handle boolean comparison
                if expected.lower() == "true":
                    return value is True
                elif expected.lower() == "false":
                    return value is False
                else:
                    return value == expected
            else:
                logger.warning(f"Unknown condition format: {condition}")
                return False
        except Exception as e:
            logger.warning(f"Error evaluating condition '{condition}': {e}")
            return False

    def _get_nested_value(self, config: Dict, key_path: str) -> Any:
        """Get nested value from config using dot notation."""
        keys = key_path.split(".")
        current = config

        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]

        return current


def main():
    """Test parameter converter."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Parameter Converter")
    parser.add_argument("--source", choices=["2d", "3d"], required=True,
                       help="Source animation style")
    parser.add_argument("--target", choices=["2d", "3d"], required=True,
                       help="Target animation style")
    parser.add_argument("--config", type=str,
                       help="Config file to convert (optional, uses defaults if not provided)")
    parser.add_argument("--validate", action="store_true",
                       help="Validate converted parameters")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Create converter
    converter = ParameterConverter()

    # Load or create test config
    if args.config:
        from anime_pipeline.config.omega_loader import load_config
        config = OmegaConf.to_container(load_config(args.config), resolve=True)
    else:
        # Use example config
        config = {
            "segmentation": {
                "alpha_threshold": 0.15 if args.source == "3d" else 0.25,
                "blur_threshold": 80 if args.source == "3d" else 100,
            },
            "clustering": {
                "min_cluster_size": 10 if args.source == "3d" else 20,
                "min_samples": 2 if args.source == "3d" else 4,
            },
            "augmentation": {
                "color_jitter": {
                    "enabled": False if args.source == "3d" else True,
                }
            }
        }

    print(f"\n=== Original Config ({args.source.upper()}) ===")
    print(OmegaConf.to_yaml(OmegaConf.create(config)))

    # Convert
    converted = converter.convert_config(config, args.source, args.target)

    print(f"\n=== Converted Config ({args.target.upper()}) ===")
    print(OmegaConf.to_yaml(OmegaConf.create(converted)))

    # Validate if requested
    if args.validate:
        try:
            converter.validate_parameters(converted, args.target)
            print(f"\n✓ Validation passed for {args.target} mode")
        except ValueError as e:
            print(f"\n✗ Validation failed:\n{e}")
            return 1

    return 0


# ----------------------------------------------------------------------------
# Lightweight helpers (used by tests and simple scripts)
# ----------------------------------------------------------------------------

_FLAT_PARAM_MAP = {
    # Common flat keys used across scripts
    "alpha_threshold": ("segmentation", "alpha_threshold"),
    "blur_threshold": ("segmentation", "blur_threshold"),
    "min_cluster_size": ("clustering", "min_cluster_size"),
    # Friendly aliases for augmentation toggles
    "color_aug": ("augmentation", "color_jitter", "enabled"),
    "flip_aug": ("augmentation", "horizontal_flip", "enabled"),
}


def _get_mapping_entry(mapping: DictConfig, path: tuple[str, ...]) -> Optional[Dict[str, Any]]:
    current: Any = mapping
    for part in path:
        if not isinstance(current, (dict, DictConfig)) or part not in current:
            return None
        current = current[part]
    if isinstance(current, (dict, DictConfig)):
        return OmegaConf.to_container(current, resolve=True)  # type: ignore[return-value]
    return None


def get_defaults(mode: AnimationStyle = "2d", mapping_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get a small, flat default-parameter dict for the requested animation mode.

    This is intentionally lightweight (not a full nested config) and is primarily
    meant for quick utilities and tests.
    """
    converter = ParameterConverter(mapping_path=mapping_path)
    defaults: Dict[str, Any] = {}

    for key, mapping_key in _FLAT_PARAM_MAP.items():
        entry = _get_mapping_entry(converter.mapping, mapping_key)
        if not entry:
            continue
        defaults[key] = entry.get(f"{mode}_default")

    return defaults


def convert_params(
    params: Dict[str, Any],
    from_mode: AnimationStyle,
    to_mode: AnimationStyle,
    mapping_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convert a flat parameter dict between 2D/3D defaults.

    This is a convenience wrapper around the mapping table, intended for scripts
    that don't maintain a full nested config structure.
    """
    if from_mode == to_mode:
        return dict(params)

    converter = ParameterConverter(mapping_path=mapping_path)
    converted: Dict[str, Any] = dict(params)

    for key, mapping_key in _FLAT_PARAM_MAP.items():
        if key not in params:
            continue

        entry = _get_mapping_entry(converter.mapping, mapping_key)
        if not entry:
            continue

        source_default = entry.get(f"{from_mode}_default")
        target_default = entry.get(f"{to_mode}_default")
        current_val = params[key]

        # Numeric scaling (preserve relative tuning)
        if isinstance(current_val, (int, float)) and isinstance(source_default, (int, float)) and isinstance(target_default, (int, float)):
            if source_default != 0:
                scale = target_default / source_default
                converted[key] = type(target_default)(current_val * scale)
            else:
                converted[key] = target_default
            continue

        # Non-numeric: direct replacement
        converted[key] = target_default

    return converted


if __name__ == "__main__":
    exit(main())
