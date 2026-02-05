#!/usr/bin/env python3
"""
Unified Stub Framework for 2D Animation Pipeline

Eliminates duplicated stub patterns across multiple modules.
Provides consistent fallback behavior and error handling.

Problem Solved:
- 8+ modules had duplicate stub fallback logic (~100 lines each)
- Inconsistent error handling and logging
- Hard to maintain and test

Solution:
- Single source of truth for stub mode detection
- Unified fallback execution pattern
- Consistent logging and error messages

Usage Example:
    from anime_pipeline.core.stub_framework import StubMode, StubConfig

    def segment_foreground(config, logger):
        return StubMode.run_with_fallback(
            model_loader=load_toonout_model,
            real_inference=_run_real_segmentation,
            stub_inference=_generate_stub_segmentation,
            config=StubConfig(**config),
            logger=logger
        )

Author: Created for Phase 4 code deduplication
Date: 2025-01-XX
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Dict
import logging


@dataclass
class StubConfig:
    """
    Common stub configuration for all pipeline modules.

    Attributes:
        enabled: Whether the module is enabled
        backend: Backend to use (stub/pytorch/onnx/tensorrt)
        use_stub: Explicit stub mode flag
        model_path: Path to model weights
        device: Device to use (cuda/cpu)
    """
    enabled: bool = True
    backend: str = "stub"
    use_stub: bool = True
    model_path: Optional[str] = None
    device: str = "cpu"

    # Optional backend-specific settings
    precision: str = "fp32"  # fp16/bf16/fp32
    batch_size: int = 1

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "StubConfig":
        """
        Create StubConfig from configuration dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            StubConfig instance
        """
        # Extract relevant fields
        kwargs = {}
        for field in ["enabled", "backend", "use_stub", "model_path",
                      "device", "precision", "batch_size"]:
            if field in config_dict:
                kwargs[field] = config_dict[field]

        return cls(**kwargs)


class StubMode:
    """
    Unified stub mode handler for all pipeline modules.

    Provides consistent fallback behavior when:
    - Model weights are missing
    - Model loading fails
    - Inference fails
    - User explicitly requests stub mode
    """

    @staticmethod
    def should_use_stub(config: StubConfig, logger: Optional[logging.Logger] = None) -> bool:
        """
        Determine if stub mode should be used.

        Decision logic:
        1. Explicit use_stub flag
        2. Backend is "stub"
        3. Model path is None
        4. Model path doesn't exist

        Args:
            config: Stub configuration
            logger: Optional logger for warnings

        Returns:
            True if stub mode should be used
        """
        # Check explicit stub flag
        if config.use_stub:
            if logger:
                logger.info("Using stub mode (explicitly requested)")
            return True

        # Check backend
        if config.backend == "stub":
            if logger:
                logger.info("Using stub mode (backend='stub')")
            return True

        # Check model path
        if not config.model_path:
            if logger:
                logger.info("Using stub mode (no model path provided)")
            return True

        # Check if model exists
        model_path = Path(config.model_path)
        if not model_path.exists():
            if logger:
                logger.warning(
                    f"Model path does not exist: {config.model_path}, "
                    f"using stub mode"
                )
            return True

        return False

    @staticmethod
    def run_with_fallback(
        model_loader: Callable[[StubConfig, logging.Logger], Any],
        real_inference: Callable[..., Any],
        stub_inference: Callable[..., Any],
        config: StubConfig,
        logger: logging.Logger,
        **kwargs
    ) -> Any:
        """
        Execute inference with automatic fallback to stub.

        Tries real model inference, falls back to stub on any error.

        Execution flow:
        1. Check if stub mode should be used
        2. If yes, run stub inference directly
        3. If no, try to load model
        4. If load fails, fall back to stub
        5. Try real inference
        6. If inference fails, fall back to stub

        Args:
            model_loader: Function to load model
                         Signature: (config: StubConfig, logger: Logger) -> model
            real_inference: Function to run real inference
                          Signature: (model, **kwargs) -> result
            stub_inference: Function to run stub inference
                          Signature: (**kwargs) -> result
            config: Stub configuration
            logger: Logger instance
            **kwargs: Additional arguments passed to inference functions

        Returns:
            Inference result (from real or stub inference)

        Example:
            >>> config = StubConfig(model_path="/path/to/model.pt")
            >>> result = StubMode.run_with_fallback(
            ...     model_loader=load_yolo,
            ...     real_inference=run_yolo_detection,
            ...     stub_inference=generate_stub_detections,
            ...     config=config,
            ...     logger=logger,
            ...     image_path="frame.jpg"
            ... )
        """
        # Check if we should use stub mode
        if StubMode.should_use_stub(config, logger):
            return stub_inference(**kwargs)

        # Try to load real model
        logger.info(f"Loading model from {config.model_path}...")
        try:
            model = model_loader(config, logger)
        except Exception as e:
            logger.warning(
                f"Model loading failed: {e}, falling back to stub mode",
                exc_info=True
            )
            return stub_inference(**kwargs)

        # Try real inference
        logger.debug("Running real inference...")
        try:
            result = real_inference(model, **kwargs)
            return result
        except Exception as e:
            logger.warning(
                f"Real inference failed: {e}, falling back to stub mode",
                exc_info=True
            )
            return stub_inference(**kwargs)

    @staticmethod
    def run_with_fallback_advanced(
        model_loader: Callable[[StubConfig, logging.Logger], Any],
        real_inference: Callable[..., Any],
        stub_inference: Callable[..., Any],
        config: StubConfig,
        logger: logging.Logger,
        validate_result: Optional[Callable[[Any], bool]] = None,
        **kwargs
    ) -> Any:
        """
        Advanced fallback with result validation.

        Same as run_with_fallback but additionally validates the result.
        Falls back to stub if validation fails.

        Args:
            model_loader: Function to load model
            real_inference: Function to run real inference
            stub_inference: Function to run stub inference
            config: Stub configuration
            logger: Logger instance
            validate_result: Optional function to validate result
                           Signature: (result) -> bool
            **kwargs: Additional arguments

        Returns:
            Validated inference result

        Example:
            >>> def validate_detections(result):
            ...     return len(result) > 0
            >>>
            >>> result = StubMode.run_with_fallback_advanced(
            ...     model_loader=load_yolo,
            ...     real_inference=run_yolo_detection,
            ...     stub_inference=generate_stub_detections,
            ...     config=config,
            ...     logger=logger,
            ...     validate_result=validate_detections,
            ...     image_path="frame.jpg"
            ... )
        """
        # Run with normal fallback
        result = StubMode.run_with_fallback(
            model_loader=model_loader,
            real_inference=real_inference,
            stub_inference=stub_inference,
            config=config,
            logger=logger,
            **kwargs
        )

        # Validate result if validator provided
        if validate_result is not None:
            try:
                is_valid = validate_result(result)
                if not is_valid:
                    logger.warning(
                        "Real inference result failed validation, "
                        "falling back to stub mode"
                    )
                    return stub_inference(**kwargs)
            except Exception as e:
                logger.warning(
                    f"Result validation failed: {e}, "
                    f"falling back to stub mode"
                )
                return stub_inference(**kwargs)

        return result


class StubRegistry:
    """
    Registry of stub implementations for different modules.

    Allows centralized management of stub behaviors.
    """

    _registry: Dict[str, Dict[str, Callable]] = {}

    @classmethod
    def register(
        cls,
        module_name: str,
        loader: Callable,
        real_inference: Callable,
        stub_inference: Callable
    ):
        """
        Register a module's stub implementation.

        Args:
            module_name: Name of the module (e.g., "yolo_detector")
            loader: Model loader function
            real_inference: Real inference function
            stub_inference: Stub inference function
        """
        cls._registry[module_name] = {
            "loader": loader,
            "real": real_inference,
            "stub": stub_inference
        }

    @classmethod
    def run(
        cls,
        module_name: str,
        config: StubConfig,
        logger: logging.Logger,
        **kwargs
    ) -> Any:
        """
        Run inference for a registered module.

        Args:
            module_name: Name of the module
            config: Stub configuration
            logger: Logger instance
            **kwargs: Additional arguments

        Returns:
            Inference result

        Raises:
            ValueError: If module not registered
        """
        if module_name not in cls._registry:
            raise ValueError(
                f"Module '{module_name}' not registered. "
                f"Available modules: {list(cls._registry.keys())}"
            )

        funcs = cls._registry[module_name]

        return StubMode.run_with_fallback(
            model_loader=funcs["loader"],
            real_inference=funcs["real"],
            stub_inference=funcs["stub"],
            config=config,
            logger=logger,
            **kwargs
        )

    @classmethod
    def list_modules(cls) -> list:
        """Get list of registered modules."""
        return list(cls._registry.keys())


# Utility functions for common stub patterns

def create_deterministic_stub(seed_str: str, base_value: float = 0.5) -> float:
    """
    Create deterministic stub value from seed string.

    Useful for generating consistent stub outputs.

    Args:
        seed_str: String to use as seed
        base_value: Base value to perturb

    Returns:
        Deterministic value in range [base_value - 0.1, base_value + 0.1]
    """
    import hashlib

    hash_val = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16)
    perturbation = (hash_val % 200 - 100) / 1000.0  # [-0.1, 0.1]

    return base_value + perturbation


def create_stub_bbox(
    image_width: int,
    image_height: int,
    seed_str: str,
    center_bias: float = 0.5
) -> tuple:
    """
    Create deterministic stub bounding box.

    Args:
        image_width: Image width
        image_height: Image height
        seed_str: Seed string for determinism
        center_bias: Bias towards center (0=corners, 1=center)

    Returns:
        (x1, y1, x2, y2) bounding box
    """
    import hashlib

    hash_val = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16)

    # Generate box parameters
    x_center = (hash_val % 100) / 100.0
    y_center = ((hash_val >> 8) % 100) / 100.0

    # Apply center bias
    x_center = x_center * (1 - center_bias) + 0.5 * center_bias
    y_center = y_center * (1 - center_bias) + 0.5 * center_bias

    # Box size (30-50% of image)
    width_ratio = 0.3 + ((hash_val >> 16) % 20) / 100.0
    height_ratio = 0.3 + ((hash_val >> 24) % 20) / 100.0

    # Calculate coordinates
    box_width = int(image_width * width_ratio)
    box_height = int(image_height * height_ratio)

    x1 = max(0, int(x_center * image_width - box_width / 2))
    y1 = max(0, int(y_center * image_height - box_height / 2))
    x2 = min(image_width, x1 + box_width)
    y2 = min(image_height, y1 + box_height)

    return (x1, y1, x2, y2)
