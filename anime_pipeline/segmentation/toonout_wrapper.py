"""
Stub-friendly ToonOut-style segmentation and background handling.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from anime_pipeline.utils.logging_utils import setup_logging
from anime_pipeline.utils.path_utils import ensure_dir

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class SegmentConfig:
    frames_dir: str = "data_frames/frames"
    detections_path: str = "metadata/detections.parquet"
    output_fg_dir: str = "data_fg/rgba"
    output_bg_dir: str = "data_bg/with_holes"
    output_fg_metadata_path: str = "metadata/fg.parquet"
    output_bg_metadata_path: str = "metadata/bg.parquet"
    model_path: Optional[str] = None  # e.g., /mnt/c/ai_models/toonout.engine or .onnx
    backend: str = "stub"  # stub | pytorch | onnx | tensorrt
    device: str = "cpu"
    precision: str = "fp32"  # fp16/bf16 if backend supports
    use_stub: bool = True
    log_dir: Optional[str] = "logs"


def segment_foreground_background(config: SegmentConfig, logger=None) -> Tuple[List[Dict], List[Dict]]:
    """
    Produce foreground cutouts and background-with-holes; stub mode draws simple boxes.
    """
    logger = logger or setup_logging("segment_fg_bg", config.log_dir)
    use_stub = config.use_stub or config.backend == "stub"
    frames_dir = Path(config.frames_dir)
    if not frames_dir.exists():
        logger.warning("Frames dir %s missing.", frames_dir)
        return [], []

    detections = _load_detections(config.detections_path, logger)
    if not detections:
        logger.info("No detections found; generating full-frame stubs.")
        detections = _build_full_frame_detections(frames_dir)

    fg_dir = ensure_dir(config.output_fg_dir)
    bg_dir = ensure_dir(config.output_bg_dir)

    model = None
    if not use_stub:
        try:
            model = _load_segmenter(config, logger)
        except Exception as exc:  # pragma: no cover - requires real deps
            logger.warning("Falling back to stub segmentation because model load failed: %s", exc)
            use_stub = True

    fg_records: List[Dict] = []
    bg_records: List[Dict] = []

    for det in detections:
        frame_path = Path(det["image_path"])
        frame_id = det["frame_id"]
        fg_id = f"{det['det_id']}_fg"
        bg_id = f"{det['frame_id']}_bg"

        rgba_path = fg_dir / f"{fg_id}.png"
        mask_path = fg_dir / f"{fg_id}_mask.png"
        bg_path = bg_dir / f"{bg_id}.png"

        width, height = _infer_image_size(frame_path)
        bbox = (
            int(det["bbox_x1"]),
            int(det["bbox_y1"]),
            int(det["bbox_x2"]),
            int(det["bbox_y2"]),
        )

        if use_stub:
            _write_stub_fg(frame_path, rgba_path, mask_path, bbox, width, height)
            _write_stub_bg(frame_path, bg_path, bbox, width, height)
        else:
            _run_segmenter(model, frame_path, rgba_path, mask_path, bg_path, bbox, width, height, logger)

        fg_records.append(
            {
                "fg_id": fg_id,
                "det_id": det["det_id"],
                "frame_id": frame_id,
                "video_id": det.get("video_id"),
                "rgba_path": str(rgba_path),
                "mask_path": str(mask_path),
                "width": width,
                "height": height,
                "quality_score": 1.0,
            }
        )
        bg_records.append(
            {
                "bg_id": bg_id,
                "frame_id": frame_id,
                "video_id": det.get("video_id"),
                "with_holes_path": str(bg_path),
                "inpainted_path": None,
                "width": width,
                "height": height,
            }
        )

    fg_path = _write_records(fg_records, config.output_fg_metadata_path, logger)
    bg_path = _write_records(bg_records, config.output_bg_metadata_path, logger)
    logger.info("Wrote %d fg records -> %s", len(fg_records), fg_path)
    logger.info("Wrote %d bg records -> %s", len(bg_records), bg_path)
    return fg_records, bg_records


def _load_detections(path: str | Path, logger) -> List[Dict]:
    path = Path(path)
    if not path.exists() and path.suffix == ".parquet":
        alt = path.with_suffix(".csv")
        if alt.exists():
            path = alt
    if not path.exists():
        return []
    try:
        import pandas as pd

        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read detections at %s due to %s", path, exc)
        return []


def _build_full_frame_detections(frames_dir: Path) -> List[Dict]:
    detections: List[Dict] = []
    for frame_path in sorted(frames_dir.glob("**/*")):
        if frame_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        video_id = frame_path.parent.name
        frame_id = f"{video_id}_{frame_path.stem}"
        width, height = _infer_image_size(frame_path)
        det_id = f"{frame_id}_d0"
        detections.append(
            {
                "det_id": det_id,
                "frame_id": frame_id,
                "video_id": video_id,
                "class_id": 0,
                "class_name": "person",
                "score": 1.0,
                "bbox_x1": 0,
                "bbox_y1": 0,
                "bbox_x2": width,
                "bbox_y2": height,
                "track_id": f"{video_id}_t0",
                "image_path": str(frame_path),
                "frame_index": 0,
            }
        )
    return detections


def _infer_image_size(path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return 512, 512


def _write_stub_fg(frame_path: Path, rgba_path: Path, mask_path: Path, bbox: Tuple[int, int, int, int], width: int, height: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        rgba_path.write_text("fg stub", encoding="utf-8")
        mask_path.write_text("mask stub", encoding="utf-8")
        return

    if frame_path.exists():
        with Image.open(frame_path) as img:
            base = img.convert("RGBA")
    else:
        base = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))

    mask = Image.new("L", base.size, color=0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=255)

    fg = Image.new("RGBA", base.size, color=(0, 0, 0, 0))
    fg.paste(base, mask=mask)

    fg.save(rgba_path)
    mask.save(mask_path)


def _write_stub_bg(frame_path: Path, bg_path: Path, bbox: Tuple[int, int, int, int], width: int, height: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        bg_path.write_text("bg stub", encoding="utf-8")
        return

    if frame_path.exists():
        with Image.open(frame_path) as img:
            base = img.convert("RGBA")
    else:
        base = Image.new("RGBA", (width, height), color=(128, 128, 128, 255))

    draw = ImageDraw.Draw(base)
    draw.rectangle(bbox, fill=(0, 0, 0, 0))
    base.save(bg_path)


def _load_segmenter(config: SegmentConfig, logger):
    if config.backend == "onnx":
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            logger.warning("onnxruntime not available, using stub: %s", exc)
            return (config.backend, None)
        if not config.model_path or not Path(config.model_path).exists():
            logger.warning("Segmentation model_path missing; using stub.")
            return (config.backend, None)
        providers = ["CPUExecutionProvider"] if config.device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Loading segmentation ONNX from %s with providers=%s", config.model_path, providers)
        session = ort.InferenceSession(config.model_path, providers=providers)
        return (config.backend, session)
    if config.backend == "pytorch":
        logger.info("PyTorch segmentation backend requested; using stub until model is available.")
        return (config.backend, None)
    if config.backend == "tensorrt":
        if not config.model_path or not Path(config.model_path).exists():
            logger.warning("Segmentation TensorRT engine missing; using stub.")
            return (config.backend, None)
        try:
            import tensorrt as trt  # type: ignore

            logger.info("Loading segmentation TensorRT engine from %s", config.model_path)
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(config.model_path, "rb") as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
            context = engine.create_execution_context() if engine else None
            logger.warning("Segmentation TensorRT forward not implemented; using stub outputs.")
            return (config.backend, context)
        except Exception as exc:
            logger.warning("Segmentation TensorRT load failed (%s); using stub.", exc)
            return (config.backend, None)
    logger.info("Segmentation backend %s requested; defaulting to stub.", config.backend)
    return (config.backend, None)


def _run_segmenter(model, frame_path: Path, rgba_path: Path, mask_path: Path, bg_path: Path, bbox: Tuple[int, int, int, int], width: int, height: int, logger) -> None:
    backend, session = model
    if session is None:
        _write_stub_fg(frame_path, rgba_path, mask_path, bbox, width, height)
        _write_stub_bg(frame_path, bg_path, bbox, width, height)
        return
    if backend == "onnx":
        try:
            input_tensor = _preprocess_image(frame_path, session)
            outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
            mask = _postprocess_mask(outputs[0], width, height)
            _apply_mask(frame_path, rgba_path, mask_path, bg_path, mask)
            return
        except Exception as exc:
            logger.warning("Segmentation ONNX failed (%s); using stub.", exc)
    elif backend == "tensorrt":
        try:
            input_tensor = _preprocess_image(frame_path, session)
            output = _run_trt_inference(session, input_tensor)
            if output is not None:
                mask = _postprocess_mask(output, width, height)
                _apply_mask(frame_path, rgba_path, mask_path, bg_path, mask)
                return
        except Exception as exc:
            logger.warning("Segmentation TensorRT failed (%s); using stub.", exc)
    # Fallback
    _write_stub_fg(frame_path, rgba_path, mask_path, bbox, width, height)
    _write_stub_bg(frame_path, bg_path, bbox, width, height)


def _preprocess_image(frame_path: Path, session) -> np.ndarray:
    input_shape = session.get_inputs()[0].shape  # type: ignore[attr-defined]
    _, _, h, w = input_shape
    with Image.open(frame_path) as img:
        img = img.convert("RGB")
        img = img.resize((w, h))
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        arr = np.expand_dims(arr, 0)
    return arr


def _postprocess_mask(output: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Expect output mask logits/probabilities. If shape is (1,1,H,W) or (1,H,W), squeeze and resize to target size.
    """
    if output.ndim == 4 and output.shape[1] == 1:
        output = output[0, 0]
    elif output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    mask = output
    # Normalize 0-1
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((width, height))
    return np.asarray(mask_img).astype(np.float32) / 255.0


def _apply_mask(frame_path: Path, rgba_path: Path, mask_path: Path, bg_path: Path, mask: np.ndarray) -> None:
    try:
        with Image.open(frame_path) as img:
            img = img.convert("RGBA")
            alpha = Image.fromarray((mask * 255).astype(np.uint8))
            img.putalpha(alpha)
            rgba_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(rgba_path)
            alpha.save(mask_path)

            # background with holes
            bg = Image.new("RGBA", img.size, color=(0, 0, 0, 0))
            inv_alpha = Image.fromarray(255 - (mask * 255).astype(np.uint8))
            bg_rgba = Image.new("RGBA", img.size, color=(0, 0, 0, 0))
            bg_rgba.putalpha(inv_alpha)
            bg_rgba.save(bg_path)
    except Exception:
        _write_stub_fg(frame_path, rgba_path, mask_path, (0, 0, img.width, img.height), img.width, img.height)
        _write_stub_bg(frame_path, bg_path, (0, 0, img.width, img.height), img.width, img.height)


def _run_trt_inference(context, input_array: np.ndarray) -> Optional[np.ndarray]:
    try:
        import tensorrt as trt  # type: ignore
        import pycuda.driver as cuda  # type: ignore
        import pycuda.autoinit  # type: ignore  # noqa: F401
    except Exception:
        return None

    engine = context.engine
    if engine.num_bindings < 2:
        return None
    input_idx = 0
    output_idx = 1
    context.set_binding_shape(input_idx, tuple(int(x) for x in input_array.shape))

    d_input = cuda.mem_alloc(input_array.nbytes)
    output_shape = tuple(int(x) for x in context.get_binding_shape(output_idx))
    host_output = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(host_output.nbytes)
    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, input_array, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, d_output, stream)
    stream.synchronize()
    return host_output


def _write_records(records: List[Dict], output_path: str | Path, logger) -> Path:
    target_path = Path(output_path)
    ensure_dir(target_path.parent)
    if not records:
        target_path.touch()
        return target_path
    try:
        import pandas as pd

        df = pd.DataFrame(records)
        if target_path.suffix == ".parquet":
            df.to_parquet(target_path, index=False)
        else:
            df.to_csv(target_path, index=False)
        return target_path
    except Exception as exc:  # pragma: no cover
        logger.warning("Falling back to CSV due to %s", exc)
        import csv

        csv_path = target_path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        return csv_path


# ============================================================================
# Phase 3.2: Per-Track Segmentation Functions
# ============================================================================

def segment_foreground_background_per_track(
    config: SegmentConfig,
    track_groups: Dict[str, List[Dict]],
    logger=None
) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    """
    Process each track independently to maintain character identity.

    This is the NEW interface for Phase 3 multi-character handling.
    Segments foreground/background for each tracked character separately.

    Args:
        config: Segmentation configuration
        track_groups: Dict mapping track_id to list of detections
                     (output from run_yolo_tracking_with_grouping)
        logger: Logger instance

    Returns:
        Dict mapping track_id to (fg_records, bg_records) tuple

    Example:
        >>> from anime_pipeline.detection.yolo_detector import run_yolo_tracking_with_grouping, YoloTrackingConfig
        >>> from anime_pipeline.segmentation.toonout_wrapper import segment_foreground_background_per_track, SegmentConfig
        >>>
        >>> # Step 1: Get track groups
        >>> track_config = YoloTrackingConfig(frames_dir="/path/to/frames")
        >>> track_groups = run_yolo_tracking_with_grouping(track_config)
        >>>
        >>> # Step 2: Segment each track
        >>> seg_config = SegmentConfig(frames_dir="/path/to/frames")
        >>> track_segments = segment_foreground_background_per_track(seg_config, track_groups)
        >>>
        >>> # Result: Each track_id has its own fg/bg segmentation
        >>> for track_id, (fg_recs, bg_recs) in track_segments.items():
        ...     print(f"{track_id}: {len(fg_recs)} foreground instances")
    """
    logger = logger or setup_logging("segment_per_track", config.log_dir)

    logger.info(f"Processing {len(track_groups)} tracks independently...")

    results = {}

    for track_id, detections in track_groups.items():
        logger.info(f"  Processing track {track_id}: {len(detections)} detections")

        # Create a temporary config with detections override
        # We pass the detections for this specific track only
        track_config = SegmentConfig(
            frames_dir=config.frames_dir,
            detections_path=config.detections_path,
            output_fg_dir=str(Path(config.output_fg_dir) / track_id),
            output_bg_dir=str(Path(config.output_bg_dir) / track_id),
            output_fg_metadata_path=str(Path(config.output_fg_metadata_path).parent / f"{track_id}_fg.parquet"),
            output_bg_metadata_path=str(Path(config.output_bg_metadata_path).parent / f"{track_id}_bg.parquet"),
            model_path=config.model_path,
            backend=config.backend,
            device=config.device,
            precision=config.precision,
            use_stub=config.use_stub,
            log_dir=config.log_dir
        )

        # Process this track's detections
        fg_records, bg_records = _segment_detections_list(
            track_config,
            detections,
            logger
        )

        # Tag records with track_id for downstream processing
        for rec in fg_records:
            rec["track_id"] = track_id

        for rec in bg_records:
            rec["track_id"] = track_id

        results[track_id] = (fg_records, bg_records)

        logger.info(f"    ✓ Track {track_id}: {len(fg_records)} fg, {len(bg_records)} bg")

    logger.info(f"✅ Per-track segmentation complete: {len(results)} tracks processed")

    return results


def _segment_detections_list(
    config: SegmentConfig,
    detections: List[Dict],
    logger
) -> Tuple[List[Dict], List[Dict]]:
    """
    Internal helper to segment a list of detections.

    This is similar to segment_foreground_background but accepts
    a detections list directly instead of loading from file.

    Args:
        config: Segmentation configuration
        detections: List of detection dictionaries
        logger: Logger instance

    Returns:
        (fg_records, bg_records) tuple
    """
    use_stub = config.use_stub or config.backend == "stub"

    fg_dir = ensure_dir(config.output_fg_dir)
    bg_dir = ensure_dir(config.output_bg_dir)

    model = None
    if not use_stub:
        try:
            model = _load_segmenter(config, logger)
        except Exception as exc:
            logger.warning(f"Falling back to stub segmentation: {exc}")
            use_stub = True

    fg_records: List[Dict] = []
    bg_records: List[Dict] = []

    for det in detections:
        frame_path = Path(det["image_path"])
        frame_id = det["frame_id"]
        fg_id = f"{det['det_id']}_fg"
        bg_id = f"{det['frame_id']}_bg"

        rgba_path = fg_dir / f"{fg_id}.png"
        mask_path = fg_dir / f"{fg_id}_mask.png"
        bg_path = bg_dir / f"{bg_id}.png"

        width, height = _infer_image_size(frame_path)
        bbox = (
            int(det["bbox_x1"]),
            int(det["bbox_y1"]),
            int(det["bbox_x2"]),
            int(det["bbox_y2"]),
        )

        if use_stub:
            _write_stub_fg(frame_path, rgba_path, mask_path, bbox, width, height)
            _write_stub_bg(frame_path, bg_path, bbox, width, height)
        else:
            _run_segmenter(model, frame_path, rgba_path, mask_path, bg_path, bbox, width, height, logger)

        fg_records.append(
            {
                "fg_id": fg_id,
                "det_id": det["det_id"],
                "frame_id": frame_id,
                "video_id": det.get("video_id"),
                "rgba_path": str(rgba_path),
                "mask_path": str(mask_path),
                "width": width,
                "height": height,
                "quality_score": 1.0,
            }
        )
        bg_records.append(
            {
                "bg_id": bg_id,
                "frame_id": frame_id,
                "video_id": det.get("video_id"),
                "with_holes_path": str(bg_path),
                "inpainted_path": None,
                "width": width,
                "height": height,
            }
        )

    # Write metadata for this track
    fg_path = _write_records(fg_records, config.output_fg_metadata_path, logger)
    bg_path = _write_records(bg_records, config.output_bg_metadata_path, logger)

    return fg_records, bg_records
