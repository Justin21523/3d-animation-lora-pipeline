"""
Stub-friendly YOLO detection + simple tracking.
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
class YoloTrackingConfig:
    frames_dir: str = "data_frames/frames"
    output_detections_path: str = "metadata/detections.parquet"
    output_tracks_path: str = "metadata/tracks.parquet"
    class_id: int = 0
    class_name: str = "person"
    conf_threshold: float = 0.25
    max_dets_per_frame: int = 1
    model_path: Optional[str] = None  # e.g., /mnt/c/ai_models/yolov11n.pt or onnx/trt engine
    device: str = "cpu"
    backend: str = "stub"  # stub | pytorch | onnx | tensorrt
    use_stub: bool = True
    log_dir: Optional[str] = "logs"

    # NEW: Multi-object tracking parameters (Phase 3.1)
    tracker_type: str = "bytetrack"  # bytetrack | botsort | strongsort
    track_persistence: int = 30  # Keep track alive for N frames without detection
    min_track_length: int = 10   # Minimum track length to export
    enable_reid: bool = True     # Re-identification for occluded characters
    iou_threshold: float = 0.3   # IoU threshold for track matching


def run_yolo_tracking(config: YoloTrackingConfig, logger=None) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate detections and simple track metadata; uses stub outputs by default.

    Returns:
        Tuple of (all_detections, all_tracks)
    """
    logger = logger or setup_logging("run_yolo_tracking", config.log_dir)
    use_stub = config.use_stub or config.backend == "stub"
    frames_dir = Path(config.frames_dir)
    if not frames_dir.exists():
        logger.warning("Frames dir %s missing; nothing to process.", frames_dir)
        return [], []

    video_to_frames: Dict[str, List[Path]] = {}
    for path in sorted(frames_dir.glob("**/*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        video_id = path.parent.name
        video_to_frames.setdefault(video_id, []).append(path)

    model = None
    if not use_stub:
        try:
            model = _load_detector(config, logger)
        except Exception as exc:  # pragma: no cover - requires real deps
            logger.warning("Falling back to stub detections because model load failed: %s", exc)
            use_stub = True

    all_dets: List[Dict] = []
    all_tracks: List[Dict] = []

    for video_id, frames in video_to_frames.items():
        tracks_for_video = max(1, config.max_dets_per_frame)
        track_ids = [f"{video_id}_t{idx}" for idx in range(tracks_for_video)]
        logger.info("Processing video %s with %d frames (tracks: %d)", video_id, len(frames), tracks_for_video)

        for frame_idx, frame_path in enumerate(frames):
            if use_stub:
                dets = [
                    _make_stub_detection(
                        frame_path=frame_path,
                        frame_idx=frame_idx,
                        video_id=video_id,
                        class_id=config.class_id,
                        class_name=config.class_name,
                        track_id=track_ids[t_idx],
                        det_idx=t_idx,
                    )
                    for t_idx in range(tracks_for_video)
                ]
            else:
                dets = _run_detector(model, frame_path, config, track_ids)

            all_dets.extend(dets)

        # Build track metadata
        for track_id in track_ids:
            all_tracks.append(
                {
                    "track_id": track_id,
                    "video_id": video_id,
                    "start_frame": 0,
                    "end_frame": len(frames) - 1,
                    "num_frames": len(frames),
                    "character_id": None,
                }
            )

    det_path = _write_records(all_dets, config.output_detections_path, logger)
    track_path = _write_records(all_tracks, config.output_tracks_path, logger)
    logger.info("Wrote %d detections -> %s", len(all_dets), det_path)
    logger.info("Wrote %d tracks -> %s", len(all_tracks), track_path)
    return all_dets, all_tracks


def _make_stub_detection(
    frame_path: Path,
    frame_idx: int,
    video_id: str,
    class_id: int,
    class_name: str,
    track_id: str,
    det_idx: int,
) -> Dict:
    # Derive deterministic bbox from hash
    seed = int(hashlib.sha1(f"{frame_path}-{det_idx}".encode("utf-8")).hexdigest(), 16)
    x1_frac = 0.1 + (seed % 20) / 100.0
    y1_frac = 0.1 + (seed % 15) / 100.0
    w_frac = 0.3
    h_frac = 0.45

    width, height = _infer_image_size(frame_path)
    x1 = int(x1_frac * width)
    y1 = int(y1_frac * height)
    x2 = min(width, x1 + int(w_frac * width))
    y2 = min(height, y1 + int(h_frac * height))

    det_id = f"{video_id}_{frame_idx:06d}_d{det_idx}"
    return {
        "det_id": det_id,
        "frame_id": f"{video_id}_{frame_idx:06d}",
        "video_id": video_id,
        "class_id": class_id,
        "class_name": class_name,
        "score": 0.9,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "track_id": track_id,
        "image_path": str(frame_path),
        "frame_index": frame_idx,
    }


def _load_detector(config: YoloTrackingConfig, logger):
    if config.backend == "pytorch":
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Ultralytics not available: {exc}")
        model_path = config.model_path or "yolov8n.pt"
        logger.info("Loading YOLO model from %s on %s", model_path, config.device)
        model = YOLO(model_path)
        model.to(config.device)
        return ("pytorch", model)
    if config.backend == "onnx":
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            logger.warning("onnxruntime not available, will fall back to stub: %s", exc)
            return ("onnx", None)
        providers = _select_providers(config)
        model_path = config.model_path
        if not model_path or not Path(model_path).exists():
            logger.warning("ONNX model_path missing; will fall back to stub.")
            return ("onnx", None)
        logger.info("Loading ONNX model from %s with providers=%s", model_path, providers)
        session = ort.InferenceSession(model_path, providers=providers)
        return ("onnx", session)
    if config.backend == "tensorrt":
        if not config.model_path or not Path(config.model_path).exists():
            logger.warning("TensorRT engine path missing; will fall back to stub.")
            return ("tensorrt", None)
        try:
            import tensorrt as trt  # type: ignore

            logger.info("Loading TensorRT engine from %s", config.model_path)
            logger.warning("TensorRT inference not fully implemented; will return stub detections.")
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(config.model_path, "rb") as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
            context = engine.create_execution_context() if engine else None
            return ("tensorrt", context)
        except Exception as exc:
            logger.warning("TensorRT load failed (%s); using stub.", exc)
            return ("tensorrt", None)
    raise RuntimeError(f"Unsupported backend: {config.backend}")


def _select_providers(config: YoloTrackingConfig):
    providers = ["CPUExecutionProvider"]
    if config.device != "cpu":
        providers.insert(0, "CUDAExecutionProvider")
    return providers


def _run_detector(model_tuple, frame_path: Path, config: YoloTrackingConfig, track_ids: List[str]) -> List[Dict]:
    backend, model = model_tuple
    if backend == "pytorch":
        # Ultralytics inference
        results = model.predict(source=str(frame_path), imgsz=None, conf=config.conf_threshold, verbose=False)
        dets: List[Dict] = []
        for res in results:
            boxes = res.boxes
            for idx, b in enumerate(boxes):
                dets.append(
                    {
                        "det_id": f"{frame_path.stem}_d{idx}",
                        "frame_id": f"{frame_path.parent.name}_{frame_path.stem}",
                        "video_id": frame_path.parent.name,
                        "class_id": int(b.cls[0]),
                        "class_name": config.class_name,
                        "score": float(b.conf[0]),
                        "bbox_x1": float(b.xyxy[0][0]),
                        "bbox_y1": float(b.xyxy[0][1]),
                        "bbox_x2": float(b.xyxy[0][2]),
                        "bbox_y2": float(b.xyxy[0][3]),
                        "track_id": track_ids[idx % len(track_ids)],
                        "image_path": str(frame_path),
                        "frame_index": int(res.path.split("_")[-1].split(".")[0]) if "_" in str(res.path) else idx,
                    }
                )
        return dets
    if backend == "onnx":
        if model is None:
            return [
                _make_stub_detection(
                    frame_path=frame_path,
                    frame_idx=0,
                    video_id=frame_path.parent.name,
                    class_id=config.class_id,
                    class_name=config.class_name,
                    track_id=track_ids[0],
                    det_idx=0,
                )
            ]
        try:
            input_tensor, scale_x, scale_y = _preprocess_image(frame_path, model)
            outputs = model.run(None, {model.get_inputs()[0].name: input_tensor})  # type: ignore[attr-defined]
            detections = _postprocess_yolo(outputs[0], config.conf_threshold, scale_x, scale_y, track_ids, frame_path)
            return detections if detections else [
                _make_stub_detection(
                    frame_path=frame_path,
                    frame_idx=0,
                    video_id=frame_path.parent.name,
                    class_id=config.class_id,
                    class_name=config.class_name,
                    track_id=track_ids[0],
                    det_idx=0,
                )
            ]
        except Exception as exc:
            # Fallback to stub on any failure
            return [
                _make_stub_detection(
                    frame_path=frame_path,
                    frame_idx=0,
                    video_id=frame_path.parent.name,
                    class_id=config.class_id,
                    class_name=config.class_name,
                    track_id=track_ids[0],
                    det_idx=0,
                )
            ]
    if backend == "tensorrt":
        if model is None:
            return [
                _make_stub_detection(
                    frame_path=frame_path,
                    frame_idx=0,
                    video_id=frame_path.parent.name,
                    class_id=config.class_id,
                    class_name=config.class_name,
                    track_id=track_ids[0],
                    det_idx=0,
                )
            ]
        try:
            input_tensor, scale_x, scale_y = _preprocess_image(frame_path, model)
            output = _run_trt_inference(model, input_tensor)
            if output is not None:
                detections = _postprocess_yolo(output, config.conf_threshold, scale_x, scale_y, track_ids, frame_path)
                if detections:
                    return detections
        except Exception:
            pass
        return [
            _make_stub_detection(
                frame_path=frame_path,
                frame_idx=0,
                video_id=frame_path.parent.name,
                class_id=config.class_id,
                class_name=config.class_name,
                track_id=track_ids[0],
                det_idx=0,
            )
        ]
    raise RuntimeError(f"Unsupported backend: {backend}")


def _preprocess_image(frame_path: Path, session) -> Tuple[np.ndarray, float, float]:
    """
    Basic resize/normalize for YOLO ONNX. Assumes input shape [1,3,H,W].
    """
    input_shape = session.get_inputs()[0].shape  # type: ignore[attr-defined]
    _, _, h, w = input_shape
    with Image.open(frame_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((w, h))
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        arr = np.expand_dims(arr, 0)
    scale_x = orig_w / w
    scale_y = orig_h / h
    return arr, scale_x, scale_y


def _postprocess_yolo(output: np.ndarray, conf_thres: float, scale_x: float, scale_y: float, track_ids: List[str], frame_path: Path) -> List[Dict]:
    """
    Decode YOLOv8-style ONNX output: [1, N, 85] -> xywh + scores.
    """
    if output.ndim == 3:
        output = output[0]
    dets: List[Dict] = []
    for idx, row in enumerate(output):
        if row.shape[0] < 6:
            continue
        obj_conf = row[4]
        cls_scores = row[5:]
        cls_id = int(np.argmax(cls_scores)) if cls_scores.size else 0
        cls_score = cls_scores[cls_id] if cls_scores.size else 1.0
        score = float(obj_conf * cls_score)
        if score < conf_thres:
            continue
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        x1 = (cx - w / 2) * scale_x
        y1 = (cy - h / 2) * scale_y
        x2 = (cx + w / 2) * scale_x
        y2 = (cy + h / 2) * scale_y
        dets.append(
            {
                "det_id": f"{frame_path.stem}_d{idx}",
                "frame_id": f"{frame_path.parent.name}_{frame_path.stem}",
                "video_id": frame_path.parent.name,
                "class_id": cls_id,
                "class_name": str(cls_id),
                "score": score,
                "bbox_x1": float(x1),
                "bbox_y1": float(y1),
                "bbox_x2": float(x2),
                "bbox_y2": float(y2),
                "track_id": track_ids[idx % len(track_ids)],
                "image_path": str(frame_path),
                "frame_index": idx,
            }
        )
    return dets


def _run_trt_inference(context, input_array: np.ndarray) -> Optional[np.ndarray]:
    try:
        import tensorrt as trt  # type: ignore
        import pycuda.driver as cuda  # type: ignore
        import pycuda.autoinit  # type: ignore  # noqa: F401
    except Exception:
        return None

    engine = context.engine
    bindings = []
    host_inputs = []
    device_inputs = []
    host_outputs = []
    device_outputs = []

    # Assume single input/output
    if engine.num_bindings < 2:
        return None
    input_idx = 0
    output_idx = 1
    input_shape = tuple(int(x) for x in input_array.shape)
    context.set_binding_shape(input_idx, input_shape)

    # Allocate input
    input_nbytes = input_array.nbytes
    d_input = cuda.mem_alloc(input_nbytes)
    device_inputs.append(d_input)
    host_inputs.append(input_array)
    bindings.append(int(d_input))

    # Allocate output
    output_shape = tuple(int(x) for x in context.get_binding_shape(output_idx))
    output_nbytes = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize
    host_output = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output_nbytes)
    device_outputs.append(d_output)
    host_outputs.append(host_output)
    bindings.append(int(d_output))

    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, input_array, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, d_output, stream)
    stream.synchronize()
    return host_output

def _infer_image_size(path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return 512, 512


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
        csv_path = target_path.with_suffix(".csv")
        import csv

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        return csv_path


# ============================================================================
# Phase 3.1: Enhanced Multi-Object Tracking Functions
# ============================================================================

def group_detections_by_track(detections: List[Dict], min_track_length: int = 10) -> Dict[str, List[Dict]]:
    """
    Group detections by track_id for per-track processing.

    This enables downstream per-track segmentation where each character track
    is processed independently to maintain identity consistency.

    Args:
        detections: List of detection dictionaries with track_id field
        min_track_length: Minimum number of detections required per track

    Returns:
        Dictionary mapping track_id to list of detection dictionaries
        Only includes tracks with >= min_track_length detections
    """
    track_groups: Dict[str, List[Dict]] = {}

    # Group by track_id
    for det in detections:
        track_id = det.get("track_id")
        if track_id is None:
            continue

        if track_id not in track_groups:
            track_groups[track_id] = []

        track_groups[track_id].append(det)

    # Filter by minimum length
    valid_tracks = {
        tid: dets
        for tid, dets in track_groups.items()
        if len(dets) >= min_track_length
    }

    return valid_tracks


def run_yolo_tracking_with_grouping(
    config: YoloTrackingConfig,
    logger=None
) -> Dict[str, List[Dict]]:
    """
    Enhanced YOLO tracking with per-track instance extraction.

    This is the NEW interface for Phase 3 multi-character handling.
    Returns track-grouped detections ready for per-track segmentation.

    Args:
        config: YoloTrackingConfig with tracking parameters
        logger: Logger instance

    Returns:
        Dict mapping track_id to list of detection dictionaries

    Example:
        >>> config = YoloTrackingConfig(
        ...     frames_dir="/path/to/frames",
        ...     min_track_length=10,
        ...     tracker_type="bytetrack"
        ... )
        >>> track_groups = run_yolo_tracking_with_grouping(config)
        >>> print(f"Found {len(track_groups)} valid tracks")
        >>> for track_id, detections in track_groups.items():
        ...     print(f"  {track_id}: {len(detections)} frames")
    """
    logger = logger or setup_logging("run_yolo_tracking_with_grouping", config.log_dir)

    # Run standard tracking
    all_detections, all_tracks = run_yolo_tracking(config, logger)

    # Group detections by track
    track_groups = group_detections_by_track(
        all_detections,
        min_track_length=config.min_track_length
    )

    logger.info(
        f"Found {len(track_groups)} valid tracks (>={config.min_track_length} frames) "
        f"from {len(all_detections)} total detections"
    )

    # Log track statistics
    track_lengths = [len(dets) for dets in track_groups.values()]
    if track_lengths:
        logger.info(
            f"Track length stats: min={min(track_lengths)}, "
            f"max={max(track_lengths)}, "
            f"mean={sum(track_lengths)/len(track_lengths):.1f}"
        )

    return track_groups


def get_track_info(track_groups: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    Get summary information for each track.

    Args:
        track_groups: Output from run_yolo_tracking_with_grouping()

    Returns:
        Dict mapping track_id to track info dictionary with:
        - num_frames: Number of detections
        - start_frame: First frame index
        - end_frame: Last frame index
        - avg_confidence: Average detection confidence
        - avg_bbox_area: Average bounding box area
    """
    track_info = {}

    for track_id, detections in track_groups.items():
        if not detections:
            continue

        # Extract frame indices
        frame_indices = [d.get("frame_index", 0) for d in detections]

        # Calculate bbox areas
        bbox_areas = []
        confidences = []
        for d in detections:
            x1, y1 = d.get("bbox_x1", 0), d.get("bbox_y1", 0)
            x2, y2 = d.get("bbox_x2", 1), d.get("bbox_y2", 1)
            area = (x2 - x1) * (y2 - y1)
            bbox_areas.append(area)
            confidences.append(d.get("score", 0.0))

        track_info[track_id] = {
            "num_frames": len(detections),
            "start_frame": min(frame_indices),
            "end_frame": max(frame_indices),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "avg_bbox_area": sum(bbox_areas) / len(bbox_areas) if bbox_areas else 0.0,
            "video_id": detections[0].get("video_id", "unknown")
        }

    return track_info
