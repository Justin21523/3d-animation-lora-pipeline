"""
Stub DWpose/OpenPose extractor: generates deterministic keypoints and pose visualizations.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from anime_pipeline.utils.logging_utils import setup_logging
from anime_pipeline.utils.path_utils import ensure_dir


@dataclass
class PoseExtractConfig:
    frames_dir: str = "data_frames/frames"
    detections_path: str = "metadata/detections.parquet"
    output_pose_dir: str = "data_pose/vis"
    output_metadata_path: str = "metadata/poses.parquet"
    pose_type: str = "dwpose"
    model_path: Optional[str] = None  # e.g., /mnt/c/ai_models/dwpose/dwpose.onnx
    backend: str = "stub"  # stub | pytorch | onnx | tensorrt
    device: str = "cpu"
    precision: str = "fp32"
    use_stub: bool = True
    log_dir: Optional[str] = "logs"


def extract_poses(config: PoseExtractConfig, logger=None) -> List[Dict]:
    """
    Generate pose keypoints for each detection. Stub uses deterministic synthetic joints.
    """
    logger = logger or setup_logging("extract_pose", config.log_dir)
    use_stub = config.use_stub or config.backend == "stub"
    detections = _load_detections(config.detections_path, logger)
    if not detections:
        logger.warning("No detections found at %s", config.detections_path)
        return []

    pose_dir = ensure_dir(config.output_pose_dir)
    pose_records: List[Dict] = []

    model = None
    if not use_stub:
        try:
            model = _load_pose_model(config, logger)
        except Exception as exc:  # pragma: no cover - real deps
            logger.warning("Falling back to stub pose due to load failure: %s", exc)
            use_stub = True

    for det in detections:
        frame_path = Path(det["image_path"])
        width, height = _infer_image_size(frame_path)
        if use_stub:
            keypoints = _generate_stub_keypoints(det["det_id"], width, height)
            pose_image_path = pose_dir / f"{det['det_id']}_pose.png"
            _draw_pose(frame_path, pose_image_path, keypoints, width, height)
        else:
            keypoints, pose_image_path = _run_pose_model(model, frame_path, det["det_id"], pose_dir, width, height, logger)

        pose_records.append(
            {
                "pose_id": f"{det['det_id']}_pose",
                "det_id": det["det_id"],
                "frame_id": det.get("frame_id"),
                "video_id": det.get("video_id"),
                "pose_type": config.pose_type,
                "keypoints": json.dumps(keypoints),
                "num_joints": len(keypoints),
                "pose_image_path": str(pose_image_path),
                "quality_score": 1.0,
            }
        )

    meta_path = _write_records(pose_records, config.output_metadata_path, logger)
    logger.info("Wrote %d pose records -> %s", len(pose_records), meta_path)
    return pose_records


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


def _infer_image_size(path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return 512, 512


def _generate_stub_keypoints(det_id: str, width: int, height: int, num_joints: int = 17) -> List[Tuple[float, float, float]]:
    seed_int = int(hashlib.sha1(det_id.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed_int)
    keypoints: List[Tuple[float, float, float]] = []
    for _ in range(num_joints):
        x = rng.uniform(0.2 * width, 0.8 * width)
        y = rng.uniform(0.2 * height, 0.8 * height)
        score = rng.uniform(0.8, 1.0)
        keypoints.append((x, y, score))
    return keypoints


def _draw_pose(frame_path: Path, pose_image_path: Path, keypoints: List[Tuple[float, float, float]], width: int, height: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        pose_image_path.write_text("pose stub", encoding="utf-8")
        return

    if frame_path.exists():
        with Image.open(frame_path) as img:
            base = img.convert("RGB")
    else:
        base = Image.new("RGB", (width, height), color=(0, 0, 0))

    draw = ImageDraw.Draw(base)
    # Simple skeleton: connect consecutive keypoints
    for idx, (x, y, _) in enumerate(keypoints):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0))
        if idx > 0:
            px, py, _ = keypoints[idx - 1]
            draw.line((px, py, x, y), fill=(0, 255, 0), width=2)

    pose_image_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(pose_image_path)


def _load_pose_model(config: PoseExtractConfig, logger):
    if config.backend == "pytorch":
        logger.info("PyTorch pose backend requested; using stub until model is available.")
        return ("pytorch", None)
    if config.backend == "onnx":
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            logger.warning("onnxruntime not available, using stub: %s", exc)
            return ("onnx", None)
        providers = ["CPUExecutionProvider"] if config.device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if not config.model_path:
            logger.warning("model_path missing; using stub.")
            return ("onnx", None)
        logger.info("Loading DWpose ONNX from %s with providers=%s", config.model_path, providers)
        session = ort.InferenceSession(config.model_path, providers=providers)
        return ("onnx", session)
    if config.backend == "tensorrt":
        if not config.model_path or not Path(config.model_path).exists():
            logger.warning("TensorRT pose engine missing; using stub.")
            return ("tensorrt", None)
        try:
            import tensorrt as trt  # type: ignore

            logger.info("Loading TensorRT pose engine from %s", config.model_path)
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(config.model_path, "rb") as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
            context = engine.create_execution_context() if engine else None
            logger.warning("Pose TensorRT forward not implemented; using stub outputs.")
            return ("tensorrt", context)
        except Exception as exc:
            logger.warning("TensorRT pose load failed (%s); using stub.", exc)
            return ("tensorrt", None)
    raise RuntimeError(f"Unsupported backend: {config.backend}")


def _run_pose_model(model_tuple, frame_path: Path, det_id: str, pose_dir: Path, width: int, height: int, logger):
    backend, model = model_tuple
    if model is None:
        keypoints = _generate_stub_keypoints(det_id, width, height)
        pose_image_path = pose_dir / f"{det_id}_pose.png"
        _draw_pose(frame_path, pose_image_path, keypoints, width, height)
        return keypoints, pose_image_path

    try:
        if backend == "onnx":
            input_tensor, sx, sy = _preprocess_image(frame_path, model)
            outputs = model.run(None, {model.get_inputs()[0].name: input_tensor})  # type: ignore[attr-defined]
            keypoints = _postprocess_keypoints(outputs[0], width, height, sx, sy)
        elif backend == "tensorrt":
            input_tensor, sx, sy = _preprocess_image(frame_path, model)
            output = _run_trt_inference(model, input_tensor)
            if output is None:
                keypoints = _generate_stub_keypoints(det_id, width, height)
            else:
                keypoints = _postprocess_keypoints(output, width, height, sx, sy)
        else:
            keypoints = _generate_stub_keypoints(det_id, width, height)
        pose_image_path = pose_dir / f"{det_id}_pose.png"
        _draw_pose(frame_path, pose_image_path, keypoints, width, height)
        return keypoints, pose_image_path
    except Exception as exc:
        logger.warning("Pose inference failed (%s); using stub keypoints.", exc)
        keypoints = _generate_stub_keypoints(det_id, width, height)
        pose_image_path = pose_dir / f"{det_id}_pose.png"
        _draw_pose(frame_path, pose_image_path, keypoints, width, height)
        return keypoints, pose_image_path


def _preprocess_image(frame_path: Path, session) -> Tuple[np.ndarray, float, float]:
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


def _postprocess_keypoints(output: np.ndarray, width: int, height: int, scale_x: float, scale_y: float) -> List[Tuple[float, float, float]]:
    """
    Handle common DWpose ONNX output: (1, K, 3) or (K, 3) with x,y,score.
    If normalized (0-1), scale to image size; otherwise treat as absolute pixels.
    """
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if output.ndim != 2 or output.shape[1] < 3:
        raise ValueError(f"Unexpected pose output shape: {output.shape}")

    keypoints: List[Tuple[float, float, float]] = []
    for x, y, s in output:
        if x <= 1.0 and y <= 1.0:
            kp_x = float(x * width)
            kp_y = float(y * height)
        else:
            kp_x = float(x * scale_x)
            kp_y = float(y * scale_y)
        keypoints.append((kp_x, kp_y, float(s)))
    return keypoints


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
