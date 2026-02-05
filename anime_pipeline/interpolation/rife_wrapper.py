"""
Stub RIFE interpolation wrapper (CPU friendly).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from anime_pipeline.utils.logging_utils import setup_logging
from anime_pipeline.utils.path_utils import ensure_dir


@dataclass
class RIFEConfig:
    input_dir: str = "outputs/animation/frames"
    output_dir: str = "outputs/animation/interpolated"
    times: int = 2  # 2x -> insert 1 frame between each pair
    model_path: Optional[str] = None  # e.g., /mnt/c/ai_models/interpolation/rife.onnx
    backend: str = "stub"  # stub | pytorch | onnx | tensorrt
    device: str = "cpu"
    precision: str = "fp32"
    use_stub: bool = True
    log_dir: Optional[str] = "logs"


def interpolate_frames(config: RIFEConfig, logger=None) -> List[Dict]:
    """
    Interpolate frames; stub duplicates input and inserts simple blends.
    """
    logger = logger or setup_logging("interpolate_frames", config.log_dir)
    use_stub = config.use_stub or config.backend == "stub"
    input_dir = Path(config.input_dir)
    if not input_dir.exists():
        logger.warning("Input dir %s missing.", input_dir)
        return []

    output_dir = ensure_dir(config.output_dir)
    frames = sorted([p for p in input_dir.glob("**/*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    records: List[Dict] = []

    model = None
    if not use_stub:
        try:
            model = _load_interpolator(config, logger)
        except Exception as exc:  # pragma: no cover
            logger.warning("Falling back to stub interpolation because model load failed: %s", exc)
            use_stub = True

    out_idx = 0
    for idx in range(len(frames)):
        frame_path = frames[idx]
        # Always copy original frame
        out_path = output_dir / f"frame_{out_idx:06d}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _copy_frame(frame_path, out_path)
        records.append({"input": str(frame_path), "output": str(out_path), "source_index": idx})
        out_idx += 1

        # Insert intermediates if not last frame
        if idx < len(frames) - 1 and config.times > 1:
            next_frame = frames[idx + 1]
            inserts = config.times - 1
            for ins in range(inserts):
                mid_path = output_dir / f"frame_{out_idx:06d}.png"
                if use_stub:
                    _blend_stub(frame_path, next_frame, mid_path, weight=(ins + 1) / (config.times))
                else:
                    _run_interpolator(model, frame_path, next_frame, mid_path, config)
                records.append({"input": f"{frame_path} -> {next_frame}", "output": str(mid_path), "source_index": idx})
                out_idx += 1

    logger.info("Interpolated %d frames into %d outputs", len(frames), len(records))
    return records


def _copy_frame(src: Path, dst: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        dst.write_text("frame copy stub", encoding="utf-8")
        return
    with Image.open(src) as img:
        img.save(dst)


def _blend_stub(src_a: Path, src_b: Path, dst: Path, weight: float) -> None:
    try:
        from PIL import Image
    except ImportError:
        dst.write_text("interpolated stub", encoding="utf-8")
        return
    with Image.open(src_a) as a, Image.open(src_b) as b:
        a = a.convert("RGB")
        b = b.convert("RGB")
        blended = Image.blend(a, b, weight)
        blended.save(dst)


def _load_interpolator(config: RIFEConfig, logger):
    if config.backend == "onnx":
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            logger.warning("onnxruntime not available, using stub: %s", exc)
            return (config.backend, None)
        if not config.model_path or not Path(config.model_path).exists():
            logger.warning("RIFE model_path missing; using stub.")
            return (config.backend, None)
        providers = ["CPUExecutionProvider"] if config.device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Loading RIFE ONNX from %s with providers=%s", config.model_path, providers)
        session = ort.InferenceSession(config.model_path, providers=providers)
        return (config.backend, session)
    if config.backend == "pytorch":
        logger.info("PyTorch interpolator backend requested; using stub until model is available.")
        return (config.backend, None)
    if config.backend == "tensorrt":
        if not config.model_path or not Path(config.model_path).exists():
            logger.warning("Interpolator TensorRT engine missing; using stub.")
            return (config.backend, None)
        try:
            import tensorrt as trt  # type: ignore

            logger.info("Loading RIFE TensorRT engine from %s", config.model_path)
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(config.model_path, "rb") as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
            context = engine.create_execution_context() if engine else None
            logger.warning("Interpolator TensorRT forward not implemented; using stub.")
            return (config.backend, context)
        except Exception as exc:
            logger.warning("RIFE TensorRT load failed (%s); using stub.", exc)
            return (config.backend, None)
    logger.info("Interpolation backend %s requested; using stub.", config.backend)
    return (config.backend, None)


def _run_interpolator(model, frame_a: Path, frame_b: Path, dst: Path, config: RIFEConfig) -> None:
    backend, session = model
    if session is None:
        _blend_stub(frame_a, frame_b, dst, weight=0.5)
        return
    if backend == "onnx":
        try:
            input_tensor = _preprocess_pair(frame_a, frame_b, session)
            outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
            _postprocess_frame(outputs[0], dst, frame_a)
            return
        except Exception:
            _blend_stub(frame_a, frame_b, dst, weight=0.5)
            return
    elif backend == "tensorrt":
        try:
            input_tensor = _preprocess_pair(frame_a, frame_b, session)
            output = _run_trt_inference(session, input_tensor)
            if output is not None:
                _postprocess_frame(output, dst, frame_a)
                return
        except Exception:
            pass
    _blend_stub(frame_a, frame_b, dst, weight=0.5)


def _preprocess_pair(frame_a: Path, frame_b: Path, session) -> np.ndarray:
    input_shape = session.get_inputs()[0].shape  # type: ignore[attr-defined]
    _, _, h, w = input_shape
    imgs = []
    for frame in (frame_a, frame_b):
        with Image.open(frame) as img:
            img = img.convert("RGB").resize((w, h))
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            imgs.append(arr)
    stacked = np.concatenate(imgs, axis=0)  # [6, H, W]
    return np.expand_dims(stacked, 0)


def _postprocess_frame(output: np.ndarray, dst: Path, ref_frame: Path) -> None:
    if output.ndim == 4:
        output = output[0]
    if output.shape[0] in (3, 4):
        output = np.transpose(output, (1, 2, 0))
    output = np.clip(output, 0, 1)
    img = Image.fromarray((output * 255).astype(np.uint8))
    with Image.open(ref_frame) as ref:
        img = img.resize(ref.size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


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
