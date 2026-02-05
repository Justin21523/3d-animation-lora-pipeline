"""
Stub Real-ESRGAN upscale wrapper (CPU friendly).
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
class RealESRGANConfig:
    input_dir: str = "data_frames/frames"
    output_dir: str = "data_restored/frames"
    scale: int = 2
    model_path: Optional[str] = None  # e.g., /mnt/c/ai_models/upscale/realesrgan-anime.onnx
    backend: str = "stub"  # stub | pytorch | onnx | tensorrt
    device: str = "cpu"
    precision: str = "fp32"
    use_stub: bool = True
    log_dir: Optional[str] = "logs"


def upscale_frames(config: RealESRGANConfig, logger=None) -> List[Dict]:
    """
    Upscale frames; stub uses Pillow resize.
    """
    logger = logger or setup_logging("upscale_frames", config.log_dir)
    use_stub = config.use_stub or config.backend == "stub"

    input_dir = Path(config.input_dir)
    if not input_dir.exists():
        logger.warning("Input dir %s missing.", input_dir)
        return []

    output_dir = ensure_dir(config.output_dir)
    records: List[Dict] = []

    model = None
    if not use_stub:
        try:
            model = _load_upscaler(config, logger)
        except Exception as exc:  # pragma: no cover - requires real deps
            logger.warning("Falling back to stub upscale because model load failed: %s", exc)
            use_stub = True

    for frame_path in sorted(input_dir.glob("**/*")):
        if frame_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        rel = frame_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if use_stub:
            _resize_stub(frame_path, out_path, config.scale)
        else:
            _run_upscaler(model, frame_path, out_path, config)

        records.append(
            {
                "input_path": str(frame_path),
                "output_path": str(out_path),
                "scale": config.scale,
                "backend": "stub" if use_stub else config.backend,
            }
        )

    logger.info("Upscaled %d frames to %s", len(records), output_dir)
    return records


def _resize_stub(src: Path, dst: Path, scale: int) -> None:
    try:
        from PIL import Image
    except ImportError:
        dst.write_text("upscaled stub", encoding="utf-8")
        return
    with Image.open(src) as img:
        new_size = (img.width * scale, img.height * scale)
        up = img.resize(new_size, resample=Image.BICUBIC)
        up.save(dst)


def _load_upscaler(config: RealESRGANConfig, logger):
    if config.backend == "onnx":
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            logger.warning("onnxruntime not available, using stub: %s", exc)
            return (config.backend, None)
        if not config.model_path or not Path(config.model_path).exists():
            logger.warning("Upscale model_path missing; using stub.")
            return (config.backend, None)
        providers = ["CPUExecutionProvider"] if config.device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Loading Real-ESRGAN ONNX from %s with providers=%s", config.model_path, providers)
        session = ort.InferenceSession(config.model_path, providers=providers)
        return (config.backend, session)
    if config.backend == "pytorch":
        logger.info("PyTorch upscaler backend requested; using stub until model is available.")
        return (config.backend, None)
    if config.backend == "tensorrt":
        if not config.model_path or not Path(config.model_path).exists():
            logger.warning("Upscale TensorRT engine missing; using stub.")
            return (config.backend, None)
        try:
            import tensorrt as trt  # type: ignore

            logger.info("Loading upscaler TensorRT engine from %s", config.model_path)
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(config.model_path, "rb") as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
            context = engine.create_execution_context() if engine else None
            logger.warning("Upscaler TensorRT forward not implemented; using stub resize.")
            return (config.backend, context)
        except Exception as exc:
            logger.warning("Upscaler TensorRT load failed (%s); using stub.", exc)
            return (config.backend, None)
    logger.info("Upscaler backend %s requested; using stub.", config.backend)
    return (config.backend, None)


def _run_upscaler(model, src: Path, dst: Path, config: RealESRGANConfig) -> None:
    backend, session = model
    if session is None:
        _resize_stub(src, dst, config.scale)
        return
    if backend == "onnx":
        try:
            input_tensor, orig_size = _preprocess_image(src, session)
            outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
            _postprocess_image(outputs[0], dst, orig_size)
            return
        except Exception:
            _resize_stub(src, dst, config.scale)
            return
    elif backend == "tensorrt":
        try:
            input_tensor, orig_size = _preprocess_image(src, session)
            output = _run_trt_inference(session, input_tensor)
            if output is not None:
                _postprocess_image(output, dst, orig_size)
                return
        except Exception:
            pass
    _resize_stub(src, dst, config.scale)


def _preprocess_image(src: Path, session):
    input_shape = session.get_inputs()[0].shape  # type: ignore[attr-defined]
    _, _, h, w = input_shape
    with Image.open(src) as img:
        img = img.convert("RGB")
        orig_size = img.size
        img = img.resize((w, h))
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
    return arr, orig_size


def _postprocess_image(output: np.ndarray, dst: Path, orig_size) -> None:
    if output.ndim == 4:
        output = output[0]
    if output.shape[0] in (3, 4):  # C,H,W
        output = np.transpose(output, (1, 2, 0))
    output = np.clip(output, 0, 1)
    img = Image.fromarray((output * 255).astype(np.uint8))
    img = img.resize(orig_size)
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
