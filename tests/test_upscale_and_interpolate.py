from pathlib import Path

from anime_pipeline.restoration.realesrgan_wrapper import RealESRGANConfig, upscale_frames
from anime_pipeline.interpolation.rife_wrapper import RIFEConfig, interpolate_frames


def test_upscale_and_interpolate_stub(tmp_path):
    # Prepare dummy frames
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True)
    try:
        from PIL import Image
    except ImportError:
        for i in range(2):
            (frames_dir / f"frame_{i:06d}.png").write_text("stub frame", encoding="utf-8")
    else:
        for i in range(2):
            img = Image.new("RGB", (16, 16), color=(i * 50, 0, 0))
            img.save(frames_dir / f"frame_{i:06d}.png")

    # Upscale
    up_cfg = RealESRGANConfig(
        input_dir=frames_dir,
        output_dir=tmp_path / "upscaled",
        scale=2,
        use_stub=True,
        log_dir=tmp_path / "logs",
    )
    up_records = upscale_frames(up_cfg)
    assert up_records
    for rec in up_records:
        assert Path(rec["output_path"]).exists()

    # Interpolate
    interp_cfg = RIFEConfig(
        input_dir=frames_dir,
        output_dir=tmp_path / "interp",
        times=2,
        use_stub=True,
        log_dir=tmp_path / "logs",
    )
    interp_records = interpolate_frames(interp_cfg)
    assert interp_records
    for rec in interp_records:
        assert Path(rec["output"]).exists()
