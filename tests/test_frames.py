from pathlib import Path

from anime_pipeline.frames.dedupe import DedupeConfig, dedupe_frames
from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames


def test_extract_frames_stub(tmp_path):
    cfg = ExtractFramesConfig(
        input_videos_dir=tmp_path / "raw",
        output_dir=tmp_path / "frames",
        metadata_path=tmp_path / "frames.parquet",
        use_stub=True,
        stub_frame_count=3,
        stub_width=64,
        stub_height=64,
        log_dir=tmp_path / "logs",
    )
    cfg.input_videos_dir.mkdir(parents=True, exist_ok=True)

    records = extract_frames(cfg)
    assert len(records) == 3
    for rec in records:
        assert Path(rec["image_path"]).exists()
        assert rec["width"] == 64
        assert rec["height"] == 64


def test_dedupe_marks_duplicates(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True)

    file_a = frames_dir / "a.png"
    file_a.write_bytes(b"same-content")
    file_b = frames_dir / "b.png"
    file_b.write_bytes(b"same-content")

    cfg = DedupeConfig(
        frames_dir=str(frames_dir),
        output_metadata_path=tmp_path / "dedupe.parquet",
        log_dir=tmp_path / "logs",
    )

    records = dedupe_frames(cfg)
    kept = [r for r in records if r["is_kept_dedupe"]]
    assert len(kept) == 1
    assert any(not r["is_kept_dedupe"] for r in records)
