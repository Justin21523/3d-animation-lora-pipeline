from pathlib import Path

from anime_pipeline.training.controlnet_trainer import ControlNetTrainingConfig, train_controlnet_pose


def test_train_controlnet_pose_stub(tmp_path):
    cfg = ControlNetTrainingConfig(
        metadata_path=tmp_path / "controlnet/metadata.parquet",
        output_dir=tmp_path / "checkpoints",
        image_size=32,
        batch_size=2,
        learning_rate=1e-3,
        max_steps=2,
        epochs=1,
        stub_dataset_size=4,
        log_dir=tmp_path / "logs",
        use_stub=True,
    )
    result = train_controlnet_pose(cfg)
    ckpt = Path(result["checkpoint"])
    assert ckpt.exists()
    assert result["steps"] == 2
