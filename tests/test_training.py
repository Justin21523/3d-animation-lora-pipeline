from pathlib import Path

from anime_pipeline.training.lora_trainer_sd import LoRATrainingConfig, train_lora_sd


def test_train_lora_sd_stub(tmp_path):
    cfg = LoRATrainingConfig(
        data_dir=tmp_path / "data",
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
    result = train_lora_sd(cfg)
    ckpt = Path(result["checkpoint"])
    assert ckpt.exists()
    assert result["steps"] == 2
