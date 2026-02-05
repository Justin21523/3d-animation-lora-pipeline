# Inazuma Eleven — SDXL Identity LoRA (UNet-only) configs

These configs are intended for **identity-only** LoRA training:

- Learn **token / character appearance**
- Leave **motion / composition** to the base model

Outputs are written to: `/mnt/data/training/lora/inazuma_eleven/<character_id>/`

## Run (recommended)

Use the queue launcher:

`python -m scripts.training.inazuma_identity_sdxl_retrain --clean-output-root`

## Notes

- Dataset comes from: `/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/1_<character_id>/`
- Latents (`*_sdxl.npz`) already exist in the dataset directories and will be reused.
- Text encoder outputs are cached to disk (SDXL dual encoders) to reduce VRAM peaks.

