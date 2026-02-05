"""
Stubbed LoRA training loop for SD-style models (CPU friendly).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from anime_pipeline.utils.logging_utils import setup_logging
from anime_pipeline.utils.path_utils import ensure_dir


@dataclass
class LoRATrainingConfig:
    data_dir: str = "lora_datasets/characters"
    output_dir: str = "checkpoints/lora_sd_stub"
    sd_model_path: Optional[str] = None  # for diffusers training
    train_data_dir: Optional[str] = None  # images for real training
    prompt: str = "1girl, best quality, anime style"
    resolution: int = 512
    image_size: int = 64
    batch_size: int = 2
    learning_rate: float = 1e-3
    max_steps: int = 10
    epochs: int = 1
    num_workers: int = 0
    use_stub: bool = True
    enable_diffusers: bool = False  # set true to attempt real training when sd_model_path is provided
    lora_rank: int = 4
    lora_alpha: int = 8
    dtype: str = "fp16"
    device: str = "cpu"
    stub_dataset_size: int = 8
    log_every_n: int = 5
    save_every_n: int = 0
    log_dir: Optional[str] = "logs"


class _RandomImageDataset(Dataset):
    def __init__(self, length: int, image_size: int):
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        torch.manual_seed(idx)
        x = torch.rand(3, self.image_size, self.image_size)
        y = torch.rand(3, self.image_size, self.image_size)
        return x, y


class _TinyUNet(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        out = self.decoder(encoded)
        return out


def train_lora_sd(config: LoRATrainingConfig, logger=None) -> Dict:
    """
    Run LoRA training. Falls back to stub if diffusers is unavailable or use_stub is True.
    """
    logger = logger or setup_logging("train_lora_sd", config.log_dir)
    if not config.use_stub and config.enable_diffusers and config.sd_model_path:
        try:
            return _train_diffusers_lora(config, logger)
        except Exception as exc:
            logger.warning("Diffusers LoRA training failed (%s); falling back to stub.", exc)

    device = torch.device("cpu")
    dataset = _RandomImageDataset(config.stub_dataset_size, config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    model = _TinyUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    ensure_dir(config.output_dir)
    global_step = 0
    total_steps = config.max_steps if config.max_steps > 0 else math.inf
    last_loss = None

    for epoch in range(config.epochs):
        for batch in dataloader:
            if global_step >= total_steps:
                break
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            global_step += 1

            if config.log_every_n and global_step % config.log_every_n == 0:
                logger.info("step=%d loss=%.6f", global_step, last_loss)

            if config.save_every_n and global_step % config.save_every_n == 0:
                _save_checkpoint(model, optimizer, config.output_dir, global_step, logger)

        if global_step >= total_steps:
            break

    ckpt_path = _save_checkpoint(model, optimizer, config.output_dir, global_step, logger, final=True)
    logger.info("Stub training complete: steps=%d last_loss=%.6f saved=%s", global_step, last_loss, ckpt_path)
    return {"steps": global_step, "last_loss": last_loss, "checkpoint": str(ckpt_path)}


def _save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, output_dir: str | Path, step: int, logger, final: bool = False) -> Path:
    path = Path(output_dir)
    ensure_dir(path)
    suffix = "final" if final else f"step{step:06d}"
    ckpt_path = path / f"lora_stub_{suffix}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
        },
        ckpt_path,
    )
    if logger:
        logger.info("Saved checkpoint to %s", ckpt_path)
    return ckpt_path


def _train_diffusers_lora(config: LoRATrainingConfig, logger):
    """
    Minimal diffusers LoRA skeleton. This is a light placeholder; expects real data/weights.
    """
    try:
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline  # type: ignore
        from diffusers.loaders import AttnProcsLayers  # type: ignore
        from diffusers import LoraConfig  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"diffusers/torch not available: {exc}")

    if not Path(config.sd_model_path).exists():  # type: ignore[arg-type]
        raise RuntimeError(f"sd_model_path not found: {config.sd_model_path}")

    dtype = torch.float16 if config.dtype.lower() == "fp16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        config.sd_model_path, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(config.device)

    # LoRA setup
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    pipe.unet.add_adapter(lora_config)
    attn_procs = AttnProcsLayers(pipe.unet.attn_processors)
    optimizer = torch.optim.Adam(attn_procs.parameters(), lr=config.learning_rate)

    # Dummy dataset: reuse stub dataset for now; replace with real loader when data is present.
    dataset = _RandomImageDataset(config.stub_dataset_size, config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    total_steps = config.max_steps if config.max_steps > 0 else math.inf
    global_step = 0
    last_loss = None
    for epoch in range(config.epochs):
        for inputs, targets in dataloader:
            if global_step >= total_steps:
                break
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)

            with torch.no_grad():
                latents = pipe.vae.encode(targets).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=pipe.text_encoder([""] * latents.shape[0]).last_hidden_state).sample
            loss = nn.MSELoss()(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            last_loss = float(loss.item())
            if config.log_every_n and global_step % config.log_every_n == 0:
                logger.info("[diffusers] step=%d loss=%.6f", global_step, last_loss)
        if global_step >= total_steps:
            break

    ensure_dir(config.output_dir)
    ckpt_path = Path(config.output_dir) / "lora_diffusers_stub.safetensors"
    try:
        pipe.save_lora_weights(config.output_dir)
    except Exception as exc:
        logger.warning("Failed to save LoRA weights (%s); saving torch state dict instead.", exc)
        torch.save(attn_procs.state_dict(), ckpt_path)
    return {"steps": global_step, "last_loss": last_loss, "checkpoint": str(ckpt_path), "mode": "diffusers"}
