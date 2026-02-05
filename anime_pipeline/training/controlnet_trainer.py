"""
Stubbed ControlNet pose training loop (CPU friendly).
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
class ControlNetTrainingConfig:
    metadata_path: str = "controlnet_datasets/pose/metadata.parquet"
    output_dir: str = "checkpoints/controlnet_pose_stub"
    sd_model_path: Optional[str] = None
    controlnet_model_path: Optional[str] = None
    train_data_dir: Optional[str] = None
    image_size: int = 64
    batch_size: int = 2
    learning_rate: float = 1e-3
    max_steps: int = 10
    epochs: int = 1
    num_workers: int = 0
    use_stub: bool = True
    enable_diffusers: bool = False
    prompt: str = "pose control"
    resolution: int = 512
    dtype: str = "fp16"
    device: str = "cpu"
    stub_dataset_size: int = 8
    log_every_n: int = 5
    save_every_n: int = 0
    log_dir: Optional[str] = "logs"


class _RandomCondDataset(Dataset):
    def __init__(self, length: int, image_size: int):
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        torch.manual_seed(idx)
        cond = torch.rand(3, self.image_size, self.image_size)
        target = torch.rand(3, self.image_size, self.image_size)
        return cond, target


class _TinyControlNet(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels * 2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
        )

    def forward(self, cond, x):
        stacked = torch.cat([cond, x], dim=1)
        encoded = self.encoder(stacked)
        out = self.decoder(encoded)
        return out


def train_controlnet_pose(config: ControlNetTrainingConfig, logger=None) -> Dict:
    """
    Run ControlNet training. Falls back to stub if diffusers is unavailable or use_stub is True.
    """
    logger = logger or setup_logging("train_controlnet_pose", config.log_dir)
    if not config.use_stub and config.enable_diffusers and config.sd_model_path and config.controlnet_model_path:
        try:
            return _train_diffusers_controlnet(config, logger)
        except Exception as exc:
            logger.warning("Diffusers ControlNet training failed (%s); falling back to stub.", exc)

    device = torch.device("cpu")
    dataset = _RandomCondDataset(config.stub_dataset_size, config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    model = _TinyControlNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    ensure_dir(config.output_dir)
    global_step = 0
    total_steps = config.max_steps if config.max_steps > 0 else math.inf
    last_loss = None

    for epoch in range(config.epochs):
        for cond, target in dataloader:
            if global_step >= total_steps:
                break
            cond = cond.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(cond, target)
            loss = criterion(output, target)
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
    ckpt_path = path / f"controlnet_stub_{suffix}.pt"
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


def _train_diffusers_controlnet(config: ControlNetTrainingConfig, logger):
    """
    Minimal ControlNet diffusers skeleton; expects real data/weights when enable_diffusers is True.
    """
    try:
        import torch
        from diffusers import (  # type: ignore
            ControlNetModel,
            DPMSolverMultistepScheduler,
            StableDiffusionControlNetPipeline,
        )
    except Exception as exc:
        raise RuntimeError(f"diffusers/torch not available: {exc}")

    if not Path(config.sd_model_path).exists():  # type: ignore[arg-type]
        raise RuntimeError(f"sd_model_path not found: {config.sd_model_path}")
    if not Path(config.controlnet_model_path).exists():  # type: ignore[arg-type]
        raise RuntimeError(f"controlnet_model_path not found: {config.controlnet_model_path}")

    dtype = torch.float16 if config.dtype.lower() == "fp16" else torch.float32
    controlnet = ControlNetModel.from_pretrained(config.controlnet_model_path, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.sd_model_path,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(config.device)

    # Dummy dataset placeholder; replace with real conditioned data.
    dataset = _RandomCondDataset(config.stub_dataset_size, config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    optimizer = torch.optim.Adam(pipe.controlnet.parameters(), lr=config.learning_rate)
    total_steps = config.max_steps if config.max_steps > 0 else math.inf
    global_step = 0
    last_loss = None

    for epoch in range(config.epochs):
        for cond, target in dataloader:
            if global_step >= total_steps:
                break
            cond = cond.to(config.device)
            target = target.to(config.device)

            with torch.no_grad():
                latents = pipe.vae.encode(target).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=pipe.text_encoder([""] * latents.shape[0]).last_hidden_state).sample
            cond_latents = cond  # placeholder; in real use, encode control image

            # Simple loss between prediction and noise (placeholder)
            loss = nn.MSELoss()(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = float(loss.item())
            global_step += 1
            if config.log_every_n and global_step % config.log_every_n == 0:
                logger.info("[diffusers-controlnet] step=%d loss=%.6f", global_step, last_loss)
        if global_step >= total_steps:
            break

    ensure_dir(config.output_dir)
    ckpt_path = Path(config.output_dir) / "controlnet_diffusers_stub"
    try:
        controlnet.save_pretrained(config.output_dir)
    except Exception as exc:
        logger.warning("Failed to save ControlNet (%s); saving state dict instead.", exc)
        torch.save(controlnet.state_dict(), ckpt_path.with_suffix(".pt"))
    return {"steps": global_step, "last_loss": last_loss, "checkpoint": str(ckpt_path), "mode": "diffusers"}
