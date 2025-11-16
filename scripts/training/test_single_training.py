#!/usr/bin/env python3
"""
Test single LoRA training run with 1 epoch to validate configuration

This script tests:
- Data loading
- Training process
- GPU utilization
- Memory management
- Error handling
"""

import sys
import subprocess
from pathlib import Path
import time

# Configuration for quick test
test_config = {
    'character': 'luca_human',
    'dataset_dir': '/mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset/luca_human/images',
    'base_model': '/mnt/c/AI_LLM_projects/ai_warehouse/tool-caches/huggingface/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14',
    'output_dir': '/mnt/data/ai_data/models/lora/luca/test_run',
    'sd_scripts_dir': '/mnt/c/AI_LLM_projects/ai_warehouse/sd-scripts',
}

print("="*70)
print("LORA TRAINING - CONFIGURATION TEST")
print("="*70)
print(f"Character: {test_config['character']}")
print(f"Dataset: {test_config['dataset_dir']}")
print(f"Output: {test_config['output_dir']}")
print(f"Test: 1 epoch only (quick validation)")
print("="*70)
print()

# Create output directory
output_dir = Path(test_config['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)
config_dir = output_dir / 'configs'
config_dir.mkdir(exist_ok=True)

# Generate dataset config
dataset_config_content = f"""# Test dataset config

[general]
shuffle_caption = true
keep_tokens = 3

[[datasets]]
resolution = 512
batch_size = 8
enable_bucket = true
min_bucket_reso = 384
max_bucket_reso = 768
bucket_reso_steps = 64

  [[datasets.subsets]]
  image_dir = "{test_config['dataset_dir']}"
  class_tokens = "luca boy"
  num_repeats = 1
  caption_extension = ".txt"
"""

config_path = config_dir / 'test_config.toml'
with open(config_path, 'w') as f:
    f.write(dataset_config_content)

print(f"✓ Config created: {config_path}\n")

# Build training command
train_script = Path(test_config['sd_scripts_dir']) / 'train_network.py'

cmd = [
    'conda', 'run', '-n', 'ai_env',
    'python', str(train_script),
    '--dataset_config', str(config_path),
    '--pretrained_model_name_or_path', test_config['base_model'],
    '--output_dir', str(output_dir),
    '--output_name', 'test_lora',
    '--learning_rate', '0.0001',
    '--unet_lr', '0.0001',
    '--text_encoder_lr', '0.00005',
    '--lr_scheduler', 'cosine_with_restarts',
    '--lr_warmup_steps', '50',
    '--optimizer_type', 'AdamW',  # Standard AdamW
    '--network_module', 'networks.lora',
    '--network_dim', '64',
    '--network_alpha', '32',
    '--max_train_epochs', '1',  # Only 1 epoch for testing
    '--save_every_n_epochs', '1',
    '--mixed_precision', 'fp16',
    '--save_model_as', 'safetensors',
    '--cache_latents',
    '--cache_latents_to_disk',
    '--gradient_checkpointing',
    '--gradient_accumulation_steps', '2',
    '--max_data_loader_n_workers', '8',
    '--seed', '42',
    '--clip_skip', '2',
]

print("Training command:")
print(" ".join(cmd[:5]) + " \\")
for i in range(5, len(cmd), 2):
    if i + 1 < len(cmd):
        print(f"  {cmd[i]} {cmd[i+1]} \\")
    else:
        print(f"  {cmd[i]}")
print()

print("="*70)
print("STARTING TEST TRAINING (1 EPOCH)")
print("="*70)
print("Monitoring: watch -n 1 nvidia-smi")
print("="*70)
print()

start_time = time.time()

try:
    # Run training with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Stream output
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()

    process.wait()

    elapsed = time.time() - start_time

    if process.returncode == 0:
        print("\n" + "="*70)
        print("✓ TEST COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        print(f"Output saved to: {output_dir}")
        print()
        print("Configuration validated! Ready for full 14-hour training.")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ TEST FAILED")
        print("="*70)
        print(f"Return code: {process.returncode}")
        print("Check error messages above")
        sys.exit(1)

except KeyboardInterrupt:
    print("\n" + "="*70)
    print("! TEST INTERRUPTED")
    print("="*70)
    process.kill()
    sys.exit(1)

except Exception as e:
    print("\n" + "="*70)
    print(f"✗ ERROR: {e}")
    print("="*70)
    import traceback
    traceback.print_exc()
    sys.exit(1)
