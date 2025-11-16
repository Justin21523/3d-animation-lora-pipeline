# Kohya LoRA Training Complete Guide

**Last Updated:** 2025-11-11
**Environment:** RTX 5080 16GB, PyTorch 2.7.1+cu128, Kohya SS sd-scripts (latest)

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Configuration Files](#configuration-files)
3. [Training Methods](#training-methods)
4. [Optimal Parameters for RTX 5080](#optimal-parameters-for-rtx-5080)
5. [Troubleshooting](#troubleshooting)
6. [Reference Commands](#reference-commands)

---

## Environment Setup

### Dedicated Conda Environment: `kohya_ss`

We maintain a **dedicated conda environment** specifically for Kohya LoRA training to ensure compatibility and prevent version conflicts.

#### Quick Setup

```bash
# Run the automated setup script
bash /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/setup_kohya_env.sh
```

#### Manual Setup

```bash
# 1. Create environment
conda create -n kohya_ss python=3.10 -y

# 2. Install PyTorch with CUDA 12.8
conda run -n kohya_ss pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. Install Kohya dependencies
conda run -n kohya_ss pip install \
    accelerate==0.30.0 \
    transformers==4.44.0 \
    diffusers[torch]==0.25.0 \
    bitsandbytes>=0.45.0 \
    safetensors==0.4.2 \
    toml==0.10.2 \
    # ... (see setup_kohya_env.sh for complete list)
```

#### Verification

```bash
conda run -n kohya_ss python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')

import bitsandbytes as bnb
print(f'bitsandbytes: {bnb.__version__}')

# Test AdamW8bit
param = torch.nn.Parameter(torch.randn(10, 10).cuda())
optimizer = bnb.optim.AdamW8bit([param], lr=1e-3)
print('✓ AdamW8bit works!')
"
```

**Expected Output:**
```
PyTorch: 2.7.1+cu128
CUDA: True
bitsandbytes: 0.48.2
✓ AdamW8bit works!
```

---

## Configuration Files

### ⚠️ IMPORTANT: Kohya Configuration System Explained

**請務必理解 Kohya 的配置系統，避免之前遇到的問題！**

Kohya SS 實際上只**官方支援一種配置方式**：

#### `--dataset_config`（唯一官方支援的 TOML 配置）

```bash
python train_network.py \
    --dataset_config dataset.toml \
    --pretrained_model_name_or_path /path/to/model \
    --output_dir /path/to/output \
    --learning_rate 0.0001 \
    # ... 其他參數都用 CLI
```

**特點：**
- ✅ **唯一官方文檔化的 TOML 配置**
- ⚠️  **只能配置數據集**（image_dir, batch_size, resolution 等）
- ❌ **不能配置訓練參數**（learning_rate, optimizer, epochs 等）
- ❌ 其他參數必須通過 CLI 傳遞

**詳細解釋請參考：** `docs/TOML_CONFIG_EXPLAINED.md`

### 為什麼之前 TOML 配置出問題？

**問題原因：**
1. 我們混淆了 `--dataset_config` 和 `--config_file` 兩種方式
2. 創建的範本使用了 `[model_arguments]`、`[training_arguments]` 等區段
3. 但 Kohya 的 `--dataset_config` 只能解析數據集配置
4. 導致訓練參數無法被讀取，必須改用 CLI

**當前解決方案：**
- 使用 `--dataset_config` + CLI 參數混合方式
- 穩定可靠，Kohya 官方支援
- 我們的迭代訓練系統已採用此方式

### TOML Configuration Structure

Kohya 支援的配置方式：
1. **dataset_config + CLI** (官方支援，當前使用)
2. **Pure CLI** (完全支援，但命令行過長)
3. **config_file** (未文檔化，不推薦依賴)

We use **method 1** (dataset_config + CLI) for reliability.

### Template Files

```
configs/
├── templates/
│   ├── lora_training_template.toml      # Main training config template
│   └── dataset_config_template.toml     # Dataset config template
└── your_character/
    ├── training_config.toml              # Your customized config
    └── dataset_config.toml               # Your dataset config
```

### Creating a New Configuration

```bash
# 1. Copy templates
mkdir -p configs/your_character
cp configs/templates/lora_training_template.toml configs/your_character/training_config.toml
cp configs/templates/dataset_config_template.toml configs/your_character/dataset_config.toml

# 2. Edit configs (replace YOUR_CHARACTER placeholders)
nano configs/your_character/training_config.toml
nano configs/your_character/dataset_config.toml

# 3. Test configuration
bash test_toml_training.sh  # Validates config structure
```

### Key Configuration Sections

#### 1. Model Arguments
```toml
[model_arguments]
pretrained_model_name_or_path = "/path/to/stable-diffusion-v1-5"
output_dir = "/path/to/output"
output_name = "character_lora_v1"
save_model_as = "safetensors"
save_precision = "fp16"
```

#### 2. Training Arguments
```toml
[training_arguments]
learning_rate = 0.0001
unet_lr = 0.0001
text_encoder_lr = 0.00005
lr_scheduler = "cosine_with_restarts"
optimizer_type = "AdamW8bit"  # or "AdamW"
max_train_epochs = 15
mixed_precision = "fp16"
gradient_checkpointing = true
gradient_accumulation_steps = 2
```

#### 3. Network Arguments (LoRA)
```toml
[network_arguments]
network_module = "networks.lora"
network_dim = 64      # LoRA rank
network_alpha = 32    # Usually rank/2
```

#### 4. Dataset Configuration
```toml
[dataset_arguments]
dataset_config = "/path/to/dataset_config.toml"
```

**In `dataset_config.toml`:**
```toml
[[datasets]]
resolution = 512
batch_size = 10

  [[datasets.subsets]]
  image_dir = "/path/to/images"
  class_tokens = "character_name trigger_word"
  num_repeats = 1
  caption_extension = ".txt"
```

---

## Training Methods

### Method 1: TOML Configuration (Recommended)

**Advantages:**
- ✅ Reproducible (save exact training settings)
- ✅ Version control friendly
- ✅ Easy to share and reuse
- ✅ Less error-prone than long CLI commands

**Usage:**
```bash
conda run -n kohya_ss python /path/to/sd-scripts/train_network.py \
    --config_file configs/your_character/training_config.toml
```

### Method 2: CLI Arguments (Deprecated)

**Disadvantages:**
- ❌ Hard to reproduce
- ❌ Error-prone (typos, missing flags)
- ❌ Difficult to track changes

**Only use this for quick tests:**
```bash
conda run -n kohya_ss python /path/to/sd-scripts/train_network.py \
    --dataset_config /path/to/dataset.toml \
    --pretrained_model_name_or_path /path/to/model \
    --output_dir /path/to/output \
    # ... (50+ more arguments)
```

---

## Optimal Parameters for RTX 5080

### Hardware Specifications
- **GPU:** NVIDIA GeForce RTX 5080
- **VRAM:** 16GB
- **CUDA:** 12.8
- **Architecture:** Blackwell

### Critical Constraints

#### ❌ DO NOT USE:
- `--xformers` flag (hardware incompatible)
- `flip_aug` or `color_aug` (breaks 3D character consistency)
- `optimizer_type = "AdamW8bit"` **without testing** (requires proper bitsandbytes)

#### ✅ RECOMMENDED:
```toml
[training_arguments]
optimizer_type = "AdamW8bit"  # Works in kohya_ss environment
# or
optimizer_type = "AdamW"      # Safe fallback, slightly more VRAM

[caching_arguments]
cache_latents = true
cache_latents_to_disk = true

[training_arguments]
gradient_checkpointing = true
gradient_accumulation_steps = 2
```

### Optimal Settings by Resolution

| Resolution | Batch Size | Network Dim | VRAM Usage | Training Speed |
|------------|------------|-------------|------------|----------------|
| 512x512    | 10         | 64          | ~15GB      | ~30s/step      |
| 512x512    | 8          | 64          | ~13GB      | ~24s/step      |
| 768x768    | 4-6        | 64          | ~14GB      | ~40s/step      |
| 1024x1024  | 2-4        | 128         | ~15GB      | ~60s/step      |

### Recommended Training Parameters

#### For Character LoRA (200-400 images):
```toml
[training_arguments]
learning_rate = 0.0001
unet_lr = 0.0001
text_encoder_lr = 0.00005
max_train_epochs = 12-15
lr_scheduler = "cosine_with_restarts"

[network_arguments]
network_dim = 64
network_alpha = 32

[[datasets]]
resolution = 512
batch_size = 10
num_repeats = 1  # Increase to 2 if <150 images
```

#### For Style LoRA (500-1000 images):
```toml
[training_arguments]
learning_rate = 0.00005  # Lower for style
max_train_epochs = 8-10

[network_arguments]
network_dim = 128  # Higher capacity for style
network_alpha = 64

[[datasets]]
batch_size = 8
num_repeats = 1
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. bitsandbytes Import Error

**Error:**
```
ModuleNotFoundError: No module named 'bitsandbytes'
ImportError: libbitsandbytes_cuda128.so not found
```

**Solution:**
```bash
# Use the kohya_ss environment
conda activate kohya_ss

# Or rebuild environment
bash setup_kohya_env.sh
```

#### 2. xformers Error

**Error:**
```
ImportError: No module named 'xformers'
```

**Solution:**
❌ **DO NOT install xformers** on RTX 5080. It's incompatible.

Remove `--xformers` flag from all training commands and configs.

#### 3. TOML Config Not Working

**Error:**
```
KeyError: 'model_arguments'
ValueError: Unknown argument: pretrained_model_name_or_path
```

**Solution:**
Kohya expects specific config structure. Use our templates:
- `configs/templates/lora_training_template.toml`
- `configs/templates/dataset_config_template.toml`

**Test config validity:**
```bash
conda run -n kohya_ss python /path/to/sd-scripts/train_network.py \
    --config_file your_config.toml \
    --help
```

#### 4. Out of Memory (OOM)

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution:**
Reduce memory usage step by step:

1. **Lower batch size:**
   ```toml
   batch_size = 8  # or 6, 4
   ```

2. **Increase gradient accumulation:**
   ```toml
   gradient_accumulation_steps = 4  # effectively doubles batch size
   ```

3. **Enable more caching:**
   ```toml
   cache_latents_to_disk = true
   cache_text_encoder_outputs_to_disk = true  # For SDXL
   ```

4. **Lower network capacity:**
   ```toml
   network_dim = 32  # instead of 64
   ```

#### 5. Training Stuck/Hanging

**Symptoms:**
- Process starts but no training progress
- GPU utilization 0%
- No error messages

**Solutions:**

**a) Check dataset paths:**
```bash
# Verify images exist
ls -lh /path/to/your/dataset/images/*.png | head

# Verify captions exist
ls -lh /path/to/your/dataset/images/*.txt | head
```

**b) Test with minimal config:**
```toml
max_train_epochs = 1
save_every_n_epochs = 1
[[datasets.subsets]]
num_repeats = 1
```

**c) Check data loader workers:**
```toml
max_data_loader_n_workers = 0  # Disable multi-processing to debug
```

#### 6. Poor Training Results

**Symptoms:**
- LoRA doesn't capture character features
- Overfitting (memorizes exact training images)
- Underfitting (barely affects output)

**Solutions:**

**For overfitting:**
- Reduce epochs: `max_train_epochs = 10` (instead of 20)
- Increase dataset size or repeats
- Lower learning rate: `learning_rate = 0.00005`
- Add caption dropout: `caption_dropout_rate = 0.1`

**For underfitting:**
- Increase epochs: `max_train_epochs = 20`
- Increase network capacity: `network_dim = 128`
- Raise learning rate: `learning_rate = 0.0002`
- Check caption quality (generic vs. specific)

---

## Reference Commands

### Environment Management

```bash
# Activate kohya environment
conda activate kohya_ss

# Deactivate
conda deactivate

# List environments
conda env list

# Update packages
conda run -n kohya_ss pip install --upgrade bitsandbytes transformers

# Remove and recreate environment
conda remove -n kohya_ss --all -y
bash setup_kohya_env.sh
```

### Training Commands

**Basic training:**
```bash
conda run -n kohya_ss python /path/to/sd-scripts/train_network.py \
    --config_file configs/my_character/training_config.toml
```

**Training with output redirect:**
```bash
conda run -n kohya_ss python /path/to/sd-scripts/train_network.py \
    --config_file configs/my_character/training_config.toml \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
```

**Training in tmux (for long sessions):**
```bash
tmux new -s training
conda activate kohya_ss
python /path/to/sd-scripts/train_network.py --config_file configs/my_config.toml

# Detach: Ctrl+B, D
# Reattach: tmux attach -t training
```

### Monitoring

```bash
# GPU utilization
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi

# TensorBoard logs
tensorboard --logdir /path/to/logs --port 6006

# Training progress (if using tee)
tail -f logs/training_20251111_120000.log
```

### File Operations

```bash
# List trained checkpoints
ls -lh /path/to/output/*.safetensors

# Check checkpoint size
du -h /path/to/output/character_lora.safetensors

# Compare configs
diff configs/version1/training_config.toml configs/version2/training_config.toml

# Find all TOML configs
find configs/ -name "*.toml" -type f
```

---

## Additional Resources

### Internal Documentation
- `configs/templates/lora_training_template.toml` - Main config template (reference)
- `configs/templates/dataset_config_template.toml` - Dataset config template (for --dataset_config)
- `docs/GPU_OPTIMIZATION_GUIDE.md` - RTX 5080 optimization guide
- `docs/TOML_CONFIG_EXPLAINED.md` - **Why TOML had issues and how to use it correctly**
- `setup_kohya_env.sh` - Environment setup script
- `test_toml_training.sh` - Config validation script

### External Resources
- [Kohya SS GitHub](https://github.com/kohya-ss/sd-scripts)
- [LoRA Training Guide](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_README-en.md)
- [Stable Diffusion Training Tips](https://rentry.org/2chAI_LoRA_Dreambooth_guide_english)

---

## Version History

| Date       | Version | Changes                                          |
|------------|---------|--------------------------------------------------|
| 2025-11-11 | 1.0     | Initial guide with kohya_ss environment setup    |
| 2025-11-11 | 1.1     | Added TOML templates and RTX 5080 optimizations  |

---

**For questions or issues, refer to:**
- Project documentation: `/docs/`
- Environment setup logs: `/tmp/kohya_env_setup.log`
- Training logs: `/mnt/data/ai_data/models/lora/*/logs/`
