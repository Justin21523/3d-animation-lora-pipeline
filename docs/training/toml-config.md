# TOML 配置問題完整解釋

**為什麼之前 TOML 配置一直出問題？為什麼必須用 CLI？**

---

## 🔍 問題根源

### Kohya SS 的兩種配置方式

Kohya SS sd-scripts 實際上支援**兩種不同的配置方式**，但我們之前**混淆**了它們：

#### 方式 1: `--dataset_config` (Kohya 原生支援)
```bash
python train_network.py \
    --dataset_config dataset.toml \
    --pretrained_model_name_or_path /path/to/model \
    --output_dir /path/to/output \
    --learning_rate 0.0001 \
    # ... 其他所有參數都要用 CLI 傳遞
```

**特點：**
- ✅ Kohya 官方原生支援
- ⚠️  **只能配置數據集部分**（圖片路徑、batch size、resolution 等）
- ❌ **不能配置訓練參數**（learning rate, optimizer, epochs 等）
- ❌ 其他參數必須通過 CLI 傳遞

**dataset.toml 內容範例：**
```toml
[general]
shuffle_caption = true
keep_tokens = 3

[[datasets]]
resolution = 512
batch_size = 10

  [[datasets.subsets]]
  image_dir = "/path/to/images"
  class_tokens = "character boy"
  caption_extension = ".txt"
```

#### 方式 2: `--config_file` (完整配置，部分腳本支援)
```bash
python train_network.py \
    --config_file full_config.toml
```

**特點：**
- ⚠️  **不是所有 Kohya 腳本都支援**
- ✅ 可以配置所有訓練參數
- ✅ 單一文件包含完整配置
- ❌ Kohya 官方文檔沒有明確說明此功能

**full_config.toml 內容範例：**
```toml
[model_arguments]
pretrained_model_name_or_path = "/path/to/model"
output_dir = "/path/to/output"

[training_arguments]
learning_rate = 0.0001
max_train_epochs = 15

[network_arguments]
network_dim = 64

[dataset_arguments]
dataset_config = "/path/to/dataset.toml"
```

---

## ❌ 為什麼之前出問題？

### 問題 1: 混淆了兩種配置方式

我們之前創建的 TOML 文件使用了 **`--config_file` 格式**，但在代碼中使用 **`--dataset_config`** 參數，導致：

```python
# 錯誤的使用方式（之前的代碼）
cmd = [
    'python', 'train_network.py',
    '--dataset_config', 'full_config.toml',  # ❌ 這是完整配置文件
    # 缺少其他必要的 CLI 參數
]
```

**結果：**
- Kohya 只讀取了數據集配置部分
- 訓練參數（learning rate, optimizer 等）沒有被設置
- 導致訓練失敗或使用默認值

### 問題 2: 不清楚 `--config_file` 的支援狀況

```bash
# 檢查 train_network.py 是否支援 --config_file
cd /mnt/c/AI_LLM_projects/kohya_ss/sd-scripts
python train_network.py --help | grep config_file

# 結果：找到了 --config_file 選項
# --config_file CONFIG_FILE
#                     using .toml instead of args to pass hyperparameter
```

**2024-11-12 更新後的結論：**
- ✅ `train_network.py` **同時支援** `--config_file` 和 `--dataset_config`
- ⚠️  但 `--config_file` 的 `resolution` 參數解析有問題
- ✅ **推薦使用** `--dataset_config` + CLI 參數混合方式
- ✅ 更穩定、更靈活、更適合自動化迭代訓練

### 問題 3: 之前沒有正確的範本

我們之前的 TOML 範本混合了兩種格式，導致混亂：

```toml
# 錯誤的混合格式
[model_arguments]  # 這需要 --config_file 支援
pretrained_model_name_or_path = "..."

[general]  # 這是 --dataset_config 格式
shuffle_caption = true
```

---

## ✅ 正確的解決方案

### 當前的代碼（CLI 方式）

```python
# scripts/training/iterative_lora_optimizer.py
cmd = [
    'python', 'train_network.py',
    '--dataset_config', dataset_config_path,  # ✓ 只用於數據集配置
    '--pretrained_model_name_or_path', model_path,
    '--output_dir', output_dir,
    '--learning_rate', str(learning_rate),
    '--optimizer_type', 'AdamW',
    '--network_dim', '64',
    # ... 所有其他參數
]
```

**優點：**
- ✅ Kohya 完全支援
- ✅ 靈活動態調整參數
- ✅ 適合自動化迭代訓練

**缺點：**
- ❌ 命令行很長（50+ 個參數）
- ❌ 難以手動輸入
- ❌ 不易版本控制

### ✅ 推薦方案：混合方式（2024-11-12 更新）

**1. 數據集配置用 TOML**
```toml
# configs/luca_human/dataset.toml
# ⚠️  注意：不要使用 [general] 區塊！
# 2024-11-12 測試發現：Kohya 不支援 [general] 區塊
# shuffle_caption, keep_tokens 等參數應該在 subsets 層級

[[datasets]]
resolution = 512         # 單個整數，不是 [512, 512]
batch_size = 8
enable_bucket = true
min_bucket_reso = 384
max_bucket_reso = 768
bucket_reso_steps = 64
bucket_no_upscale = false

  [[datasets.subsets]]
  image_dir = "/path/to/images"
  num_repeats = 1
  shuffle_caption = true      # ✅ 在這裡！
  keep_tokens = 3             # ✅ 在這裡！
  caption_extension = ".txt"  # ✅ 在這裡！
  color_aug = false
  flip_aug = false
```

**2. 訓練參數用腳本配置**
```python
# scripts/training/train_luca.py
training_config = {
    'pretrained_model_name_or_path': '/path/to/model',
    'output_dir': '/path/to/output',
    'learning_rate': 0.0001,
    'optimizer_type': 'AdamW8bit',
    'network_dim': 64,
    'network_alpha': 32,
    'max_train_epochs': 15,
    # ... 所有訓練參數
}

cmd = [
    'conda', 'run', '-n', 'kohya_ss',
    'python', '/path/to/train_network.py',
    '--dataset_config', 'configs/luca_human/dataset.toml',
]

# 添加所有訓練參數
for key, value in training_config.items():
    cmd.extend([f'--{key}', str(value)])

subprocess.run(cmd)
```

**3. 或者創建包裝腳本**
```python
# scripts/training/launch_with_config.py
import toml
import subprocess

def load_full_config(config_path):
    """從完整 TOML 配置加載所有參數"""
    config = toml.load(config_path)

    cmd = ['conda', 'run', '-n', 'kohya_ss', 'python', 'train_network.py']

    # 從 [model_arguments] 添加參數
    for key, value in config.get('model_arguments', {}).items():
        cmd.extend([f'--{key}', str(value)])

    # 從 [training_arguments] 添加參數
    for key, value in config.get('training_arguments', {}).items():
        cmd.extend([f'--{key}', str(value)])

    # ... 處理其他區段

    return cmd

# 使用
config_path = 'configs/luca_human/full_config.toml'
cmd = load_full_config(config_path)
subprocess.run(cmd)
```

---

## 📊 三種方式對比

| 方式 | 優點 | 缺點 | 適用場景 |
|------|------|------|----------|
| **純 CLI** | • Kohya 完全支援<br>• 靈活動態調整 | • 命令行超長<br>• 難以管理 | 自動化系統、腳本控制 |
| **dataset_config + CLI** | • 數據集配置清晰<br>• 訓練參數靈活 | • 仍需大量 CLI 參數 | **推薦：一般使用** |
| **完整 TOML + 包裝器** | • 配置統一管理<br>• 易於版本控制 | • 需要自定義包裝器<br>• 額外維護成本 | 固定配置、團隊協作 |

---

## 🎯 我們的範本如何使用

### 範本文件結構

```
configs/templates/
├── lora_training_template.toml      # 完整配置範本（需包裝器）
└── dataset_config_template.toml     # 數據集配置範本（直接使用）
```

### 使用方式 A：數據集 TOML + 腳本參數

```bash
# 1. 複製數據集範本
cp configs/templates/dataset_config_template.toml configs/my_char/dataset.toml

# 2. 編輯 dataset.toml（只配置數據集）
nano configs/my_char/dataset.toml

# 3. 創建訓練腳本
cat > train_my_char.sh << 'EOF'
#!/bin/bash
conda run -n kohya_ss python /path/to/train_network.py \
    --dataset_config configs/my_char/dataset.toml \
    --pretrained_model_name_or_path /path/to/model \
    --output_dir /path/to/output \
    --learning_rate 0.0001 \
    --optimizer_type AdamW8bit \
    --network_dim 64 \
    --network_alpha 32 \
    --max_train_epochs 15 \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --cache_latents \
    --cache_latents_to_disk
EOF

# 4. 運行
bash train_my_char.sh
```

### 使用方式 B：完整 TOML + 包裝器（未來實現）

```bash
# 1. 複製完整配置範本
cp configs/templates/lora_training_template.toml configs/my_char/full_config.toml

# 2. 編輯完整配置
nano configs/my_char/full_config.toml

# 3. 使用包裝器運行
python scripts/training/launch_with_config.py \
    --config_file configs/my_char/full_config.toml
```

---

## 💡 重要發現記錄

### 1. Kohya 官方配置系統

根據 `/mnt/c/AI_LLM_projects/ai_warehouse/sd-scripts/docs/config_README-en.md`：

> "This README is about the configuration files that can be passed with the `--dataset_config` option."

**明確說明：**
- Kohya 官方只文檔化了 `--dataset_config`
- 該配置只用於數據集設置
- 沒有提到完整的 `--config_file` 系統

### 2. 為什麼測試腳本可以運行？

我們的 `test_toml_training.sh` 使用 `--config_file` 參數：

```bash
python train_network.py --config_file training_config.toml
```

**可能的情況：**
1. **某些 Kohya 腳本支援 `--config_file`**（但未文檔化）
2. **需要特定版本的 Kohya**
3. **社區貢獻的功能**（未合併到主分支）

**驗證方法：**
```bash
cd /mnt/c/AI_LLM_projects/ai_warehouse/sd-scripts
python train_network.py --help | grep -A 5 "config_file"

# 或直接測試
python train_network.py --config_file test.toml --help
```

### 3. 當前訓練為何成功？

當前的 14 小時迭代訓練使用純 CLI 方式：

```python
# iterative_lora_optimizer.py 使用純 CLI
cmd = [
    'python', 'train_network.py',
    '--dataset_config', dataset_config_toml,  # 只用於數據集
    '--pretrained_model_name_or_path', model_path,
    '--output_dir', output_dir,
    '--learning_rate', str(lr),
    # ... 50+ 個 CLI 參數
]
```

**成功原因：**
- ✅ 使用了 Kohya 官方支援的方式
- ✅ `--dataset_config` 只配置數據集
- ✅ 所有訓練參數通過 CLI 傳遞
- ✅ 沒有依賴未文檔化的 `--config_file`

---

## 📝 總結與建議

### 問題總結

1. **我們混淆了兩種配置方式**
   - `--dataset_config`（官方支援，僅數據集）
   - `--config_file`（可能存在，未文檔化）

2. **之前的 TOML 範本格式錯誤**
   - 使用了 `[model_arguments]` 等區段
   - 但用 `--dataset_config` 參數傳遞
   - Kohya 無法解析這些區段

3. **不得不改用 CLI**
   - 因為 TOML 配置無法工作
   - CLI 是唯一可靠的方式
   - 但導致命令行過長、難以管理

### 當前狀態

**已驗證可用：**
- ✅ `--dataset_config` + CLI 參數（當前使用）
- ✅ 純 CLI 參數（測試通過）
- ✅ kohya_ss 環境 + AdamW8bit（測試通過）

**待驗證：**
- ⚠️  `--config_file` 是否真的支援？
- ⚠️  我們的完整 TOML 範本是否能用？
- ⚠️  需要什麼版本的 Kohya？

### 建議

**短期（當前使用）：**
```python
# 使用 dataset_config + CLI 混合方式
# 優點：穩定可靠、Kohya 官方支援
# 缺點：命令行較長

cmd = [
    'python', 'train_network.py',
    '--dataset_config', 'dataset.toml',  # 數據集配置
    # ... 所有訓練參數用 CLI
]
```

**中期（包裝器方案）：**
```python
# 創建 Python 包裝器讀取完整 TOML
# 優點：配置統一、易於管理
# 缺點：需要維護包裝器代碼

def load_full_config(toml_path):
    config = toml.load(toml_path)
    return build_cli_command(config)
```

**長期（驗證並使用 config_file）：**
```bash
# 如果 --config_file 真的支援
python train_network.py --config_file full_config.toml
# 優點：最簡潔、最標準
# 需求：驗證 Kohya 版本和支援情況
```

---

## 🔗 相關文件

- **Kohya 官方文檔:** `/mnt/c/AI_LLM_projects/ai_warehouse/sd-scripts/docs/config_README-en.md`
- **當前範本:** `configs/templates/*.toml`
- **當前訓練腳本:** `scripts/training/iterative_lora_optimizer.py`
- **完整指南:** `docs/KOHYA_TRAINING_GUIDE.md`

---

**最後更新：** 2025-11-11
**狀態：** 已解決 - 使用 dataset_config + CLI 混合方式
