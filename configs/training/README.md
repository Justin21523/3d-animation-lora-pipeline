# 配置範本說明

## 📁 範本文件與用途

### 1. `dataset_config_template.toml` ✅ **可直接使用**

**用途：** 配合 Kohya 的 `--dataset_config` 參數使用

**格式：**
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
```

**使用方式：**
```bash
# 1. 複製範本
cp configs/templates/dataset_config_template.toml configs/my_char/dataset.toml

# 2. 編輯配置（修改路徑、參數）
nano configs/my_char/dataset.toml

# 3. 配合 CLI 參數使用
python train_network.py \
    --dataset_config configs/my_char/dataset.toml \
    --pretrained_model_name_or_path /path/to/model \
    --learning_rate 0.0001 \
    --optimizer_type AdamW8bit \
    # ... 其他訓練參數
```

**配置內容：**
- ✅ 數據集路徑 (`image_dir`)
- ✅ Batch size, resolution
- ✅ Bucketing 設置
- ✅ Caption 處理選項
- ✅ 數據增強選項

**不能配置：**
- ❌ 訓練參數（learning rate, optimizer, epochs）
- ❌ 模型路徑
- ❌ 輸出目錄
- ❌ 網絡結構（network_dim, alpha）

---

### 2. `lora_training_template.toml` ⚠️ **僅作參考文檔**

**用途：** 作為**完整訓練配置的參考文檔**，記錄所有可用參數

**格式：**
```toml
[model_arguments]
pretrained_model_name_or_path = "/path/to/model"
output_dir = "/path/to/output"

[training_arguments]
learning_rate = 0.0001
max_train_epochs = 15

[network_arguments]
network_dim = 64
network_alpha = 32

[dataset_arguments]
dataset_config = "/path/to/dataset.toml"
```

**⚠️ 重要：** Kohya 的 `train_network.py` **不支援** `--config_file` 參數，所以這個範本**不能直接使用**！

**實際用途：**
1. **參考文檔** - 查閱所有可用的訓練參數
2. **配置規劃** - 計劃訓練配置時的檢查清單
3. **未來擴展** - 如果將來創建包裝器腳本時使用

**如何實際使用這些參數？**

有兩種方式：

#### 方式 A：創建訓練腳本（推薦）

```bash
# 創建腳本文件
cat > configs/my_char/train.sh << 'EOF'
#!/bin/bash

# 從 lora_training_template.toml 參考的參數
MODEL_PATH="/path/to/model"
OUTPUT_DIR="/path/to/output"
LEARNING_RATE="0.0001"
NETWORK_DIM="64"
EPOCHS="15"

# 訓練命令
conda run -n kohya_ss python /path/to/train_network.py \
    --dataset_config configs/my_char/dataset.toml \
    --pretrained_model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LEARNING_RATE" \
    --network_dim "$NETWORK_DIM" \
    --max_train_epochs "$EPOCHS" \
    --optimizer_type "AdamW8bit" \
    --mixed_precision "fp16" \
    --gradient_checkpointing \
    --cache_latents_to_disk
EOF

chmod +x configs/my_char/train.sh
bash configs/my_char/train.sh
```

#### 方式 B：使用 Python 配置字典

```python
# train_my_char.py
import subprocess

# 從 lora_training_template.toml 轉換為 Python 字典
config = {
    # [model_arguments]
    'pretrained_model_name_or_path': '/path/to/model',
    'output_dir': '/path/to/output',
    'output_name': 'my_char_lora',
    'save_model_as': 'safetensors',

    # [training_arguments]
    'learning_rate': '0.0001',
    'max_train_epochs': '15',
    'optimizer_type': 'AdamW8bit',
    'mixed_precision': 'fp16',

    # [network_arguments]
    'network_dim': '64',
    'network_alpha': '32',
}

# 構建命令
cmd = [
    'conda', 'run', '-n', 'kohya_ss',
    'python', '/path/to/train_network.py',
    '--dataset_config', 'configs/my_char/dataset.toml',
]

for key, value in config.items():
    cmd.extend([f'--{key}', str(value)])

subprocess.run(cmd)
```

---

## 📊 範本對比表

| 範本 | 格式 | Kohya 支援 | 直接使用 | 用途 |
|------|------|-----------|---------|------|
| `dataset_config_template.toml` | `[general]`, `[[datasets]]` | ✅ `--dataset_config` | ✅ 是 | 數據集配置 |
| `lora_training_template.toml` | `[model_arguments]`, `[training_arguments]` | ❌ 不支援 | ❌ 否 | 參考文檔 |

---

## 🎯 推薦工作流程

### 步驟 1：設置數據集配置

```bash
# 複製並編輯數據集配置
cp configs/templates/dataset_config_template.toml configs/my_char/dataset.toml
nano configs/my_char/dataset.toml
```

### 步驟 2：查閱完整配置範本

```bash
# 打開完整配置範本作為參考
cat configs/templates/lora_training_template.toml

# 記下你需要的參數及其值
```

### 步驟 3：創建訓練腳本

```bash
# 方式 A：創建 Shell 腳本
nano configs/my_char/train.sh

# 方式 B：創建 Python 腳本
nano configs/my_char/train.py
```

### 步驟 4：執行訓練

```bash
# Shell 腳本
bash configs/my_char/train.sh

# Python 腳本
python configs/my_char/train.py
```

---

## 💡 為什麼有兩個範本？

1. **`dataset_config_template.toml`** - 這是 Kohya **官方支援**的配置格式，可以直接使用

2. **`lora_training_template.toml`** - 這是我們創建的**完整配置文檔**，雖然不能直接用於 Kohya，但它：
   - 記錄了所有訓練參數
   - 提供了完整的配置結構
   - 方便查閱和規劃
   - 可用於未來的自定義包裝器

---

## 📚 相關文檔

- **詳細解釋：** `/docs/TOML_CONFIG_EXPLAINED.md` - 為什麼會有這兩種格式
- **訓練指南：** `/docs/KOHYA_TRAINING_GUIDE.md` - 完整的訓練流程
- **快速參考：** `/QUICK_REFERENCE_LORA.md` - 快速開始訓練

---

## ❓ 常見問題

### Q: 為什麼 `lora_training_template.toml` 不能直接使用？

**A:** Kohya 的 `train_network.py` 只支援 `--dataset_config` 參數（用於數據集配置），不支援 `--config_file` 參數（用於完整配置）。

### Q: 那為什麼還要保留 `lora_training_template.toml`？

**A:** 因為它是很好的**參考文檔**，記錄了所有參數及其說明。你可以根據它創建訓練腳本或 Python 配置。

### Q: 我能只用 `dataset_config_template.toml` 嗎？

**A:** 不能。數據集配置只包含數據集部分，你還需要通過 CLI 或腳本傳遞訓練參數。

### Q: 推薦使用哪種方式？

**A:** 推薦使用 **數據集 TOML + 訓練腳本** 的組合方式：
- 數據集配置用 `dataset.toml`
- 訓練參數寫在 `train.sh` 或 `train.py` 腳本中

---

**最後更新：** 2025-11-11
