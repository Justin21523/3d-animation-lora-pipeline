# 配置範本使用對比

## 快速理解

```
dataset_config_template.toml  →  --dataset_config  →  ✅ Kohya 直接支援
lora_training_template.toml    →  --config_file     →  ❌ Kohya 不支援（僅參考）
```

## 視覺對比

### ✅ 可用：`dataset_config_template.toml`

```
你的配置：                            Kohya 讀取：
┌─────────────────────────┐          ┌─────────────────────────┐
│ dataset.toml            │          │ train_network.py        │
│                         │          │                         │
│ [general]               │  ──────> │ --dataset_config ✓      │
│ [[datasets]]            │  支援    │ 讀取數據集配置          │
│   [[datasets.subsets]]  │          │                         │
└─────────────────────────┘          │ + CLI 參數              │
                                     │ --learning_rate         │
CLI 參數：                           │ --optimizer_type        │
--learning_rate 0.0001              │ --network_dim           │
--optimizer_type AdamW8bit          └─────────────────────────┘
--network_dim 64
```

### ❌ 不可用：`lora_training_template.toml`

```
你的配置：                            Kohya 讀取：
┌─────────────────────────┐          ┌─────────────────────────┐
│ full_config.toml        │          │ train_network.py        │
│                         │          │                         │
│ [model_arguments]       │  ────X─> │ --config_file ✗         │
│ [training_arguments]    │  不支援  │ 無法解析                │
│ [network_arguments]     │          │                         │
└─────────────────────────┘          └─────────────────────────┘

結果：訓練失敗或使用錯誤的默認值
```

---

## 實際使用範例

### 場景 1：訓練 Luca 角色

#### ✅ 正確方式

```bash
# 1. 創建數據集配置
cat > configs/luca/dataset.toml << 'TOML'
[general]
shuffle_caption = true
keep_tokens = 3

[[datasets]]
resolution = 512
batch_size = 10

  [[datasets.subsets]]
  image_dir = "/mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset/luca_human/images"
  class_tokens = "luca boy"
  num_repeats = 1
  caption_extension = ".txt"
TOML

# 2. 查閱完整範本（參考訓練參數）
cat configs/templates/lora_training_template.toml | grep -A 5 "\[training_arguments\]"

# 3. 創建訓練腳本
cat > configs/luca/train.sh << 'BASH'
#!/bin/bash
conda run -n kohya_ss python /path/to/train_network.py \
    --dataset_config configs/luca/dataset.toml \
    --pretrained_model_name_or_path /path/to/sd-v1-5 \
    --output_dir /path/to/output \
    --learning_rate 0.0001 \
    --optimizer_type AdamW8bit \
    --network_dim 64 \
    --max_train_epochs 15 \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --cache_latents_to_disk
BASH

# 4. 執行訓練
bash configs/luca/train.sh
```

#### ❌ 錯誤方式

```bash
# ❌ 嘗試使用完整配置文件
cat > configs/luca/full_config.toml << 'TOML'
[model_arguments]
pretrained_model_name_or_path = "/path/to/model"

[training_arguments]
learning_rate = 0.0001
TOML

# ❌ 這個命令會失敗
python train_network.py --config_file configs/luca/full_config.toml
# 錯誤：Unknown argument: --config_file
```

---

## 三種實際工作流程

### 方式 A：Shell 腳本（最簡單）

```bash
# 文件結構
configs/my_char/
├── dataset.toml           # 從 dataset_config_template.toml 複製
└── train.sh               # 手動創建，參考 lora_training_template.toml

# 使用
bash configs/my_char/train.sh
```

**優點：** 簡單直接，易於修改
**缺點：** 不適合動態參數調整

### 方式 B：Python 配置字典（靈活）

```bash
# 文件結構
configs/my_char/
├── dataset.toml           # 從 dataset_config_template.toml 複製
├── config.py              # Python 配置字典
└── train.py               # Python 訓練腳本

# config.py
TRAINING_CONFIG = {
    'pretrained_model_name_or_path': '/path/to/model',
    'learning_rate': '0.0001',
    # ... 所有參數
}

# 使用
python configs/my_char/train.py
```

**優點：** 靈活，可編程
**缺點：** 需要維護 Python 代碼

### 方式 C：迭代訓練系統（自動化）

```python
# 使用我們的迭代訓練系統
python scripts/training/launch_iterative_training.py

# 系統內部：
# 1. 生成 dataset.toml
# 2. 根據評估結果調整參數
# 3. 構建 CLI 命令
# 4. 自動執行訓練
```

**優點：** 全自動，參數優化
**缺點：** 複雜系統，難以自定義

---

## 決策流程圖

```
需要訓練 LoRA？
    ↓
是否使用自動化系統？
    ├─ 是 → 使用 launch_iterative_training.py
    │       （系統自動處理配置）
    └─ 否 → 手動配置
            ↓
        需要什麼配置？
            ├─ 數據集 → 使用 dataset_config_template.toml
            │           複製 → 編輯 → --dataset_config
            │
            └─ 訓練參數 → 查閱 lora_training_template.toml
                         選擇方式：
                         ├─ Shell 腳本（簡單）
                         ├─ Python 字典（靈活）
                         └─ 直接 CLI（測試）
```

---

## 記憶口訣

```
dataset_config_template  =  數據集配置  =  直接使用  ✓
lora_training_template   =  完整配置    =  僅供參考  ℹ
```

**關鍵：**
- **可以用** `--dataset_config` + 數據集 TOML
- **不能用** `--config_file` + 完整 TOML
- **訓練參數** 必須用 CLI 或腳本傳遞

---

**最後更新：** 2025-11-11
