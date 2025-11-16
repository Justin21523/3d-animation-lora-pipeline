# LoRA Training Quick Reference

**快速參考指南 - 適用於所有未來的 LoRA 訓練**

---

## 🚀 快速開始（5 分鐘設置）

### ⚠️ 重要：TOML 配置說明
**Kohya 只支援 `--dataset_config`（數據集配置），訓練參數需用 CLI 或腳本傳遞。**
詳細說明見：`docs/TOML_CONFIG_EXPLAINED.md`

### 1. 啟動專用環境
```bash
conda activate kohya_ss
```

### 2. 從範本創建數據集配置
```bash
# 創建角色目錄
mkdir -p configs/your_character

# 複製數據集範本（只需這個）
cp configs/templates/dataset_config_template.toml configs/your_character/dataset.toml

# 編輯配置（替換路徑和參數）
nano configs/your_character/dataset.toml
```

### 3. 創建訓練腳本
```bash
# 創建訓練腳本（包含所有訓練參數）
cat > configs/your_character/train.sh << 'EOF'
#!/bin/bash
conda run -n kohya_ss python /mnt/c/AI_LLM_projects/ai_warehouse/sd-scripts/train_network.py \
    --dataset_config configs/your_character/dataset.toml \
    --pretrained_model_name_or_path /path/to/model \
    --output_dir /path/to/output \
    --learning_rate 0.0001 \
    --optimizer_type AdamW8bit \
    --network_dim 64 \
    --max_train_epochs 15 \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --cache_latents_to_disk
EOF

chmod +x configs/your_character/train.sh
```

### 4. 開始訓練
```bash
bash configs/your_character/train.sh
```

---

## ⚙️ RTX 5080 最佳配置（複製即用）

### SD 1.5 - 角色 LoRA（200-400 張圖片）
```toml
[training_arguments]
learning_rate = 0.0001
unet_lr = 0.0001
text_encoder_lr = 0.00005
optimizer_type = "AdamW8bit"  # 或 "AdamW"
max_train_epochs = 15
batch_size = 10  # 在 dataset_config.toml 中設置
gradient_accumulation_steps = 2

[network_arguments]
network_dim = 64
network_alpha = 32

[caching_arguments]
cache_latents = true
cache_latents_to_disk = true
```

### SD 1.5 - 風格 LoRA（500-1000 張圖片）
```toml
[training_arguments]
learning_rate = 0.00005  # 較低
max_train_epochs = 10
batch_size = 8

[network_arguments]
network_dim = 128  # 較高容量
network_alpha = 64
```

### ⭐ SDXL - 角色 LoRA（16GB VRAM優化）
```toml
[model]
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
network_dim = 128
network_alpha = 96

[training_arguments]
# 核心16GB優化（必須）
optimizer_type = "AdamW8bit"           # ⭐ 省40% VRAM
mixed_precision = "bf16"               # ⭐ 省25% VRAM
full_bf16 = true
gradient_checkpointing = true          # ⭐ 省30% VRAM
cache_latents = true
vae_batch_size = 1

# 學習率（SDXL需要較低）
learning_rate = 0.0001                 # SD 1.5的77%
text_encoder_lr = 0.00006
unet_lr = 0.0001

# Batch設置
train_batch_size = 1                   # ⭐ 小batch for VRAM
gradient_accumulation_steps = 8        # ⭐ 維持effective batch=8

# 訓練時長
max_train_epochs = 20                  # 比SD 1.5多2-4 epochs
save_every_n_epochs = 2

# SDXL特有設置
resolution = "1024,1024"
enable_bucket = true
bucket_no_upscale = true

# 穩定性（與SD 1.5相同）
min_snr_gamma = 5.0
noise_offset = 0.05
lr_scheduler = "cosine_with_restarts"
```

**SDXL vs SD 1.5 對比：**
| 項目 | SD 1.5 | SDXL | 備註 |
|------|--------|------|------|
| **訓練時間** | 2-3小時 | 5-6小時 | 2.5倍 |
| **VRAM** | 10-12GB | 14-15GB | +30% |
| **解析度** | 512px | 1024px | 2倍細節 |
| **視覺品質** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 顯著提升 |
| **檔案大小** | 140MB | 800MB | 6倍 |

---

## 📊 VRAM 使用對照表

### SD 1.5
| 解析度     | Batch Size | Network Dim | VRAM 使用 | 建議場景      |
|-----------|------------|-------------|-----------|--------------|
| 512×512   | 10         | 64          | ~15GB     | 角色 LoRA     |
| 512×512   | 8          | 64          | ~13GB     | 安全設置      |
| 768×768   | 6          | 64          | ~14GB     | 高解析度角色  |

### SDXL（16GB優化）
| 解析度     | Batch Size | Network Dim | VRAM 使用 | 優化技術 |
|-----------|------------|-------------|-----------|---------|
| 1024×1024 | 1          | 128         | ~14-15GB  | AdamW8bit + BF16 + Grad Checkpoint |
| 768×768   | 1          | 128         | ~12-13GB  | 同上（降低解析度）|
| 1024×1024 | 1          | 128         | ~12-13GB  | 同上 + Flash Attention 2 |

---

## ⚠️ 重要限制（RTX 5080）

### ❌ 絕對不能用：
- `--xformers` 標記
- `flip_aug = true`（3D 角色）
- `color_aug = true`（3D 角色）

### ✅ 必須啟用：
- `gradient_checkpointing = true`
- `cache_latents_to_disk = true`
- `mixed_precision = "fp16"`

---

## 🔧 常見問題速查

### 訓練卡住/掛起
```bash
# 檢查數據集
ls /path/to/images/*.png | wc -l  # 檢查圖片數量
ls /path/to/images/*.txt | wc -l  # 檢查標註數量

# 降低 workers
max_data_loader_n_workers = 0  # 改為 0 進行除錯
```

### OOM (記憶體不足)
```toml
# 方案 1: 降低 batch size
batch_size = 6  # 從 10 → 6

# 方案 2: 增加梯度累積
gradient_accumulation_steps = 4  # 從 2 → 4

# 方案 3: 降低網絡容量
network_dim = 32  # 從 64 → 32
```

### bitsandbytes 錯誤
```bash
# 確保使用正確環境
conda activate kohya_ss

# 重建環境（如果需要）
bash /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/setup_kohya_env.sh
```

---

## 📁 文件位置速查

### 配置範本
```
configs/templates/
├── lora_training_template.toml      # 主訓練配置
└── dataset_config_template.toml     # 數據集配置
```

### 環境設置
```bash
# 設置腳本
/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/setup_kohya_env.sh

# 測試腳本
/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/test_toml_training.sh
```

### 文檔
```
docs/
├── KOHYA_TRAINING_GUIDE.md          # 完整指南
├── GPU_OPTIMIZATION_GUIDE.md        # GPU 優化
└── guides/tools/                     # 工具使用指南
```

---

## 🎯 典型訓練流程

### 單次訓練（15 epochs）
```bash
# 1. 準備數據集（圖片 + 標註）
/mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset/luca_human/images/
├── img001.png
├── img001.txt
├── img002.png
├── img002.txt
...

# 2. 創建配置（從範本）
configs/luca_human/
├── training_config.toml
└── dataset_config.toml

# 3. 訓練
conda activate kohya_ss
python /path/to/sd-scripts/train_network.py \
    --config_file configs/luca_human/training_config.toml

# 4. 監控
nvidia-smi  # 檢查 GPU
tail -f logs/training.log  # 查看進度
```

### 迭代訓練（自動優化）
```bash
# 使用現有的迭代訓練系統
python scripts/training/launch_iterative_training.py

# 或使用 tmux（長時間運行）
bash scripts/training/start_training_tmux.sh
```

---

## 📈 訓練品質判斷

### 好的跡象 ✓
- 損失穩定下降（0.25 → 0.15）
- GPU 使用率 80-100%
- 訓練速度穩定（~30秒/步）
- 檢查點文件正常生成（每 3 epochs）

### 壞的跡象 ✗
- 損失震盪或上升
- GPU 使用率低於 50%
- 訓練速度突然變慢
- OOM 錯誤

### 調整建議
```python
# 過擬合（損失很低但效果差）
→ 減少 epochs: 15 → 10
→ 增加數據集或 repeats
→ 降低學習率: 0.0001 → 0.00005

# 欠擬合（損失高且效果差）
→ 增加 epochs: 15 → 20
→ 增加網絡容量: dim 64 → 128
→ 提高學習率: 0.0001 → 0.0002
```

---

## 🔗 相關資源

### 內部文檔
- [完整訓練指南](docs/KOHYA_TRAINING_GUIDE.md)
- [GPU 優化指南](docs/GPU_OPTIMIZATION_GUIDE.md)
- [項目總覽](LLM_PROVIDER.md)

### 範本文件
- [訓練配置範本](configs/templates/lora_training_template.toml)
- [數據集配置範本](configs/templates/dataset_config_template.toml)

### 外部資源
- [Kohya SS GitHub](https://github.com/kohya-ss/sd-scripts)
- [LoRA 訓練指南](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_README-en.md)

---

## 💡 專業建議

1. **總是使用 TOML 配置**（不要用 CLI 參數）
2. **總是在 kohya_ss 環境中訓練**（不要用 ai_env）
3. **總是測試 1 epoch** 再進行完整訓練
4. **總是記錄配置**（版本控制 TOML 文件）
5. **總是監控 GPU** 確保高利用率

---

## 🚀 SDXL 快速訓練（16GB VRAM）

### 前置條件
- ✅ 已完成 SD 1.5 訓練並找到最佳超參數
- ✅ 有 410 張高品質curated數據集
- ✅ GPU VRAM ≥ 16GB

### 一鍵啟動SDXL訓練

```bash
# 1. 準備SDXL數據集（使用SD 1.5的相同圖片）
bash scripts/training/prepare_kohya_dataset.sh \
  --source-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_final_data \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_sdxl_training \
  --repeat 10 \
  --name luca \
  --validate

# 2. 啟動SDXL訓練（自動處理所有優化）
bash scripts/training/start_sdxl_16gb_training.sh

# 3. 監控VRAM（新終端）
watch -n 1 nvidia-smi

# 4. 訓練完成後評估
conda run -n ai_env python scripts/evaluation/sota_lora_evaluator.py \
  --evaluate-samples \
  --lora-dir /mnt/data/ai_data/models/lora/luca/sdxl_trial1 \
  --sample-dir /mnt/data/ai_data/models/lora/luca/sdxl_trial1/sample \
  --output-dir /mnt/data/ai_data/models/lora/luca/sdxl_trial1/evaluation
```

### SDXL訓練預期

| 項目 | 數值 |
|------|------|
| **訓練時間** | 5-6 小時 |
| **VRAM峰值** | 14-15GB |
| **Checkpoints** | 10 個（每 2 epochs） |
| **視覺改善** | +40% 面部細節, +35% 光影 |

### 關鍵優化技術（已內建在腳本中）

1. **AdamW8bit** - 省40% VRAM
2. **Gradient Checkpointing** - 省30% VRAM
3. **BF16 Mixed Precision** - 省25% VRAM
4. **Latent Caching** - 省2GB VRAM

### OOM故障排除

如果遇到 Out of Memory：

```bash
# 方法1: 降低解析度（修改configs/training/sdxl_16gb_optimized.toml）
resolution = "768,768"  # 從1024降到768

# 方法2: 凍結text encoder
train_text_encoder = false  # 省約2GB VRAM
```

### 完整SDXL文檔

詳細說明見：`docs/guides/SDXL_16GB_TRAINING_GUIDE.md`

---

## 🧪 LoRA 品質測試（訓練後必做）

### 快速啟動測試
```bash
# 一鍵測試 Trial 3.5 最佳檢查點
bash scripts/evaluation/test_trial35_lora.sh
```

### 全面測試（推薦）
```bash
# 測試任何 LoRA checkpoint
python scripts/evaluation/comprehensive_lora_test.py \
    /path/to/lora.safetensors \
    --base-model runwayml/stable-diffusion-v1-5 \
    --seeds 3 \
    --steps 30 \
    --cfg-scale 7.5

# 輸出：
# - outputs/lora_testing/*/TEST_REPORT.md  # 詳細品質報告
# - outputs/lora_testing/*/grids/         # 比較網格圖
# - outputs/lora_testing/*/images/        # 測試圖片（按類別）
```

### 測試類別（自動執行9大類，每類5個prompts × 3個seeds = 135張圖）
1. **Portraits** - 肖像特寫（各種表情）
2. **Full Body** - 全身姿勢與構圖
3. **Angles** - 不同視角（正面/側面/四分之三）
4. **Environments** - 各種背景場景
5. **Expressions** - 情感表達（開心/驚訝/思考）
6. **Actions** - 動態動作（揮手/指向/站立）
7. **Clothing** - 服裝變化
8. **Lighting** - 光照條件（戲劇性/柔和/明亮）
9. **Compositions** - 畫面構圖（中景/特寫/廣角）

### 品質檢查清單
```
✅ 角色身份穩定（所有圖片識別一致）
✅ 提示詞遵循度（準確執行指令）
✅ 無解剖錯誤或偽影
✅ 保持Pixar 3D風格（平滑陰影、PBR材質）
✅ 面部特徵準確穩定
✅ 光照和陰影適當
✅ 無過擬合跡象（不過於類似訓練數據）
✅ 能處理多種角度和姿勢
✅ 背景/環境渲染可接受
✅ 整體品質達到生產標準
```

### 測試參數優化
| 用途 | Seeds | Steps | CFG | 總圖數 | 時間 (RTX 3090) |
|-----|-------|-------|-----|-------|----------------|
| **快速測試** | 1 | 20 | 7.0 | 45張 | ~5分鐘 |
| **標準測試** | 3 | 30 | 7.5 | 135張 | ~15分鐘 |
| **高品質測試** | 5 | 50 | 7.5 | 225張 | ~30分鐘 |

### 下一步決策
```bash
# ✅ 如果測試通過 → 進入SDXL訓練
bash scripts/training/start_sdxl_16gb_training.sh

# ❌ 如果測試失敗 → 檢查問題並調整
# 常見問題：
# - 過擬合 → 減少epochs或增加數據
# - 欠擬合 → 增加epochs或調高學習率
# - 不穩定 → 檢查數據集品質
```

---

## 📌 推薦Workflow

```
✅ 階段1: SD 1.5訓練（Trial 3.5）
   └─ 時間: ~2.2小時
   └─ 目標: 驗證超參數和數據集品質

✅ 階段2: 全面品質測試 ⭐ **NEW**
   └─ 時間: ~15分鐘
   └─ 目標: 生成135張測試圖（9類×5prompts×3seeds）
   └─ 工具: comprehensive_lora_test.py
   └─ 命令: bash scripts/evaluation/test_trial35_lora.sh

✅ 階段3: 評估測試結果
   └─ 時間: ~10分鐘
   └─ 目標: 檢查品質清單，確認無過擬合/欠擬合

✅ 階段4: SDXL訓練（如需更高品質）
   └─ 時間: ~5-6小時
   └─ 目標: 2倍解析度 + 顯著視覺改善

✅ 階段5: 最終對比選擇
   └─ 時間: ~20分鐘
   └─ 目標: 根據需求選擇SD 1.5或SDXL
```

---

## 🔗 相關資源

### 主要指南
- [Luca完整訓練指南](docs/guides/LUCA_TRAINING_GUIDE.md) - 包含SD 1.5和SDXL
- [SDXL 16GB訓練完整文檔](docs/guides/SDXL_16GB_TRAINING_GUIDE.md)
- [替代模型完整指南](docs/guides/ALTERNATIVE_MODELS_FOR_PIXAR_STYLE.md) - ⭐ **NEW** FLUX.1, SD 3.5, Hunyuan等
- [GPU優化指南](docs/GPU_OPTIMIZATION_GUIDE.md)

### 配置文件
- SD 1.5: `configs/training/luca_trial35.toml`
- SDXL: `configs/training/sdxl_16gb_optimized.toml`

---

**最後更新：** 2025-11-15 (v1.2 - 新增LoRA品質測試系統)
**環境：** `kohya_ss` (PyTorch 2.7.1+cu128, bitsandbytes 0.48.2)
**硬體：** RTX 5080 16GB
**支援模型：** SD 1.5 + SDXL (16GB優化)
