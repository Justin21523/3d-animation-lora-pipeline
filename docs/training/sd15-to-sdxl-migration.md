# SD1.5 → SDXL LoRA 訓練遷移指南

## 概述

本指南說明如何將在 **SD1.5** (512×512) 上優化的超參數配置遷移到 **SDXL** (1024×1024)，以獲得更高品質的圖片生成結果。

---

## ✅ 核心結論

### 可以遷移的內容
- ✅ **超參數配置**（learning rate、network_dim、optimizer 等）
- ✅ **訓練數據集**（相同的圖片和 captions）
- ✅ **訓練策略**（scheduler、warmup、gradient accumulation）
- ✅ **數據增強政策**（no flip、no color aug for 3D）

### 需要調整的內容
- ⚠️ **分辨率**：512 → 1024
- ⚠️ **Batch size**：可能需要降低（VRAM 限制）
- ⚠️ **訓練時間**：預期增加 2-3 倍
- ⚠️ **Base model**：v1-5-pruned → SDXL base/checkpoint
- ⚠️ **Network dim**：可能需要微調（SDXL 更大）

### 完全不同的部分
- ❌ **模型架構**：SD1.5 = U-Net + CLIP；SDXL = 2x larger U-Net + dual text encoders
- ❌ **VRAM 需求**：SD1.5 ~8GB → SDXL ~16-24GB
- ❌ **Text encoder**：1 個 → 2 個（CLIP-L + OpenCLIP-G）

---

## 📋 完整遷移流程

### **階段 1：等待 SD1.5 優化完成**

當前狀態：
```bash
# 檢查優化進度
bash /mnt/data/ai_data/models/lora/luca/optimization_overnight/monitor_optimization_progress.sh

# 查看收斂狀態
tail -30 /mnt/data/ai_data/models/lora/luca/optimization_overnight/convergence_monitor.log

# 查看最新 trial
ls -lt /mnt/data/ai_data/models/lora/luca/optimization_overnight/trial_* | head -5
```

**預期時間**：1.5-2 天（50 trials）

---

### **階段 2：提取最佳超參數**

當優化完成後：

```bash
# 1. 查看最終報告
cat /mnt/data/ai_data/models/lora/luca/optimization_overnight/CONVERGENCE_ALERT.txt

# 2. 找到最佳 trial 編號（假設是 trial_0025）
BEST_TRIAL="trial_0025"

# 3. 提取參數
cat /mnt/data/ai_data/models/lora/luca/optimization_overnight/$BEST_TRIAL/params.json
```

**輸出示例**：
```json
{
  "learning_rate": 0.0003,
  "network_dim": 64,
  "network_alpha": 32,
  "optimizer_type": "AdamW8bit",
  "lr_scheduler": "cosine_with_restarts",
  "gradient_accumulation_steps": 2,
  "max_train_epochs": 12
}
```

**記錄這些值 → 用於 SDXL 訓練**

---

### **階段 3：準備 SDXL 環境**

#### 3.1 下載 SDXL Base Model

```bash
# 創建 SDXL 模型目錄
mkdir -p /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl

# 選項 A：HuggingFace 官方 SDXL 1.0
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 \
  --local-dir /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl/base-1.0

# 選項 B：使用 .safetensors 單檔（推薦）
# 下載地址：https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```

#### 3.2 驗證數據集（相同數據可用於 SDXL）

```bash
# 檢查圖片和 captions
ls /mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset_v2_smart/luca_human/images/*.png | wc -l
ls /mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset_v2_smart/luca_human/captions/*.txt | wc -l

# 應該看到相同數量的檔案（例如：359 張圖片 + 359 個 captions）
```

✅ **無需重新處理圖片！SDXL 訓練時會自動 resize 到 1024×1024**

---

### **階段 4：創建 SDXL 訓練配置**

#### 4.1 使用已創建的 SDXL 數據集配置

文件已創建：`/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/sdxl/luca_human_dataset_sdxl.toml`

**關鍵差異**：
```toml
# SD1.5
resolution = 512
batch_size = 8
min_bucket_reso = 384
max_bucket_reso = 768

# SDXL
resolution = 1024
batch_size = 4  # 降低以適應 VRAM
min_bucket_reso = 768
max_bucket_reso = 1536
```

#### 4.2 構建 SDXL 訓練命令（使用 SD1.5 最佳參數）

假設 SD1.5 最佳參數為：
- learning_rate: 0.0003
- network_dim: 64
- network_alpha: 32
- optimizer_type: AdamW8bit
- lr_scheduler: cosine_with_restarts
- gradient_accumulation_steps: 2
- max_train_epochs: 12

**SDXL 訓練命令**：
```bash
cd /mnt/c/AI_LLM_projects/kohya_ss/sd-scripts

conda run -n kohya_ss python train_network.py \
  --dataset_config /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/sdxl/luca_human_dataset_sdxl.toml \
  --pretrained_model_name_or_path /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl/sd_xl_base_1.0.safetensors \
  --output_dir /mnt/data/ai_data/models/lora/luca/sdxl_v1 \
  --output_name luca_sdxl_v1 \
  --network_module networks.lora \
  --network_dim 64 \
  --network_alpha 32 \
  --learning_rate 0.0003 \
  --text_encoder_lr 0.0002 \
  --unet_lr 0.0003 \
  --max_train_epochs 12 \
  --save_every_n_epochs 2 \
  --save_model_as safetensors \
  --save_precision fp16 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --gradient_accumulation_steps 2 \
  --optimizer_type AdamW8bit \
  --lr_scheduler cosine_with_restarts \
  --lr_scheduler_num_cycles 3 \
  --lr_warmup_steps 100 \
  --logging_dir /mnt/data/ai_data/models/lora/luca/sdxl_v1/logs \
  --log_with tensorboard \
  --seed 42 \
  --clip_skip 2 \
  --cache_latents \
  --cache_latents_to_disk \
  --max_data_loader_n_workers 8 \
  --persistent_data_loader_workers \
  --xformers \
  --max_token_length 225 \
  --bucket_reso_steps 64 \
  --bucket_no_upscale
```

**SDXL 特有參數說明**：
- `--text_encoder_lr`：SDXL 有 2 個 text encoders，需要獨立設置
- `--unet_lr`：U-Net 學習率（通常與 learning_rate 相同）
- `--max_token_length 225`：SDXL 支持更長的 tokens（SD1.5 = 77）
- `--xformers`：記憶體優化（必須）

---

### **階段 5：執行 SDXL 訓練**

#### 5.1 啟動訓練（使用 nohup 背景運行）

```bash
cd /mnt/c/AI_LLM_projects/kohya_ss/sd-scripts

nohup conda run -n kohya_ss python train_network.py \
  --dataset_config /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/sdxl/luca_human_dataset_sdxl.toml \
  --pretrained_model_name_or_path /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl/sd_xl_base_1.0.safetensors \
  --output_dir /mnt/data/ai_data/models/lora/luca/sdxl_v1 \
  --output_name luca_sdxl_v1 \
  --network_module networks.lora \
  --network_dim 64 \
  --network_alpha 32 \
  --learning_rate 0.0003 \
  --text_encoder_lr 0.0002 \
  --unet_lr 0.0003 \
  --max_train_epochs 12 \
  --save_every_n_epochs 2 \
  --save_model_as safetensors \
  --save_precision fp16 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --gradient_accumulation_steps 2 \
  --optimizer_type AdamW8bit \
  --lr_scheduler cosine_with_restarts \
  --lr_scheduler_num_cycles 3 \
  --lr_warmup_steps 100 \
  --logging_dir /mnt/data/ai_data/models/lora/luca/sdxl_v1/logs \
  --log_with tensorboard \
  --seed 42 \
  --clip_skip 2 \
  --cache_latents \
  --cache_latents_to_disk \
  --max_data_loader_n_workers 8 \
  --persistent_data_loader_workers \
  --xformers \
  --max_token_length 225 \
  --bucket_reso_steps 64 \
  --bucket_no_upscale \
  > /mnt/data/ai_data/models/lora/luca/sdxl_v1/training.log 2>&1 &

echo "SDXL 訓練已啟動，PID: $!"
```

#### 5.2 監控訓練進度

```bash
# 查看實時日誌
tail -f /mnt/data/ai_data/models/lora/luca/sdxl_v1/training.log

# 查看已生成的 checkpoints
ls -lh /mnt/data/ai_data/models/lora/luca/sdxl_v1/*.safetensors

# 查看 TensorBoard
tensorboard --logdir /mnt/data/ai_data/models/lora/luca/sdxl_v1/logs --port 6007 --bind_all
```

---

### **階段 6：評估 SDXL LoRA**

#### 6.1 測試單個 checkpoint

```bash
conda run -n kohya_ss python /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/evaluation/evaluate_single_checkpoint.py \
  --checkpoint /mnt/data/ai_data/models/lora/luca/sdxl_v1/luca_sdxl_v1-000008.safetensors \
  --base-model /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl/sd_xl_base_1.0.safetensors \
  --output-dir /mnt/data/ai_data/models/lora/luca/sdxl_v1/evaluation_epoch8 \
  --num-samples 16 \
  --device cuda \
  --resolution 1024
```

#### 6.2 對比 SD1.5 vs SDXL 質量

生成相同 prompts 的圖片：

**SD1.5 (512×512)**：
```bash
# 使用 SD1.5 最佳 checkpoint
lora_path=/mnt/data/ai_data/models/lora/luca/optimization_overnight/trial_0025/lora_trial_25.safetensors
```

**SDXL (1024×1024)**：
```bash
# 使用 SDXL 訓練的 checkpoint
lora_path=/mnt/data/ai_data/models/lora/luca/sdxl_v1/luca_sdxl_v1-000008.safetensors
```

**測試 prompt**：
```
"a 3d animated human character, young boy with brown hair, blue eyes, wearing a striped shirt, pixar style, smooth shading, soft lighting, detailed facial features, cinematic quality"
```

---

## 🔍 SD1.5 vs SDXL 關鍵差異

### 1. **分辨率與細節**

| 特性 | SD1.5 | SDXL |
|------|-------|------|
| Native resolution | 512×512 | 1024×1024 |
| 細節層次 | 中等 | 高（2x pixels） |
| 面部清晰度 | 中等 | 優秀 |
| 紋理質感 | 基本 | 細膩 |
| 畫質感受 | 「遊戲截圖」| 「電影劇照」|

### 2. **模型架構**

| 組件 | SD1.5 | SDXL |
|------|-------|------|
| U-Net 參數量 | ~0.9B | ~2.6B (3x larger) |
| Text encoders | 1x CLIP-L | 2x (CLIP-L + OpenCLIP-G) |
| Cross-attention | Single | Dual pooled + text |
| 條件化能力 | 基本 | 強（更理解複雜 prompts） |

### 3. **訓練成本**

| 資源 | SD1.5 | SDXL |
|------|-------|------|
| VRAM 需求 | 8-12 GB | 16-24 GB |
| 訓練時間 (1 epoch) | ~15 分鐘 | ~30-45 分鐘 (2-3x) |
| Batch size | 8-16 | 2-4 |
| 總訓練時長 (12 epochs) | 3-4 小時 | 6-10 小時 |

### 4. **LoRA 文件大小**

| Network Dim | SD1.5 LoRA | SDXL LoRA |
|-------------|-----------|----------|
| 32 | ~37 MB | ~95 MB |
| 64 | ~73 MB | ~190 MB |
| 128 | ~146 MB | ~380 MB |

**建議**：SDXL 使用 network_dim=64 或 128（與 SD1.5 相同值即可）

---

## ⚙️ 超參數遷移建議

### 直接使用（無需調整）

✅ **Learning rate**：SD1.5 的最佳 learning rate 通常直接適用於 SDXL
- 原因：LoRA 訓練的學習率主要取決於**數據集特性**和**訓練策略**，而非基礎模型大小

✅ **Optimizer**：AdamW、AdamW8bit、Lion、Prodigy 等都相同

✅ **LR Scheduler**：cosine、cosine_with_restarts、polynomial 等策略相同

✅ **Gradient accumulation steps**：保持相同（但可能因 batch size 降低而需要調整）

### 需要微調

⚠️ **Batch size**：
- SD1.5：8-16
- SDXL：2-4（因 VRAM 限制）
- 如果 batch size 降低 2 倍，考慮將 gradient_accumulation_steps 增加 2 倍以保持等效 batch size

⚠️ **Network dim**：
- SD1.5 最佳值可直接使用
- 但 SDXL 更大，可選擇性測試更高的 dim（如 SD1.5 用 64 → SDXL 試 64 或 96）

⚠️ **Max train epochs**：
- SDXL 可能需要稍多 epochs（+20%）來充分收斂
- 建議：如果 SD1.5 最佳是 12 epochs，SDXL 可嘗試 12-15 epochs

### 新增設置

➕ **Text encoder learning rates**（SDXL 專有）：
```bash
--text_encoder_lr 0.0002    # 通常設為 learning_rate 的 0.5-0.8x
--unet_lr 0.0003            # 與 learning_rate 相同
```

➕ **Max token length**（SDXL 支持更長）：
```bash
--max_token_length 225      # SD1.5 = 77, SDXL = 75+150 = 225
```

➕ **Xformers 記憶體優化**（必須）：
```bash
--xformers  # 降低 VRAM 使用，加速訓練
```

---

## 📊 預期結果對比

### 圖片質量提升

| 評估指標 | SD1.5 | SDXL | 提升幅度 |
|---------|-------|------|---------|
| 分辨率 | 512×512 | 1024×1024 | **4x pixels** |
| 面部細節 | 7/10 | 9/10 | **+28%** |
| 紋理清晰度 | 6/10 | 9/10 | **+50%** |
| 光影細膩度 | 7/10 | 9/10 | **+28%** |
| Prompt 理解 | 7/10 | 9/10 | **+28%** |
| 整體視覺衝擊 | 「不錯」| 「驚艷」| **質變** |

### 實際使用場景

**SD1.5 適合**：
- 快速原型測試
- 低分辨率應用（社交媒體、網頁）
- VRAM 受限環境

**SDXL 適合**：
- 專業級內容生成
- 高分辨率輸出（印刷、海報、視頻）
- 追求最佳視覺質量
- 商業用途

---

## 🚀 最佳實踐建議

### 1. **迭代策略：先 SD1.5，後 SDXL**

✅ **推薦流程**（當前正在執行）：
1. 在 SD1.5 上進行**大規模超參數優化**（50 trials）→ 找到最佳配置
2. 使用最佳配置在 SDXL 上訓練 → 獲得高質量 LoRA
3. （可選）在 SDXL 上進行**小規模微調優化**（10-20 trials）→ 針對 SDXL 特性進一步優化

**優點**：
- SD1.5 訓練快 → 節省優化時間和成本
- 超參數高度可遷移 → 減少 SDXL 搜索空間
- 最終 SDXL 質量有保障

### 2. **VRAM 管理**

如果遇到 VRAM 不足：

```bash
# 選項 A：降低 batch size + 增加 gradient accumulation
--batch_size 2
--gradient_accumulation_steps 4  # 等效 batch size = 2*4 = 8

# 選項 B：使用 8bit optimizer
--optimizer_type AdamW8bit  # 節省 ~40% VRAM

# 選項 C：啟用 CPU offload（極端情況）
--lowvram
--medvram
```

### 3. **數據集無需重新處理**

✅ **512×512 圖片可直接用於 SDXL 訓練**
- Kohya 腳本會自動 upscale 到 1024×1024
- Bucket 系統會處理不同 aspect ratios
- Captions 完全相同

❌ **不建議手動 upscale 圖片**：
- 不會改善質量（SDXL 訓練時會 resize）
- 浪費磁碟空間（4x 大小）
- 增加 I/O 時間

### 4. **Checkpoint 評估頻率**

```bash
# SDXL 訓練較慢，建議每 2 epochs 保存 checkpoint
--save_every_n_epochs 2

# 評估更少的 checkpoints（SDXL 生成慢）
--num-samples 8  # 而非 SD1.5 的 16
```

### 5. **SDXL 專用優化（可選）**

當 SD1.5 優化完成後，如果想進一步優化 SDXL：

```bash
# 創建 SDXL 優化目錄
mkdir -p /mnt/data/ai_data/models/lora/luca/optimization_sdxl

# 運行小規模優化（10-20 trials）
conda run -n kohya_ss python \
  /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/optimization/optuna_hyperparameter_search.py \
  --dataset-config /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/sdxl/luca_human_dataset_sdxl.toml \
  --base-model /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl/sd_xl_base_1.0.safetensors \
  --output-dir /mnt/data/ai_data/models/lora/luca/optimization_sdxl \
  --study-name luca_sdxl_optimization \
  --n-trials 20 \
  --device cuda \
  --init-params /mnt/data/ai_data/models/lora/luca/optimization_overnight/best_params.json
```

**優化範圍建議**（SDXL 特定）：
- `text_encoder_lr`: [0.00015, 0.00025]（圍繞 SD1.5 的 learning_rate）
- `unet_lr`: 使用 SD1.5 最佳 learning_rate
- `network_dim`: [64, 96, 128]（比 SD1.5 略高）
- 其餘參數：固定為 SD1.5 最佳值

---

## 📂 文件結構

### SD1.5 優化結果
```
/mnt/data/ai_data/models/lora/luca/optimization_overnight/
├── trial_0001/
├── trial_0002/
├── ...
├── trial_0025/  ← 假設最佳
│   ├── params.json
│   ├── lora_trial_25.safetensors
│   └── realtime_evaluations/
├── optimization.log
└── CONVERGENCE_ALERT.txt
```

### SDXL 訓練結果
```
/mnt/data/ai_data/models/lora/luca/sdxl_v1/
├── luca_sdxl_v1-000002.safetensors  (epoch 2)
├── luca_sdxl_v1-000004.safetensors  (epoch 4)
├── luca_sdxl_v1-000006.safetensors  (epoch 6)
├── luca_sdxl_v1-000008.safetensors  (epoch 8)
├── luca_sdxl_v1-000010.safetensors  (epoch 10)
├── luca_sdxl_v1.safetensors  (final, epoch 12)
├── training.log
├── logs/  (TensorBoard)
└── evaluation_epoch8/  (測試圖片)
```

---

## 🎯 實戰範例：完整命令

假設 SD1.5 優化完成，最佳參數為：
- learning_rate: 0.0003
- network_dim: 64
- network_alpha: 32
- optimizer_type: AdamW8bit
- lr_scheduler: cosine_with_restarts
- gradient_accumulation_steps: 2
- max_train_epochs: 12

### 步驟 1：下載 SDXL base model
```bash
cd /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```

### 步驟 2：啟動 SDXL 訓練
```bash
cd /mnt/c/AI_LLM_projects/kohya_ss/sd-scripts

nohup conda run -n kohya_ss python train_network.py \
  --dataset_config /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/sdxl/luca_human_dataset_sdxl.toml \
  --pretrained_model_name_or_path /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl/sd_xl_base_1.0.safetensors \
  --output_dir /mnt/data/ai_data/models/lora/luca/sdxl_v1 \
  --output_name luca_sdxl_v1 \
  --network_module networks.lora \
  --network_dim 64 \
  --network_alpha 32 \
  --learning_rate 0.0003 \
  --text_encoder_lr 0.0002 \
  --unet_lr 0.0003 \
  --max_train_epochs 12 \
  --save_every_n_epochs 2 \
  --save_model_as safetensors \
  --save_precision fp16 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --gradient_accumulation_steps 2 \
  --optimizer_type AdamW8bit \
  --lr_scheduler cosine_with_restarts \
  --lr_scheduler_num_cycles 3 \
  --lr_warmup_steps 100 \
  --logging_dir /mnt/data/ai_data/models/lora/luca/sdxl_v1/logs \
  --log_with tensorboard \
  --seed 42 \
  --clip_skip 2 \
  --cache_latents \
  --cache_latents_to_disk \
  --max_data_loader_n_workers 8 \
  --persistent_data_loader_workers \
  --xformers \
  --max_token_length 225 \
  --bucket_reso_steps 64 \
  --bucket_no_upscale \
  > /mnt/data/ai_data/models/lora/luca/sdxl_v1/training.log 2>&1 &

echo "SDXL 訓練已啟動，PID: $!"
```

### 步驟 3：監控訓練
```bash
# 實時日誌
tail -f /mnt/data/ai_data/models/lora/luca/sdxl_v1/training.log

# 查看生成的 checkpoints
watch -n 60 'ls -lh /mnt/data/ai_data/models/lora/luca/sdxl_v1/*.safetensors'

# TensorBoard 可視化
tensorboard --logdir /mnt/data/ai_data/models/lora/luca/sdxl_v1/logs --port 6007 --bind_all
```

### 步驟 4：評估最佳 checkpoint
```bash
conda run -n kohya_ss python /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/evaluation/evaluate_single_checkpoint.py \
  --checkpoint /mnt/data/ai_data/models/lora/luca/sdxl_v1/luca_sdxl_v1-000008.safetensors \
  --base-model /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl/sd_xl_base_1.0.safetensors \
  --output-dir /mnt/data/ai_data/models/lora/luca/sdxl_v1/evaluation_final \
  --num-samples 16 \
  --device cuda \
  --resolution 1024
```

---

## ❓ FAQ

### Q1: SD1.5 和 SDXL 的 LoRA 可以互相使用嗎？
**A**: ❌ **不能**。SD1.5 和 SDXL 的模型架構不同，LoRA 權重維度不匹配。必須分別訓練。

### Q2: 必須重新處理圖片到 1024×1024 嗎？
**A**: ❌ **不需要**。Kohya 腳本會自動處理 resize，512×512 圖片可直接用於 SDXL 訓練。

### Q3: SDXL 訓練一定比 SD1.5 慢嗎？
**A**: ✅ **是的**，預期慢 2-3 倍。原因：
- 模型更大（2.6B vs 0.9B）
- 分辨率更高（4x pixels）
- 雙 text encoders

### Q4: 如果 VRAM 不足怎麼辦？
**A**: 選項：
1. 降低 batch_size 到 2（配合 gradient_accumulation_steps）
2. 使用 AdamW8bit optimizer
3. 啟用 --xformers（必須）
4. 極端情況：--lowvram 或 --medvram

### Q5: SD1.5 最佳 network_dim 是 64，SDXL 應該用多少？
**A**: **64 或 128**。SDXL 更大，可選擇性增加 dim，但通常 SD1.5 的最佳值直接適用。

### Q6: 訓練完 SDXL 後，還需要優化嗎？
**A**: **可選**。如果 SD1.5 優化已找到好配置，直接訓練通常足夠。若追求極致，可做小規模 SDXL 專用優化（10-20 trials）。

### Q7: SDXL 生成速度比 SD1.5 慢嗎？
**A**: ✅ **是的**，推理時間約慢 1.5-2x。但質量提升值得等待。

### Q8: 可以在 SDXL 上繼續使用相同的 prompts 嗎？
**A**: ✅ **可以**，且 SDXL 通常理解得更好（雙 text encoders），可嘗試更複雜的描述。

---

## 🎓 總結

### ✅ 核心要點
1. **超參數可遷移**：SD1.5 優化結果直接適用於 SDXL
2. **數據集相同**：無需重新處理圖片和 captions
3. **質量顯著提升**：1024×1024 + 更強模型 = 視覺質變
4. **成本可接受**：訓練時間 2-3x，VRAM 需求增加

### 📝 推薦工作流程
1. **當前**：完成 SD1.5 優化（50 trials）→ 找最佳配置
2. **下一步**：使用最佳配置訓練 SDXL → 獲得高質量 1024×1024 LoRA
3. **可選**：SDXL 專用微調優化（10-20 trials）→ 進一步提升

### 🚀 立即行動
等當前 SD1.5 優化完成後（預計 1.5-2 天），即可：
1. 提取最佳超參數
2. 下載 SDXL base model
3. 使用本指南的命令啟動 SDXL 訓練
4. 享受 4K 品質的角色生成！

---

**文檔版本**: 1.0
**最後更新**: 2025-11-12
**適用於**: Kohya SS sd-scripts, Stable Diffusion 1.5 & SDXL
