# SDXL 訓練快速參考卡

## 🎯 核心概念

**可以遷移**: ✅ 超參數、數據集、訓練策略
**需要調整**: ⚠️ 分辨率、batch size、base model
**完全不同**: ❌ 模型架構、VRAM 需求、訓練時間

---

## 📊 SD1.5 vs SDXL 對比表

| 項目 | SD1.5 | SDXL |
|------|-------|------|
| **分辨率** | 512×512 | 1024×1024 |
| **模型大小** | 0.9B | 2.6B |
| **Text Encoders** | 1 (CLIP-L) | 2 (CLIP-L + OpenCLIP-G) |
| **VRAM 需求** | 8-12 GB | 16-24 GB |
| **Batch Size** | 8-16 | 2-4 |
| **訓練時間/epoch** | ~15 min | ~30-45 min |
| **LoRA 大小 (dim=64)** | ~73 MB | ~190 MB |
| **圖片質量** | 7/10 | 9/10 |

---

## ⚡ 一鍵啟動 SDXL 訓練

### 前置條件
```bash
# 1. SD1.5 優化已完成，提取最佳參數
BEST_LR=0.0003
BEST_DIM=64
BEST_ALPHA=32
BEST_OPTIMIZER="AdamW8bit"
BEST_SCHEDULER="cosine_with_restarts"
BEST_GRAD_ACCUM=2
BEST_EPOCHS=12

# 2. 下載 SDXL base model
cd /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```

### 訓練命令（複製即用）
```bash
cd /mnt/c/AI_LLM_projects/kohya_ss/sd-scripts

nohup conda run -n kohya_ss python train_network.py \
  --dataset_config /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/sdxl/luca_human_dataset_sdxl.toml \
  --pretrained_model_name_or_path /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl/sd_xl_base_1.0.safetensors \
  --output_dir /mnt/data/ai_data/models/lora/luca/sdxl_v1 \
  --output_name luca_sdxl_v1 \
  --network_module networks.lora \
  --network_dim $BEST_DIM \
  --network_alpha $BEST_ALPHA \
  --learning_rate $BEST_LR \
  --text_encoder_lr $(echo "$BEST_LR * 0.67" | bc -l) \
  --unet_lr $BEST_LR \
  --max_train_epochs $BEST_EPOCHS \
  --save_every_n_epochs 2 \
  --save_model_as safetensors \
  --save_precision fp16 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --gradient_accumulation_steps $BEST_GRAD_ACCUM \
  --optimizer_type $BEST_OPTIMIZER \
  --lr_scheduler $BEST_SCHEDULER \
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

---

## 🔧 關鍵參數說明

### SDXL 特有參數（必須添加）
```bash
--text_encoder_lr 0.0002       # 雙 text encoder，獨立設置
--unet_lr 0.0003               # U-Net 學習率
--max_token_length 225         # SDXL 支持更長 tokens
--xformers                     # 記憶體優化（必須）
```

### 從 SD1.5 遷移的參數（直接使用）
```bash
--network_dim 64               # SD1.5 最佳值
--network_alpha 32             # SD1.5 最佳值
--learning_rate 0.0003         # SD1.5 最佳值
--optimizer_type AdamW8bit     # SD1.5 最佳值
--lr_scheduler cosine_with_restarts  # SD1.5 最佳值
--gradient_accumulation_steps 2      # SD1.5 最佳值
--max_train_epochs 12          # SD1.5 最佳值（或 +20%）
```

### 需要調整的參數
```bash
--batch_size 4                 # SD1.5: 8 → SDXL: 4（VRAM 限制）
--text_encoder_lr              # 新增，約為 learning_rate × 0.5-0.8
```

---

## 📈 監控命令

### 實時日誌
```bash
tail -f /mnt/data/ai_data/models/lora/luca/sdxl_v1/training.log
```

### 查看 Checkpoints
```bash
watch -n 60 'ls -lh /mnt/data/ai_data/models/lora/luca/sdxl_v1/*.safetensors'
```

### TensorBoard 可視化
```bash
tensorboard --logdir /mnt/data/ai_data/models/lora/luca/sdxl_v1/logs --port 6007 --bind_all
```

### 訓練進度估算
```bash
# 查看當前 epoch
grep -oP "Epoch \d+/\d+" /mnt/data/ai_data/models/lora/luca/sdxl_v1/training.log | tail -1

# 查看 loss 趨勢
grep "loss:" /mnt/data/ai_data/models/lora/luca/sdxl_v1/training.log | tail -20
```

---

## 🧪 評估 SDXL Checkpoint

### 單個 Checkpoint 測試
```bash
conda run -n kohya_ss python /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/evaluation/evaluate_single_checkpoint.py \
  --checkpoint /mnt/data/ai_data/models/lora/luca/sdxl_v1/luca_sdxl_v1-000008.safetensors \
  --base-model /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl/sd_xl_base_1.0.safetensors \
  --output-dir /mnt/data/ai_data/models/lora/luca/sdxl_v1/eval_epoch8 \
  --num-samples 16 \
  --device cuda \
  --resolution 1024
```

### 對比 SD1.5 vs SDXL 質量
生成相同 prompt 的圖片並比較：
- **SD1.5**: 512×512, 較快, 質量中等
- **SDXL**: 1024×1024, 較慢, 質量優秀

---

## ⚠️ 常見問題快速解決

### ❌ OOM (Out of Memory)
```bash
# 解決方案 1：降低 batch size
--batch_size 2
--gradient_accumulation_steps 4  # 等效 batch size = 8

# 解決方案 2：啟用記憶體優化
--xformers  # 必須
--gradient_checkpointing  # 必須

# 解決方案 3：極端情況
--lowvram
--medvram
```

### 🐌 訓練太慢
- **正常現象**：SDXL 預期慢 2-3 倍
- **無法加速**：模型大小和分辨率決定
- **建議**：使用 overnight 訓練

### 🖼️ 圖片質量不佳
- **檢查**：是否使用了 SD1.5 最佳超參數？
- **檢查**：dataset 配置是否正確（resolution=1024）？
- **檢查**：是否在正確的 epoch（通常 epoch 6-10 最佳）？

### 📦 LoRA 檔案太大
- **正常**：SDXL LoRA 約 2.5 倍 SD1.5 大小
- **優化**：降低 network_dim（但可能影響質量）
- **建議**：保持 dim=64 或 128

---

## 🎓 最佳實踐 Checklist

- [ ] SD1.5 優化已完成，最佳參數已提取
- [ ] SDXL base model 已下載
- [ ] 數據集路徑檢查（使用相同的 SD1.5 數據集）
- [ ] VRAM 充足（至少 16GB，建議 24GB）
- [ ] 使用 `--xformers` 和 `--gradient_checkpointing`
- [ ] `text_encoder_lr` 設為 `learning_rate × 0.67`
- [ ] `batch_size` 降低到 2-4
- [ ] `max_token_length` 設為 225
- [ ] 使用 nohup 背景運行
- [ ] 啟動 TensorBoard 監控
- [ ] 每 2 epochs 保存 checkpoint
- [ ] 預留 6-10 小時訓練時間

---

## 📂 目錄結構參考

```
/mnt/data/ai_data/models/lora/luca/
├── optimization_overnight/           # SD1.5 優化結果
│   ├── trial_0025/                  # 假設最佳 trial
│   │   ├── params.json              # 提取超參數
│   │   └── lora_trial_25.safetensors
│   └── CONVERGENCE_ALERT.txt
│
└── sdxl_v1/                         # SDXL 訓練結果
    ├── luca_sdxl_v1-000002.safetensors  (epoch 2)
    ├── luca_sdxl_v1-000004.safetensors  (epoch 4)
    ├── luca_sdxl_v1-000006.safetensors  (epoch 6)
    ├── luca_sdxl_v1-000008.safetensors  (epoch 8)
    ├── luca_sdxl_v1-000010.safetensors  (epoch 10)
    ├── luca_sdxl_v1.safetensors  (final, epoch 12)
    ├── training.log
    ├── logs/  (TensorBoard)
    └── eval_epoch8/  (測試圖片)
```

---

## 🚀 快速決策樹

```
SD1.5 優化完成？
 ├─ 是 → 提取最佳參數 → 啟動 SDXL 訓練
 └─ 否 → 等待完成（監控收斂狀態）

VRAM 充足（≥16GB）？
 ├─ 是 → batch_size=4
 └─ 否 → batch_size=2 + gradient_accumulation_steps=4

需要極致質量？
 ├─ 是 → SDXL（1024×1024）
 └─ 否 → SD1.5（512×512）足夠

時間緊迫？
 ├─ 是 → 使用 SD1.5（快 2-3x）
 └─ 否 → 使用 SDXL（質量更好）
```

---

## 📞 需要幫助？

**詳細指南**: 查看 `SD15_TO_SDXL_MIGRATION.md`
**監控腳本**: 使用 `monitor_optimization_progress.sh`
**評估工具**: 使用 `evaluate_single_checkpoint.py`

---

**最後更新**: 2025-11-12
**版本**: 1.0
