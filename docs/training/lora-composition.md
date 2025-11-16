# LoRA Composition 快速啟動指南

## 🎯 目標

從當前的 **Character LoRA** 擴展到**多類型 LoRA 生態系統**，實現：
- ✅ Luca 角色 LoRA（當前）
- 🆕 Portorosso 背景 LoRA
- 🆕 動作 LoRA（跑步、跳躍等）
- 🆕 表情 LoRA（開心、驚訝等）

最終目標：**生成 "Luca 在 Portorosso 奔跑並露出開心笑容" 的圖片**！

---

## 📋 工作流程總覽

```
當前狀態（進行中）
└─> Character LoRA 優化（50 trials）
    └─> 提取最佳超參數

下一階段（Character LoRA 完成後）
└─> Background LoRA 訓練
    ├─> 提取背景 layers
    ├─> Background inpainting
    ├─> 場景聚類
    ├─> 訓練 Background LoRA（使用 Character LoRA 最佳參數）
    └─> Pose/Expression LoRA 訓練（平行進行）

最終階段
└─> LoRA Composition 測試
    └─> 組合 Character + Background + Pose + Expression
        └─> 生成測試圖片
```

---

## 🚀 階段 1：Character LoRA 優化（當前）

### 狀態
✅ 50 trials 優化運行中（PID 80250）
✅ 自動收斂監控運行中（PID 93767）
✅ 預計 1.5-2 天完成

### 監控命令
```bash
# 查看優化進度
bash /mnt/data/ai_data/models/lora/luca/optimization_overnight/monitor_optimization_progress.sh

# 查看收斂狀態
tail -30 /mnt/data/ai_data/models/lora/luca/optimization_overnight/convergence_monitor.log
```

### 完成後動作
```bash
# 1. 查看收斂報告
cat /mnt/data/ai_data/models/lora/luca/optimization_overnight/CONVERGENCE_ALERT.txt

# 2. 提取最佳參數
BEST_TRIAL=$(grep "Best trial:" /mnt/data/ai_data/models/lora/luca/optimization_overnight/CONVERGENCE_ALERT.txt | grep -oP 'Trial \d+' | grep -oP '\d+')
cat /mnt/data/ai_data/models/lora/luca/optimization_overnight/trial_$(printf '%04d' $BEST_TRIAL)/params.json

# 3. 保存最佳參數
cp /mnt/data/ai_data/models/lora/luca/optimization_overnight/trial_$(printf '%04d' $BEST_TRIAL)/params.json \
   /mnt/data/ai_data/models/lora/luca/best_hyperparameters.json
```

---

## 🚀 階段 2：Background LoRA 訓練

### 前置條件
✅ Character LoRA 優化已完成
✅ 最佳超參數已提取
✅ SAM2 分割時已自動生成背景 layers

### 步驟 2.1：檢查現有背景數據
```bash
# 檢查背景 layers（應該已經存在）
ls /mnt/data/ai_data/datasets/3d-anime/luca/segmented/background/*.png | wc -l
```

如果背景數據不存在，需要重新運行分割：
```bash
python scripts/generic/segmentation/layered_segmentation.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/frames \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/segmented \
  --model sam2 \
  --extract-characters  # 會同時生成 background/
```

### 步驟 2.2：Background Inpainting（移除角色殘留）
```bash
python scripts/generic/inpainting/background_inpainting.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/segmented/background \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_clean \
  --method lama \
  --device cuda \
  --log-file /tmp/background_inpainting.log
```

**預期時間**：約 30-60 分鐘（取決於幀數）

### 步驟 2.3：場景聚類（按位置/風格分組）
```bash
python scripts/generic/clustering/character_clustering.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_clean \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/scene_clusters \
  --min-cluster-size 15 \
  --min-samples 3 \
  --similarity-threshold 0.75 \
  --use-face-detection false  # 背景不需要人臉檢測
```

**預期結果**：
```
scene_clusters/
├── character_0/  (Portorosso town center)
├── character_1/  (Beach scenes)
├── character_2/  (Indoor scenes)
└── noise/
```

### 步驟 2.4：選擇主要場景並準備訓練數據
```bash
# 假設 character_0 是 Portorosso 主要場景
python scripts/generic/training/prepare_training_data.py \
  --character-dirs /mnt/data/ai_data/datasets/3d-anime/luca/scene_clusters/character_0 \
  --output-dir /mnt/data/ai_data/training_data/portorosso_background \
  --character-name "portorosso" \
  --generate-captions \
  --caption-model qwen2_vl \
  --caption-prefix "3d animated background, italian seaside town, pixar style" \
  --target-size 300
```

### 步驟 2.5：創建 Background LoRA 訓練配置
```bash
# 複製並修改 Character LoRA 配置
cp /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/luca_human_dataset.toml \
   /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/portorosso_background_dataset.toml

# 編輯配置（修改 image_dir）
```

**修改內容**：
```toml
[[datasets.subsets]]
image_dir = "/mnt/data/ai_data/training_data/portorosso_background/images"
num_repeats = 1
shuffle_caption = true
keep_tokens = 3
caption_extension = ".txt"
color_aug = false
flip_aug = false
```

### 步驟 2.6：訓練 Background LoRA（使用最佳超參數）
```bash
# 從 best_hyperparameters.json 讀取參數
BEST_LR=$(cat /mnt/data/ai_data/models/lora/luca/best_hyperparameters.json | python3 -c "import sys, json; print(json.load(sys.stdin)['learning_rate'])")
BEST_DIM=$(cat /mnt/data/ai_data/models/lora/luca/best_hyperparameters.json | python3 -c "import sys, json; print(json.load(sys.stdin)['network_dim'])")
BEST_ALPHA=$(cat /mnt/data/ai_data/models/lora/luca/best_hyperparameters.json | python3 -c "import sys, json; print(json.load(sys.stdin)['network_alpha'])")
BEST_OPTIMIZER=$(cat /mnt/data/ai_data/models/lora/luca/best_hyperparameters.json | python3 -c "import sys, json; print(json.load(sys.stdin)['optimizer_type'])")
BEST_SCHEDULER=$(cat /mnt/data/ai_data/models/lora/luca/best_hyperparameters.json | python3 -c "import sys, json; print(json.load(sys.stdin)['lr_scheduler'])")
BEST_GRAD_ACCUM=$(cat /mnt/data/ai_data/models/lora/luca/best_hyperparameters.json | python3 -c "import sys, json; print(json.load(sys.stdin)['gradient_accumulation_steps'])")
BEST_EPOCHS=$(cat /mnt/data/ai_data/models/lora/luca/best_hyperparameters.json | python3 -c "import sys, json; print(json.load(sys.stdin)['max_train_epochs'])")

# 啟動訓練
cd /mnt/c/AI_LLM_projects/kohya_ss/sd-scripts

nohup conda run -n kohya_ss python train_network.py \
  --dataset_config /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/portorosso_background_dataset.toml \
  --pretrained_model_name_or_path /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors \
  --output_dir /mnt/data/ai_data/models/lora/luca/portorosso_background \
  --output_name portorosso_bg \
  --network_module networks.lora \
  --network_dim $BEST_DIM \
  --network_alpha $BEST_ALPHA \
  --learning_rate $BEST_LR \
  --text_encoder_lr $(echo "$BEST_LR * 0.8" | bc -l) \
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
  --logging_dir /mnt/data/ai_data/models/lora/luca/portorosso_background/logs \
  --log_with tensorboard \
  --seed 42 \
  --clip_skip 2 \
  --cache_latents \
  --cache_latents_to_disk \
  --max_data_loader_n_workers 8 \
  --persistent_data_loader_workers \
  > /mnt/data/ai_data/models/lora/luca/portorosso_background/training.log 2>&1 &

echo "Background LoRA 訓練已啟動，PID: $!"
```

**預期時間**：2-4 小時（取決於 epochs）

### 監控訓練
```bash
# 實時日誌
tail -f /mnt/data/ai_data/models/lora/luca/portorosso_background/training.log

# 查看 checkpoints
ls -lh /mnt/data/ai_data/models/lora/luca/portorosso_background/*.safetensors

# TensorBoard
tensorboard --logdir /mnt/data/ai_data/models/lora/luca/portorosso_background/logs --port 6006 --bind_all
```

---

## 🚀 階段 3：Pose LoRA 訓練（可選）

### 如果需要動作控制，可訓練 Pose LoRA：

```bash
# 1. 姿態估計（使用已有的 character instances）
python scripts/generic/pose/pose_estimation.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/segmented/characters \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/pose_annotated \
  --model rtmpose-m \
  --device cuda

# 2. 動作聚類
python scripts/generic/clustering/action_clustering.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/pose_annotated \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/action_clusters \
  --actions running,jumping,walking,standing

# 3. 訓練 Running Pose LoRA
# ... (similar to Background LoRA training)
```

---

## 🎬 階段 4：LoRA Composition 測試

### 前置條件
✅ Character LoRA 已訓練完成
✅ Background LoRA 已訓練完成
✅ (可選) Pose/Expression LoRA 已訓練完成

### 測試命令
```bash
python scripts/evaluation/test_lora_composition.py \
  --base-model /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors \
  --character-lora /mnt/data/ai_data/models/lora/luca/luca_character.safetensors \
  --background-lora /mnt/data/ai_data/models/lora/luca/portorosso_background/portorosso_bg.safetensors \
  --character-weight 1.0 \
  --background-weight 0.8 \
  --prompts \
    "luca, a young boy with brown hair and blue eyes, wearing blue striped shirt, in italian seaside town portorosso, colorful buildings, blue sky, pixar style, 3d animation" \
    "luca standing in portorosso town center, happy expression, sunny day, cinematic lighting" \
    "luca near the beach in portorosso, waves in background, warm sunset lighting" \
  --output-dir /mnt/data/ai_data/models/lora/luca/composition_tests \
  --num-samples 4 \
  --steps 30 \
  --guidance-scale 7.5 \
  --width 512 \
  --height 512 \
  --device cuda
```

### 如果有 Pose + Expression LoRA：
```bash
python scripts/evaluation/test_lora_composition.py \
  --base-model /path/to/sd1.5 \
  --character-lora luca_character.safetensors \
  --background-lora portorosso_background.safetensors \
  --pose-lora running_pose.safetensors \
  --expression-lora happy_expression.safetensors \
  --character-weight 1.0 \
  --background-weight 0.8 \
  --pose-weight 0.7 \
  --expression-weight 0.6 \
  --prompts \
    "luca running in portorosso, happy expression, dynamic motion, blue striped shirt, sunny day" \
  --output-dir composition_test_full \
  --num-samples 8
```

---

## 📊 預期結果

### Character LoRA Only（當前）
✅ 生成 Luca 角色準確
❌ 背景隨機
❌ 動作不可控
❌ 表情不可控

### Character + Background LoRA
✅ 生成 Luca 角色準確
✅ **Portorosso 場景識別**（義大利海邊小鎮）
❌ 動作不可控
❌ 表情不可控

### Character + Background + Pose + Expression LoRA
✅ 生成 Luca 角色準確
✅ Portorosso 場景識別
✅ **動作控制**（奔跑姿態）
✅ **表情控制**（開心笑容）

**最終效果**：「Luca 在 Portorosso 奔跑並露出開心笑容」的完整場景！

---

## ⏱️ 時間估算

| 階段 | 任務 | 預計時間 |
|------|-----|---------|
| ✅ **當前** | Character LoRA 優化（50 trials） | 1.5-2 天 |
| 🔜 **下一步** | Background inpainting + 聚類 | 1-2 小時 |
| 🔜 **下一步** | Background LoRA 訓練 | 2-4 小時 |
| ⚠️ **可選** | Pose LoRA 訓練 | 2-4 小時 |
| ⚠️ **可選** | Expression LoRA 訓練 | 2-4 小時 |
| 🎬 **測試** | LoRA Composition 測試 | 30 分鐘 |
| **總計** | 從現在到完整系統 | **2-3 天** |

---

## 🎯 建議優先級

### 最小可行方案（MVP）
1. ✅ Character LoRA（進行中）
2. 🔥 Background LoRA（高優先級）
3. 🎬 測試 Character + Background 組合

**優點**：最快看到效果（只需額外 3-6 小時）

### 完整方案
1. ✅ Character LoRA
2. 🔥 Background LoRA
3. 🔥 Pose LoRA（1 種動作，如 running）
4. 🔥 Expression LoRA（1 種表情，如 happy）
5. 🎬 測試完整組合

**優點**：完全控制，可生成複雜場景

---

## 💡 關鍵提示

### 1. **超參數遷移**
✅ Character LoRA 的最佳超參數可直接用於 Background/Pose/Expression LoRA
- 節省大量優化時間
- 參數已被證明有效

### 2. **數據分離純度**
⚠️ **關鍵**：確保不同 LoRA 的訓練數據純淨
- Character LoRA：透明背景或純色背景
- Background LoRA：完全移除角色（用 LaMa inpainting）
- Pose LoRA：單一動作，多視角
- Expression LoRA：面部清晰，表情明確

### 3. **LoRA 權重平衡**
推薦起始權重：
- Character: **1.0**（核心）
- Background: **0.7-0.9**（避免過度影響角色）
- Pose: **0.6-0.8**（輔助控制）
- Expression: **0.5-0.7**（精細調整）

### 4. **Prompt 工程**
✅ **良好結構**：
```
[Character trigger] [Pose trigger] [Expression trigger] in [Background trigger], [lighting], [style]
```

示例：
```
"luca, running pose, happy expression, in portorosso town, sunset lighting, pixar style, 3d animation"
```

---

## 📞 需要幫助？

**詳細技術文檔**: `MULTI_TYPE_LORA_SYSTEM.md`
**SDXL 升級指南**: `SD15_TO_SDXL_MIGRATION.md`
**工具腳本**:
- `scripts/generic/inpainting/background_inpainting.py`
- `scripts/evaluation/test_lora_composition.py`

---

## ✅ Checklist

**當前階段**：
- [x] Character LoRA 優化運行中
- [ ] 優化完成，最佳參數已提取

**下一階段**（Character LoRA 完成後）：
- [ ] 檢查背景 layers 數據
- [ ] 運行 Background inpainting
- [ ] 場景聚類
- [ ] 準備 Background 訓練數據
- [ ] 訓練 Background LoRA
- [ ] (可選) 訓練 Pose LoRA
- [ ] (可選) 訓練 Expression LoRA

**最終測試**：
- [ ] 測試 Character + Background 組合
- [ ] 測試完整 LoRA 組合（如有 Pose/Expression）
- [ ] 調整 LoRA 權重
- [ ] 生成最終展示圖片

---

**目標**：從單一 Character LoRA → 多類型 LoRA 生態系統 → 完全可控的場景生成！

**最後更新**: 2025-11-12
