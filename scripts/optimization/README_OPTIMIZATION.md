# Optuna超參數優化系統使用指南

## 📋 系統概述

這是一個基於 Optuna 的自動化超參數優化系統，專門為 Pixar 風格 3D 動畫 LoRA 訓練設計。

### 核心功能

1. **自動超參數搜索** - 使用 TPE (Tree-structured Parzen Estimator) 算法
2. **多目標優化** - 同時優化 brightness 和 contrast
3. **增強評估指標** - LPIPS、CLIP consistency、Pixar style score
4. **結果分析** - 自動生成報告、視覺化圖表
5. **最佳配置導出** - 自動生成訓練配置文件

## 🎯 優化目標

針對 Pixar 風格的特定目標：

- **Brightness（亮度）**: 目標 0.50，範圍 0.4-0.6
- **Contrast（對比度）**: 目標 0.20，範圍 0.15-0.25（Pixar 低對比特徵）
- **Consistency（一致性）**: 降低 brightness 和 contrast 的標準差

## 🔧 超參數搜索空間

### 學習率（最關鍵）
- `learning_rate`: 5e-5 到 2e-4（對數均勻分布）
- `text_encoder_lr`: 3e-5 到 1e-4（對數均勻分布）

### 網絡架構
- `network_dim`: 64, 96, 128, 192, 256
- `network_alpha`: 32, 48, 64, 96, 128

### 優化器
- `optimizer_type`: AdamW, AdamW8bit, Lion, Adafactor

### 學習率調度器
- `lr_scheduler`: cosine, cosine_with_restarts, polynomial

### 訓練設定
- `gradient_accumulation_steps`: 1, 2, 4
- `max_train_epochs`: 8, 10, 12, 15
- `lr_warmup_steps`: 50-200（步進 50）

## 📦 系統組件

```
scripts/optimization/
├── optuna_hyperparameter_search.py    # 主優化腳本
├── enhanced_metrics.py                 # 增強評估指標
├── analyze_optuna_results.py          # 結果分析工具
└── README_OPTIMIZATION.md             # 本文件
```

## 🚀 快速開始

### 步驟 1: 確認 V6 訓練完成

```bash
# 檢查訓練狀態
ls -lh /mnt/data/ai_data/models/lora/luca/iterative_overnight_v6/*.safetensors

# 應該看到 6 個 checkpoints (epoch 2, 4, 6, 8, 10, 12)
```

### 步驟 2: 運行超參數優化（30 trials）

```bash
# 切換到項目目錄
cd /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline

# 啟動優化（使用 nohup 避免中斷）
nohup /home/b0979/.conda/envs/kohya_ss/bin/python \
  scripts/optimization/optuna_hyperparameter_search.py \
  --dataset-config configs/training/luca_human_dataset.toml \
  --output-dir /mnt/data/ai_data/models/lora/luca/optimization_results \
  --study-name luca_pixar_optimization \
  --n-trials 30 \
  --device cuda \
  > /tmp/optuna_optimization.log 2>&1 &

echo "優化已啟動，PID: $!"
```

### 步驟 3: 監控進度

```bash
# 實時查看日誌
tail -f /tmp/optuna_optimization.log

# 檢查已完成的 trials
ls /mnt/data/ai_data/models/lora/luca/optimization_results/trial_*/

# 查看 Optuna 資料庫
sqlite3 /mnt/data/ai_data/models/lora/luca/optimization_results/optuna_study.db \
  "SELECT number, state, value FROM trials ORDER BY value LIMIT 10;"
```

### 步驟 4: 分析結果

```bash
# 等優化完成後，運行結果分析
/home/b0979/.conda/envs/kohya_ss/bin/python \
  scripts/optimization/analyze_optuna_results.py \
  --results-dir /mnt/data/ai_data/models/lora/luca/optimization_results \
  --top-n 10
```

### 步驟 5: 查看結果

```bash
# 檢視最佳參數
cat /mnt/data/ai_data/models/lora/luca/optimization_results/results/best_parameters.json

# 查看詳細報告
cat /mnt/data/ai_data/models/lora/luca/optimization_results/results/analysis/OPTIMIZATION_REPORT.md

# 查看視覺化圖表
ls /mnt/data/ai_data/models/lora/luca/optimization_results/results/analysis/*.png
```

## 📊 輸出結構

```
optimization_results/
├── optuna_study.db                          # Optuna 資料庫
├── trial_0001/                              # Trial 1
│   ├── params.json                          # 超參數
│   ├── lora_trial_0001.safetensors         # 訓練的 checkpoint
│   ├── training.log                         # 訓練日誌
│   └── evaluation/                          # 評估結果
│       ├── metrics.json
│       ├── EVALUATION_SUMMARY.txt
│       └── sample_*.png                     # 測試圖片
├── trial_0002/
│   └── ...
└── results/
    ├── best_parameters.json                 # 最佳超參數
    ├── all_trials.json                      # 所有 trials 數據
    ├── optimization_history.png             # 優化歷史圖
    ├── param_importances.png                # 參數重要性圖
    └── analysis/                            # 詳細分析
        ├── summary_statistics.json
        ├── top_10_trials.csv
        ├── score_evolution.png
        ├── metrics_comparison.png
        ├── parameter_correlation.png
        ├── best_training_config.txt         # 可直接使用的配置
        └── OPTIMIZATION_REPORT.md           # 完整報告
```

## 🎯 評估指標說明

### Combined Score（組合分數）

```python
brightness_error = abs(mean_brightness - 0.50)
contrast_error = abs(mean_contrast - 0.20)

brightness_score = brightness_error + 0.5 * std_brightness
contrast_score = contrast_error + 0.5 * std_contrast

combined_score = brightness_score + contrast_score  # 越低越好
```

### Pixar Style Score（Pixar 風格分數）

加權組合：
- Brightness in range (0.4-0.6): 30%
- Contrast in range (0.15-0.25): 40%（最重要）
- Saturation in range (0.3-0.5): 20%
- Consistency bonus: 10%

## ⚙️ 進階設定

### 調整 Trial 數量

```bash
# 快速測試（10 trials）
--n-trials 10

# 標準搜索（30 trials）
--n-trials 30

# 深度搜索（50 trials）
--n-trials 50
```

### 使用不同的 Sampler

修改 `optuna_hyperparameter_search.py`：

```python
# TPE (預設) - 適合大多數情況
sampler = optuna.samplers.TPESampler(seed=42)

# Random - 基準比較
sampler = optuna.samplers.RandomSampler(seed=42)

# CMA-ES - 連續參數優化
sampler = optuna.samplers.CmaEsSampler(seed=42)
```

### 多目標優化（進階）

如果想同時優化多個目標：

```python
# 修改 create_study 為多目標
study = optuna.create_study(
    directions=["minimize", "minimize"],  # [brightness_score, contrast_score]
    sampler=optuna.samplers.NSGAIISampler(seed=42),
)

# 修改 objective 返回多個值
return [metrics["brightness_score"], metrics["contrast_score"]]
```

## 📈 預期結果

基於 V6 的 baseline：
- V6 Epoch 2: Brightness 0.444, Contrast 0.190
- V6 Epoch 4: Brightness 0.425, Contrast 0.199

**優化目標:**
- Brightness: 0.45-0.55（更接近 0.50）
- Contrast: 0.18-0.22（更穩定在 0.20 附近）
- Consistency: 降低 std（brightness_std < 0.05, contrast_std < 0.02）

## 🔍 常見問題

### Q1: 優化需要多久？

每個 trial 包含完整訓練（8-15 epochs）+ 評估：
- 單個 trial: ~30-60 分鐘（取決於 epochs）
- 30 trials: ~15-30 小時

**建議:** 使用 `nohup` 和 `tmux` 進行長時間運行

### Q2: 如何恢復中斷的優化？

Optuna 自動保存進度到 SQLite：

```bash
# 使用相同參數重新運行即可自動恢復
--study-name luca_pixar_optimization  # 相同名稱
--storage sqlite:///path/to/optuna_study.db  # 相同資料庫
```

### Q3: 如何提前停止不良的 trials？

實現 pruning callback：

```python
# 在 objective 函數中添加
if epoch == 2:  # 檢查早期結果
    trial.report(intermediate_score, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

### Q4: 記憶體不足怎麼辦？

1. 減少評估樣本數量: `--num-samples 4`（預設 8）
2. 減少 batch size（修改 dataset config）
3. 啟用 gradient checkpointing（已啟用）

## 🎓 最佳實踐

### 1. 分階段優化

```bash
# 階段 1: 粗搜索（10 trials, 8 epochs）
--n-trials 10 --max-epochs 8

# 階段 2: 精細搜索（20 trials, 12 epochs）
--n-trials 20 --max-epochs 12

# 階段 3: 驗證（5 trials, 15 epochs）
--n-trials 5 --max-epochs 15
```

### 2. 參數空間調整

如果初步結果顯示某些參數表現好：

```python
# 縮小搜索範圍
"learning_rate": trial.suggest_float("learning_rate", 8e-5, 1.5e-4, log=True),
"network_dim": trial.suggest_categorical("network_dim", [128, 192, 256]),
```

### 3. 使用 V6 結果作為 baseline

保存 V6 最佳 checkpoint 作為比較基準：

```bash
cp /mnt/data/ai_data/models/lora/luca/iterative_overnight_v6/luca_v6-000004.safetensors \
   /mnt/data/ai_data/models/lora/luca/baseline_v6_epoch4.safetensors
```

## 📚 相關文檔

- Optuna 官方文檔: https://optuna.readthedocs.io/
- TPE Algorithm: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
- ITERATIVE_OPTIMIZATION_GUIDE.md: 詳細優化策略

## 🆘 故障排除

### Error: "CUDA out of memory"

```bash
# 解決方案：減少 workers 和 batch size
--max_data_loader_n_workers 4  # 預設 8
--train_batch_size 1            # 如果需要
```

### Error: "Checkpoint not found"

檢查訓練日誌：

```bash
tail -100 /path/to/trial_XXXX/training.log
```

### Error: "LPIPS/CLIP not available"

重新安裝依賴：

```bash
conda run -n kohya_ss pip install lpips git+https://github.com/openai/CLIP.git
```

## 🎉 完成後的下一步

1. **選擇最佳配置**
   ```bash
   cat results/analysis/best_training_config.txt
   ```

2. **進行完整訓練**
   ```bash
   # 使用最佳參數訓練完整版本（如 20-30 epochs）
   ```

3. **A/B 測試**
   ```bash
   # 比較優化前後的 LoRA 質量
   ```

4. **生產部署**
   ```bash
   # 將最佳 LoRA 用於實際生成
   ```

---

**祝優化順利！** 🚀

有任何問題，請參考 `/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/docs/guides/ITERATIVE_OPTIMIZATION_GUIDE.md`
