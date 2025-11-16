# Checkpoint Evaluation Upgrade - 中間 Checkpoint 評估功能

## 修改日期
2025-11-12

## 修改內容

### 1. 優化腳本修改
**文件**: `/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/optimization/optuna_hyperparameter_search.py`

**修改位置**: 第 160 行

**添加參數**:
```python
"--save_every_n_epochs", "2",  # Save checkpoint every 2 epochs for intermediate evaluation
```

### 2. 修改效果

#### 修改前（當前運行的 optimization_overnight）
- ❌ 每個 trial 只保存**一個最終 checkpoint**
- ❌ 必須等待所有 epochs 完成才能看到結果
- ❌ 無法觀察訓練過程中的品質變化
- 示例：15 epochs trial → 只有 `lora_trial_1.safetensors`

#### 修改後（下次優化開始時生效）
- ✅ 每 2 個 epochs 保存一次 checkpoint
- ✅ 實時評估系統立即檢測和評估新 checkpoint
- ✅ 可以觀察訓練曲線和品質演變
- 示例：15 epochs trial → 7-8 個 checkpoints
  - `lora_trial_1-000002.safetensors` (epoch 2)
  - `lora_trial_1-000004.safetensors` (epoch 4)
  - `lora_trial_1-000006.safetensors` (epoch 6)
  - `lora_trial_1-000008.safetensors` (epoch 8)
  - `lora_trial_1-000010.safetensors` (epoch 10)
  - `lora_trial_1-000012.safetensors` (epoch 12)
  - `lora_trial_1-000014.safetensors` (epoch 14)
  - `lora_trial_1.safetensors` (final, epoch 15)

### 3. 各 Epoch 配置的 Checkpoint 數量

| 配置 Epochs | Checkpoint 數量 | 生成時間點 |
|------------|----------------|-----------|
| 8 epochs   | 4-5 個         | epoch 2, 4, 6, 8 (final) |
| 10 epochs  | 5-6 個         | epoch 2, 4, 6, 8, 10 (final) |
| 12 epochs  | 6-7 個         | epoch 2, 4, 6, 8, 10, 12 (final) |
| 15 epochs  | 7-8 個         | epoch 2, 4, 6, 8, 10, 12, 14, 15 (final) |

### 4. 磁碟空間需求

**單個 checkpoint 大小**: ~73-150 MB (視 network_dim 而定)

**預估總空間**:
- 50 trials × 平均 6 checkpoints × 100 MB = **約 30 GB**
- 加上評估圖片 (50 trials × 6 cp × 8 samples × ~2MB) = **約 5 GB**
- **總計**: 約 35 GB

當前 `/mnt/data` 磁碟空間充足，無需擔心。

### 5. 實時評估系統狀態

✅ **已完全就緒**，無需任何修改！

- **自動評估管理器**: 運行中 (PID 83980)
- **評估器邏輯**: 使用 `glob("*.safetensors")` 自動檢測所有 checkpoint
- **狀態追蹤**: 通過 `evaluation_state.json` 避免重複評估
- **評估頻率**: 每 30 秒掃描一次新 checkpoint

**工作流程**:
1. 訓練到 epoch 2 → 保存 `lora_trial_X-000002.safetensors`
2. 評估器 30 秒內檢測到 → 立即開始評估
3. 生成 8 張測試圖片 + metrics.json
4. 保存到 `trial_X/realtime_evaluations/lora_trial_X-000002/`
5. 更新 `evaluation_summary.json`
6. 繼續監控，等待下一個 checkpoint

### 6. 何時生效

**重要**: 修改已完成，但**不影響當前運行的優化**

- ❌ 當前的 50 trials 優化（PID 80250）: 仍使用舊配置（只保存最終 checkpoint）
- ✅ 下次啟動新的優化時: 自動使用新配置（每 2 epochs 保存）

**建議時機**:
等當前 50 trials 完成後（預計需要 2-3 天），再啟動新的優化 run 來測試中間 checkpoint 評估功能。

### 7. 優點與成本

#### 優點
✅ **觀察訓練動態**: 看到品質如何從 epoch 2 → 4 → 6... → final 演變
✅ **提早發現問題**: 如果某配置在前幾個 epoch 就表現不佳，可以記錄
✅ **更精細的分析**: 可以繪製 "epoch vs quality" 曲線
✅ **中間 checkpoint 備份**: 如果最終 checkpoint 過擬合，可以回退到 epoch 10/12

#### 成本
⚠️ **磁碟空間**: 增加 ~30 GB（可接受）
⚠️ **評估時間**: 每個 trial 需要評估 4-8 次而非 1 次（GPU 會更忙碌）
⚠️ **I/O 負載**: 更頻繁的檔案寫入和讀取

**總體評估**: 成本完全可接受，收益巨大！

### 8. 監控指令

#### 查看當前優化進度
```bash
tail -30 /mnt/data/ai_data/models/lora/luca/optimization_overnight/optimization.log
```

#### 查看評估管理器狀態
```bash
tail -30 /mnt/data/ai_data/models/lora/luca/optimization_overnight/evaluator_manager.log
```

#### 查看某個 trial 的所有 checkpoints
```bash
ls -lh /mnt/data/ai_data/models/lora/luca/optimization_overnight/trial_0001/*.safetensors
```

#### 查看某個 trial 的評估結果
```bash
ls -d /mnt/data/ai_data/models/lora/luca/optimization_overnight/trial_0001/realtime_evaluations/*/
```

#### 實時監控（所有 trials）
```bash
bash /mnt/data/ai_data/models/lora/luca/optimization_overnight/watch_realtime_evaluation.sh
```

### 9. 下次啟動新優化的指令

當前優化完成後，啟動新的優化 run（會自動使用新配置）:

```bash
cd /mnt/data/ai_data/models/lora/luca

# 創建新的優化目錄
mkdir -p optimization_with_intermediate_checkpoints

# 啟動優化（會自動使用修改後的腳本）
conda run -n kohya_ss python \
  /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/optimization/optuna_hyperparameter_search.py \
  --dataset-config /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/luca_human_dataset.toml \
  --base-model /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors \
  --output-dir /mnt/data/ai_data/models/lora/luca/optimization_with_intermediate_checkpoints \
  --study-name luca_intermediate_checkpoint_study \
  --n-trials 20 \
  --device cuda

# 同時啟動評估管理器（自動管理所有 trials）
python3 /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/optimization/auto_evaluator_manager.py \
  --optimization-dir /mnt/data/ai_data/models/lora/luca/optimization_with_intermediate_checkpoints \
  --device cuda \
  --num-samples 8 \
  --check-interval 30 \
  --evaluator-check-interval 30
```

### 10. 預期結果示例

**Trial 5 (假設 12 epochs, gradient_accumulation_steps=2)**

```
trial_0005/
├── params.json
├── training.log
├── logs/
│   └── [tensorboard logs]
├── lora_trial_5-000002.safetensors  (73 MB, epoch 2)
├── lora_trial_5-000004.safetensors  (73 MB, epoch 4)
├── lora_trial_5-000006.safetensors  (73 MB, epoch 6)
├── lora_trial_5-000008.safetensors  (73 MB, epoch 8)
├── lora_trial_5-000010.safetensors  (73 MB, epoch 10)
├── lora_trial_5.safetensors         (73 MB, final epoch 12)
├── realtime_evaluation.log
└── realtime_evaluations/
    ├── evaluation_state.json
    ├── evaluation_summary.json
    ├── lora_trial_5-000002/
    │   ├── metrics.json
    │   ├── sample_0.png
    │   ├── sample_1.png
    │   └── ... (8 samples)
    ├── lora_trial_5-000004/
    │   └── ... (8 samples)
    ├── lora_trial_5-000006/
    │   └── ... (8 samples)
    ├── lora_trial_5-000008/
    │   └── ... (8 samples)
    ├── lora_trial_5-000010/
    │   └── ... (8 samples)
    └── lora_trial_5/
        └── ... (8 samples, final)
```

**evaluation_summary.json 示例**:
```json
{
  "trial_name": "trial_0005",
  "total_checkpoints_evaluated": 6,
  "checkpoints": {
    "lora_trial_5-000002": {
      "epoch": 2,
      "mean_brightness": 0.512,
      "mean_contrast": 45.3,
      "mean_saturation": 0.623,
      "evaluation_time": "2025-11-15T10:23:45"
    },
    "lora_trial_5-000004": {
      "epoch": 4,
      "mean_brightness": 0.528,
      "mean_contrast": 48.1,
      "mean_saturation": 0.645,
      "evaluation_time": "2025-11-15T10:45:12"
    },
    // ... epochs 6, 8, 10, 12
  },
  "last_update": "2025-11-15T12:34:56"
}
```

從這個 summary 可以看到**品質隨 epoch 演變的趨勢**！

### 11. 技術細節

#### Kohya ss-scripts 的 checkpoint 命名規則
- `--save_every_n_epochs N`: 每 N 個 epochs 保存時，命名為 `{output_name}-{epoch:06d}.safetensors`
- 最終 checkpoint: `{output_name}.safetensors`（沒有 epoch 後綴）

**示例**:
```
--output_name lora_trial_5
--max_train_epochs 12
--save_every_n_epochs 2

生成:
lora_trial_5-000002.safetensors
lora_trial_5-000004.safetensors
lora_trial_5-000006.safetensors
lora_trial_5-000008.safetensors
lora_trial_5-000010.safetensors
lora_trial_5.safetensors  (final, 沒有數字後綴)
```

#### 評估器掃描邏輯
```python
# realtime_checkpoint_evaluator.py:75-82
def scan_for_new_checkpoints(self) -> list[Path]:
    """Scan trial directory for new .safetensors checkpoints"""
    all_checkpoints = sorted(self.trial_dir.glob("*.safetensors"))
    new_checkpoints = [
        cp for cp in all_checkpoints
        if cp.name not in self.evaluated_checkpoints
    ]
    return new_checkpoints
```

- 使用 `glob("*.safetensors")` 掃描所有 .safetensors 文件
- 通過文件名（不是內容）判斷是否已評估
- 自動處理任何命名格式的 checkpoint

**結論**: 無需修改評估器，完全相容中間 checkpoint！

### 12. FAQ

**Q: 當前的 50 trials 會受影響嗎？**
A: 不會。當前運行的進程使用的是記憶體中已載入的舊代碼，不會受到檔案修改影響。

**Q: 我可以手動測試修改後的腳本嗎？**
A: 可以！等當前優化完成後，啟動一個小規模測試（例如 3 trials）來驗證功能。

**Q: 如果中間 checkpoint 佔用太多空間怎麼辦？**
A: 可以在優化完成後，刪除非最佳 trial 的中間 checkpoints，只保留最終版本和評估結果。

**Q: 評估器會不會因為太多 checkpoint 而變慢？**
A: 不會。評估器使用狀態追蹤，已評估的 checkpoint 不會重複評估。每個新 checkpoint 只評估一次。

**Q: 可以改成每 3 個 epochs 保存嗎？**
A: 可以！修改腳本中的 `"--save_every_n_epochs", "2"` → `"3"` 即可。

**Q: 最終評估時用哪個 checkpoint？**
A: Optuna 仍然使用**最終 checkpoint**（沒有數字後綴）進行最終評分，中間 checkpoint 只用於觀察訓練過程。

---

## 總結

✅ **修改已完成**: 優化腳本已添加 `--save_every_n_epochs 2` 參數
✅ **評估系統就緒**: 實時評估管理器完全支援多 checkpoint
✅ **下次生效**: 等當前 50 trials 完成後，新優化會自動使用新配置
✅ **成本可接受**: 約 35 GB 磁碟空間，GPU 評估時間略增
✅ **收益巨大**: 可觀察訓練動態，提早發現問題，更精細分析

**建議**: 等當前優化完成後，啟動一個 10-20 trials 的測試 run，驗證中間 checkpoint 評估功能！
