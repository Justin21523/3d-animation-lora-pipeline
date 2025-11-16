# LoRA訓練優化方案 - 針對臉部特徵一致性

## 核心問題：臉部特徵不一致
用戶反饋：「臉型和特徵有差異，看起來很明顯」

## 根本原因分析：
1. **Learning Rate過高** → 特徵空間漂移，臉部細節不穩定
2. **Epochs過多** → 過度訓練破壞了原始特徵的一致性
3. **Text Encoder LR太低** → 身份概念學習不足，UNet主導導致變形
4. **Network Dim盲目增長** → 過大容量學到噪音而非穩定特徵

## 改進策略（優先級排序）：

### 1. 大幅降低Learning Rate（最關鍵）
```python
'learning_rate': 8.0e-5,       # 從1.0e-4進一步降低20%
'text_encoder_lr': 6.0e-5,     # 從5.0e-5提升20% - 加強身份學習
```
**理由**：較低的UNet LR防止特徵漂移，較高的Text Encoder LR加強身份概念

### 2. 減少Epoch數到最保守值
```python
'max_train_epochs': 10,        # 從12降到10
```

### 3. 限制Network Capacity增長
```python
# 不允許network_dim超過96（已經很大）
# 當consistency低時，不增加dim，而是降低LR
```

### 4. 改進Early Stopping機制
```python
# 從"停止訓練"改為"警告並回退到最佳參數"
if self.consecutive_degradations >= 2:
    print("⚠️  質量下降 - 恢復最佳參數並降低學習率")
    # 回到最佳iteration的參數
    # 並且進一步降低20% LR
    # 繼續訓練（不停止）
```

### 5. 針對Consistency分數的特殊處理
```python
# 當consistency < 0.75時（臉部不一致）：
if consistency < 0.75:
    # 不增加network_dim！
    # 反而降低learning rate
    new_params['learning_rate'] *= 0.85
    new_params['text_encoder_lr'] *= 1.1  # 相對提升
    reasons.append("降低LR以改善臉部特徵一致性")
```

### 6. 添加Text Encoder相對權重保護
```python
# 確保text_encoder_lr至少是unet_lr的60%
te_ratio = new_params['text_encoder_lr'] / new_params['learning_rate']
if te_ratio < 0.6:
    new_params['text_encoder_lr'] = new_params['learning_rate'] * 0.6
```

## 建議的最佳起始參數（針對3D角色臉部一致性）
```python
BEST_PARAMS_3D_CONSISTENCY = {
    'learning_rate': 8.0e-5,       # 降低
    'text_encoder_lr': 6.0e-5,     # 提高（相對）
    'network_dim': 96,             # 固定不增長
    'network_alpha': 48,
    'max_train_epochs': 10,        # 降低
    'batch_size': 8,
    'lr_scheduler': 'cosine',
    'optimizer_type': 'AdamW',
}
```

## 期望效果：
✓ 臉部特徵更穩定
✓ 減少distortion現象
✓ 保持角色身份的一致性
✓ 訓練不會被強制停止，而是自動調整參數繼續
