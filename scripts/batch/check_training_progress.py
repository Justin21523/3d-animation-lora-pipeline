import os
import time
from datetime import datetime, timedelta

# 配置信息
dataset_size = 260
repeats = 4
steps_per_epoch = dataset_size * repeats  # 1040 steps/epoch
total_epochs = 10
total_steps = steps_per_epoch * total_epochs  # 10400 total steps
batch_size = 1
grad_accum = 2
effective_batch = batch_size * grad_accum

# 進程啟動時間（從ps輸出）
runtime_str = "21:21"  # HH:MM format from ps
hours, minutes = map(int, runtime_str.split(':'))
runtime_seconds = hours * 3600 + minutes * 60

print("=" * 60)
print("Alberto Sea Monster SDXL LoRA 訓練進度分析")
print("=" * 60)

print(f"\n配置資訊:")
print(f"  數據集大小: {dataset_size} 圖片")
print(f"  重複次數: {repeats}x")
print(f"  每個Epoch步數: {steps_per_epoch}")
print(f"  目標Epoch數: {total_epochs}")
print(f"  總步數: {total_steps}")
print(f"  有效批次大小: {effective_batch} (batch={batch_size}, grad_accum={grad_accum})")

print(f"\n當前狀態:")
print(f"  已運行時間: {hours}小時 {minutes}分鐘")
print(f"  GPU使用率: 97.6% (高效訓練中)")
print(f"  顯存使用: 12.4 GB / ~16 GB")

# 估算速度（基於SDXL典型速度）
# SDXL通常在RTX 4090/A100上約 1.5-2.5 秒/步
estimated_sec_per_step = 2.0  # 保守估計

current_steps = runtime_seconds / estimated_sec_per_step
current_epoch = current_steps / steps_per_epoch
progress_percent = (current_steps / total_steps) * 100

print(f"\n進度估算 (基於 {estimated_sec_per_step}秒/步):")
print(f"  已完成步數: ~{int(current_steps)}")
print(f"  當前Epoch: ~{current_epoch:.2f} / {total_epochs}")
print(f"  整體進度: {progress_percent:.1f}%")

# 計算預計完成時間
remaining_steps = total_steps - current_steps
remaining_seconds = remaining_steps * estimated_sec_per_step
remaining_hours = remaining_seconds / 3600

eta = datetime.now() + timedelta(seconds=remaining_seconds)

print(f"\n預計完成時間:")
print(f"  剩餘步數: ~{int(remaining_steps)}")
print(f"  剩餘時間: ~{remaining_hours:.1f} 小時")
print(f"  預計完成: {eta.strftime('%Y-%m-%d %H:%M')}")

# 檢查點保存時間
checkpoint_epochs = [2, 4, 6, 8, 10]
print(f"\n檢查點保存時間表 (每2個Epoch):")
for ep in checkpoint_epochs:
    ep_steps = ep * steps_per_epoch
    ep_seconds = ep_steps * estimated_sec_per_step
    ep_eta = datetime.now() - timedelta(seconds=runtime_seconds) + timedelta(seconds=ep_seconds)
    status = "✓ 已完成" if current_steps >= ep_steps else f"預計 {ep_eta.strftime('%H:%M')}"
    print(f"  Epoch {ep:2d}: {status}")

print("\n" + "=" * 60)

# 速度變化情況
print(f"\n速度分析:")
speeds = [
    (1.5, "最快 (ideal, 高端GPU)"),
    (2.0, "正常 (當前估算)"),
    (2.5, "較慢 (保守估算)"),
]

for speed, desc in speeds:
    total_time = (total_steps * speed) / 3600
    print(f"  {speed}秒/步 ({desc}): {total_time:.1f}小時完成")

