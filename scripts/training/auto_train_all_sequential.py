#!/usr/bin/env python3
"""
完整自動化LoRA訓練腳本 - Character 2-7
每個角色訓練完成後自動測試，然後接續下一個
"""

import subprocess
import time
import os
import sys
from datetime import datetime

CHARACTERS = [2, 3, 4, 5, 6, 7]
BASE_MODEL = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/anything-v5-PrtRE.safetensors"
EVAL_DIR = "/mnt/data/ai_data/lora_evaluation"
PROJECT_ROOT = "/mnt/c/AI_LLM_projects/anime-lora-pipeline"

def log(message):
    """打印帶時間戳的日誌"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def run_command(cmd, log_file=None):
    """執行命令並返回結果"""
    log(f"執行命令: {' '.join(cmd[:3])}...")

    if log_file:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            process.wait()
            return process.returncode
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode

def train_character(char_num):
    """訓練指定角色"""
    log(f"=" * 70)
    log(f"📌 Character{char_num} 開始訓練")
    log(f"=" * 70)

    config_file = f"{PROJECT_ROOT}/configs/character_loras/character{char_num}_config.toml"
    log_file = f"/tmp/character{char_num}_training.log"

    cmd = [
        "conda", "run", "-n", "ai_env",
        "python", f"{PROJECT_ROOT}/sd-scripts/train_network.py",
        "--config_file", config_file
    ]

    start_time = time.time()
    returncode = run_command(cmd, log_file)
    elapsed = time.time() - start_time

    if returncode == 0:
        log(f"✅ Character{char_num} 訓練完成（耗時: {elapsed/3600:.1f}小時）")
        return True
    else:
        log(f"❌ Character{char_num} 訓練失敗！")
        log(f"   錯誤日誌: {log_file}")
        # 顯示最後30行日誌
        with open(log_file, 'r') as f:
            lines = f.readlines()
            log("   最後30行日誌:")
            for line in lines[-30:]:
                print(f"   {line.rstrip()}")
        return False

def test_character(char_num):
    """測試指定角色的LoRA"""
    log(f"🧪 Character{char_num} 開始測試")

    lora_path = f"/mnt/data/ai_data/models/lora/yokai_characters/character{char_num}/yokai_character{char_num}_lora.safetensors"

    if not os.path.exists(lora_path):
        log(f"⚠️  找不到LoRA檔案: {lora_path}")
        return False

    output_dir = f"{EVAL_DIR}/character{char_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_file = f"/tmp/character{char_num}_eval.log"

    cmd = [
        "conda", "run", "-n", "ai_env",
        "python", f"{PROJECT_ROOT}/scripts/evaluate_lora.py",
        lora_path,
        "--base_model", BASE_MODEL,
        "--output_dir", output_dir,
        "--num_samples", "8",
        "--seed", "42"
    ]

    returncode = run_command(cmd, log_file)

    if returncode == 0:
        log(f"✅ Character{char_num} 測試完成")
        log(f"   測試圖片: {output_dir}")
        return True
    else:
        log(f"⚠️  Character{char_num} 測試失敗（不影響後續訓練）")
        return False

def main():
    log("=" * 70)
    log("🚀 自動化LoRA訓練流程啟動")
    log("=" * 70)
    log(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"待訓練角色: {CHARACTERS}")
    log("")

    # Character2已經在訓練，等待完成
    if 2 in CHARACTERS:
        log("⏳ Character2 已在訓練中，等待完成...")
        log("   檢查訓練進程...")

        # 等待Character2訓練完成
        while True:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )
            if "character2_config.toml" in result.stdout:
                log("   Character2 仍在訓練中，60秒後再次檢查...")
                time.sleep(60)
            else:
                log("✅ Character2 訓練已完成")
                break

        # 測試Character2
        test_character(2)

        # 等待GPU冷卻
        log("⏳ 等待GPU冷卻 (15秒)...")
        time.sleep(15)

        # 移除Character2，繼續其他角色
        CHARACTERS.remove(2)

    # 訓練剩餘角色
    for char_num in CHARACTERS:
        success = train_character(char_num)

        if not success:
            log(f"❌ Character{char_num} 訓練失敗，停止自動化流程")
            sys.exit(1)

        # 等待GPU冷卻
        log("⏳ 等待GPU冷卻 (15秒)...")
        time.sleep(15)

        # 測試
        test_character(char_num)

        # 等待下一個角色
        if char_num != CHARACTERS[-1]:
            log("⏳ 等待GPU冷卻 (15秒)...")
            time.sleep(15)

        log("")

    log("=" * 70)
    log("🎉 所有角色訓練和測試完成！")
    log("=" * 70)
    log(f"結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("")
    log("生成的模型：")
    for i in range(2, 8):
        lora_path = f"/mnt/data/ai_data/models/lora/yokai_characters/character{i}/yokai_character{i}_lora.safetensors"
        if os.path.exists(lora_path):
            size = os.path.getsize(lora_path) / (1024 * 1024)
            log(f"  ✅ Character{i}: {lora_path} ({size:.1f}MB)")
        else:
            log(f"  ❌ Character{i}: 未找到")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n⚠️  用戶中斷訓練")
        sys.exit(1)
    except Exception as e:
        log(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
