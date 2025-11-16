# SDD1 (3.6TB) 掛載完整指南

## 問題診斷

你的 3.6TB SDD1 硬碟沒有自動掛載到 `/mnt/data`。這是因為：

1. **WSL2 預設不會自動掛載外部 Linux 格式硬碟**（如 ext4）
2. 需要手動配置掛載流程
3. `/etc/fstab` 中的 UUID 配置需要硬碟先被 WSL 識別

## 解決方案

### 方法 A：一鍵自動掛載（推薦）

**步驟 1：在 Windows PowerShell (系統管理員) 執行**

```powershell
# 切換到專案目錄
cd C:\AI_LLM_projects\3d-animation-lora-pipeline

# 執行掛載輔助腳本
.\scripts\setup\mount_drive_windows.ps1
```

這個腳本會：
- 列出所有實體硬碟
- 自動識別 3.6TB 的硬碟
- 提供正確的 `wsl --mount` 命令

**步驟 2：根據腳本輸出執行掛載命令**

假設你的硬碟是 PHYSICALDRIVE1：

```powershell
# 掛載整個硬碟（推薦）
wsl --mount \\.\PHYSICALDRIVE1 --bare

# 或者掛載特定分割區（如果有多個分割區）
wsl --mount \\.\PHYSICALDRIVE1 --partition 1
```

**步驟 3：在 WSL 中執行自動掛載腳本**

```bash
# 診斷當前狀態
bash scripts/setup/diagnose_drives.sh

# 自動掛載到 /mnt/data
sudo bash scripts/setup/auto_mount_sdd1.sh
```

**步驟 4：設定開機自動掛載**

```bash
# 配置開機自動執行
sudo bash scripts/setup/setup_auto_mount.sh

# 重啟 WSL 測試
# 在 Windows PowerShell (系統管理員) 執行：
# wsl --shutdown
# 然後重新開啟 WSL
```

---

### 方法 B：手動逐步掛載

#### B1. 在 Windows 端掛載硬碟

**在 Windows PowerShell (系統管理員) 執行：**

```powershell
# 1. 列出所有實體硬碟
GET-PhysicalDisk | Select-Object DeviceID, FriendlyName, Size

# 2. 找到你的 3.6TB 硬碟（假設是 Disk 1）
# 3. 掛載到 WSL
wsl --mount \\.\PHYSICALDRIVE1 --bare

# 4. 驗證掛載成功
wsl -e lsblk
```

#### B2. 在 WSL 中掛載到檔案系統

```bash
# 1. 找到裝置名稱（通常是 /dev/sdb 或 /dev/sdc）
sudo lsblk

# 2. 確認 UUID
sudo blkid | grep ext4

# 3. 建立掛載點
sudo mkdir -p /mnt/data

# 4. 掛載
sudo mount -t ext4 /dev/sdX /mnt/data

# 5. 驗證
df -h /mnt/data
ls -lh /mnt/data/
```

#### B3. 設定自動掛載

```bash
# 編輯 /etc/wsl.conf
sudo nano /etc/wsl.conf
```

加入以下內容：

```ini
[boot]
command = /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/setup/auto_mount_sdd1.sh

[automount]
enabled = true
mountFsTab = true
```

儲存後重啟 WSL：

```powershell
# 在 Windows PowerShell (系統管理員)
wsl --shutdown
```

---

## 腳本說明

### 1. `diagnose_drives.sh`
**功能：** 診斷當前硬碟和掛載狀態

```bash
bash scripts/setup/diagnose_drives.sh
```

顯示：
- 所有區塊裝置
- UUID 資訊
- 掛載狀態
- /etc/fstab 配置
- 目錄內容

### 2. `auto_mount_sdd1.sh`
**功能：** 自動尋找並掛載 3.6TB SDD1 到 /mnt/data

```bash
sudo bash scripts/setup/auto_mount_sdd1.sh
```

執行：
- 按 UUID 搜尋裝置
- 按大小搜尋 ext4 分割區（>3.5TB）
- 驗證裝置大小
- 檢查是否已掛載
- 執行掛載操作
- 驗證 ai_data 目錄

### 3. `setup_auto_mount.sh`
**功能：** 配置開機自動掛載

```bash
sudo bash scripts/setup/setup_auto_mount.sh
```

配置：
- **systemd 服務**（如果可用）
  - 建立 `/etc/systemd/system/mount-sdd1.service`
  - 啟用服務
- **或 /etc/profile.d 腳本**（fallback）
  - 建立 `/etc/profile.d/mount-sdd1.sh`
  - 更新 `/etc/wsl.conf`

### 4. `mount_drive_windows.ps1`
**功能：** Windows 端輔助工具，自動識別硬碟

```powershell
.\scripts\setup\mount_drive_windows.ps1
```

顯示：
- 所有實體硬碟
- 候選 3.6TB 硬碟
- 分割區資訊
- 正確的 wsl --mount 命令

---

## 驗證掛載成功

### 檢查掛載狀態

```bash
# 1. 檢查掛載點
df -h /mnt/data

# 2. 檢查內容
ls -lh /mnt/data/

# 3. 檢查 ai_data
ls -lh /mnt/data/ai_data/
```

### 預期輸出

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdb        3.6T  XXG   X.XT  XX% /mnt/data

/mnt/data/ai_data/:
total XXG
drwxrwxr-x 2 user user 4.0K Nov 14 12:00 datasets
drwxrwxr-x 2 user user 4.0K Nov 14 12:00 models
drwxrwxr-x 2 user user 4.0K Nov 14 12:00 training_data
...
```

---

## 常見問題

### Q1: 執行 `wsl --mount` 時出現「找不到裝置」錯誤

**原因：** 硬碟在 Windows 中沒有初始化或被佔用

**解決：**
1. 在 Windows 中開啟「磁碟管理」
2. 檢查硬碟狀態
3. 如果顯示「離線」，右鍵選擇「連線」
4. 如果有掛載為 Windows 磁碟機（如 E:），先卸載

### Q2: `auto_mount_sdd1.sh` 找不到裝置

**原因：** 硬碟還沒通過 `wsl --mount` 掛載到 WSL

**解決：**
先在 Windows PowerShell 執行：
```powershell
wsl --mount \\.\PHYSICALDRIVE1 --bare
```

### Q3: 重啟 WSL 後掛載消失

**原因：** 沒有配置自動掛載

**解決：**
```bash
sudo bash scripts/setup/setup_auto_mount.sh
```

然後重啟 WSL：
```powershell
wsl --shutdown
```

### Q4: Permission denied 錯誤

**原因：** 沒有 root 權限

**解決：**
所有掛載操作都需要 `sudo`：
```bash
sudo bash scripts/setup/auto_mount_sdd1.sh
```

### Q5: UUID 不匹配

**原因：** 硬碟 UUID 改變或配置錯誤

**解決：**
```bash
# 1. 查看實際 UUID
sudo blkid | grep ext4

# 2. 更新 /etc/fstab
sudo nano /etc/fstab

# 3. 將 UUID 改為正確的值
UUID=<正確的UUID>  /mnt/data  ext4  defaults,nofail  0  2
```

---

## 自動化流程

### 完整自動化命令

**一次性執行所有步驟：**

```bash
# 在 WSL 中執行
cd /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline

# 1. 診斷
bash scripts/setup/diagnose_drives.sh

# 2. 掛載（如果裝置已在 WSL 中可見）
sudo bash scripts/setup/auto_mount_sdd1.sh

# 3. 設定自動掛載
sudo bash scripts/setup/setup_auto_mount.sh

# 4. 驗證
df -h /mnt/data
ls -lh /mnt/data/ai_data/
```

---

## 系統配置檔案參考

### /etc/fstab

```bash
# /etc/fstab - Justin WSL2 configuration
# Mount ext4 4TB data drive for AI datasets

UUID=03aa943d-f94a-4a9a-842f-0f980176747c  /mnt/data  ext4  defaults,nofail  0  2
```

### /etc/wsl.conf

```ini
[boot]
command = /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/setup/auto_mount_sdd1.sh

[automount]
enabled = true
mountFsTab = true
```

### systemd 服務（如果使用）

**檔案位置：** `/etc/systemd/system/mount-sdd1.service`

```ini
[Unit]
Description=Auto-mount SDD1 (3.6TB) drive to /mnt/data
After=local-fs.target
Before=basic.target

[Service]
Type=oneshot
ExecStart=/bin/bash /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/setup/auto_mount_sdd1.sh
RemainAfterExit=yes
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**管理命令：**

```bash
# 查看狀態
sudo systemctl status mount-sdd1

# 查看日誌
sudo journalctl -u mount-sdd1 -f

# 重新啟動
sudo systemctl restart mount-sdd1

# 停用自動掛載
sudo systemctl disable mount-sdd1
```

---

## 總結

### 推薦工作流程

1. **Windows PowerShell (系統管理員)：**
   ```powershell
   .\scripts\setup\mount_drive_windows.ps1
   wsl --mount \\.\PHYSICALDRIVE1 --bare
   ```

2. **WSL：**
   ```bash
   sudo bash scripts/setup/auto_mount_sdd1.sh
   sudo bash scripts/setup/setup_auto_mount.sh
   ```

3. **重啟測試：**
   ```powershell
   wsl --shutdown
   # 重新開啟 WSL，檢查 df -h /mnt/data
   ```

### 日常使用

掛載配置完成後，每次 WSL 啟動時會自動：
1. 執行 `/etc/wsl.conf` 中的 boot command
2. 或執行 systemd 服務 `mount-sdd1.service`
3. 自動掛載 SDD1 到 `/mnt/data`

你只需要確保在 Windows 端硬碟保持連接狀態即可。

---

## 相關檔案

- `scripts/setup/diagnose_drives.sh` - 診斷工具
- `scripts/setup/auto_mount_sdd1.sh` - 自動掛載腳本
- `scripts/setup/setup_auto_mount.sh` - 自動掛載配置
- `scripts/setup/mount_drive_windows.ps1` - Windows 輔助工具
- `docs/setup/MOUNT_SDD1_GUIDE.md` - 本指南

## 支援

如遇問題，請執行診斷並提供輸出：

```bash
bash scripts/setup/diagnose_drives.sh > mount_diagnosis.txt 2>&1
```
