# System Protection & OOM Prevention

Complete system protection suite to prevent memory exhaustion and system crashes during AI training.

## 🎯 Purpose

Prevents Ubuntu/Linux system crashes caused by:
- **RAM exhaustion** (Out of Memory - OOM)
- **GPU memory overflow** (CUDA OOM)
- **Desktop environment crashes** from memory pressure
- **Training process termination** by OOM Killer

## 📋 Available Scripts

### 1. **set_oom_priorities.sh**
Sets OOM killer priorities to protect critical processes.

```bash
bash scripts/system_protection/set_oom_priorities.sh
```

**What it does:**
- Protects training processes from being killed (-300 score)
- Protects desktop environment (-500 score)
- Marks browsers and non-critical apps as killable first (500+ score)

**Usage:** Run once when starting training session.

---

### 2. **memory_watchdog.sh**
Real-time RAM and swap monitoring with desktop notifications.

```bash
# Start in background
nohup scripts/system_protection/memory_watchdog.sh > /dev/null 2>&1 &
```

**Features:**
- Checks memory every 10 seconds
- Desktop notifications at 80% (warning) and 90% (critical)
- Logs top memory consumers
- Alerts on high swap usage (>80%)

**Logs:** `/tmp/memory_watchdog.log`

---

### 3. **gpu_watchdog.sh**
GPU memory and temperature monitoring.

```bash
# Start in background
nohup scripts/system_protection/gpu_watchdog.sh > /dev/null 2>&1 &
```

**Features:**
- Monitors GPU memory usage (alerts at 95%)
- Temperature warnings (85°C+)
- Per-GPU monitoring for multi-GPU setups

**Logs:** `/tmp/gpu_watchdog.log`

---

### 4. **safe_train_wrapper.sh**
Wrapper script that adds protection to any training command.

```bash
scripts/system_protection/safe_train_wrapper.sh \
  conda run -n kohya_ss accelerate launch sdxl_train_network.py --config_file ...
```

**Features:**
- Pre-flight memory checks
- Automatic watchdog startup
- OOM priority configuration
- Captures training logs
- Post-mortem OOM detection and reporting

**Best Practice:** Always use this wrapper for long training runs.

---

### 5. **unified_monitor_dashboard.sh**
Comprehensive real-time monitoring dashboard.

```bash
scripts/system_protection/unified_monitor_dashboard.sh
```

**Shows:**
- System RAM and swap usage with colored bars
- GPU status (temperature, utilization, memory)
- Active training processes
- OOM protection status
- Top memory consumers
- Recent alerts

**Usage:** Run in separate terminal for live monitoring.

---

### 6. **increase_swap.sh** (Optional, requires sudo)
Increases swap file size for better OOM protection.

```bash
sudo scripts/system_protection/increase_swap.sh
```

**Default:** Creates 24GB swap file (adjust in script as needed).

**Recommended:** 16-32GB swap for systems with 64GB RAM doing AI training.

---

### 7. **install_watchdog_service.sh** (Optional, requires sudo)
Installs systemd services for automatic watchdog startup on boot.

```bash
sudo scripts/system_protection/install_watchdog_service.sh
```

**Creates services:**
- `memory-watchdog.service`
- `gpu-watchdog.service`

**Check status:**
```bash
systemctl status memory-watchdog gpu-watchdog
```

---

## 🚀 Quick Start Guide

### For a Single Training Session

```bash
# 1. Start watchdogs
nohup scripts/system_protection/memory_watchdog.sh &
nohup scripts/system_protection/gpu_watchdog.sh &

# 2. Set OOM priorities
bash scripts/system_protection/set_oom_priorities.sh

# 3. Start training with protection
scripts/system_protection/safe_train_wrapper.sh \
  conda run -n kohya_ss accelerate launch sdxl_train_network.py --config_file <config>

# 4. Monitor in another terminal
scripts/system_protection/unified_monitor_dashboard.sh
```

---

### For Automatic Protection (Set and Forget)

```bash
# 1. Install system services (one-time setup)
sudo scripts/system_protection/install_watchdog_service.sh

# 2. (Optional) Increase swap if needed
sudo scripts/system_protection/increase_swap.sh

# 3. Use safe wrapper for all training
scripts/system_protection/safe_train_wrapper.sh <your_training_command>
```

---

## 🛡️ Protection Mechanisms

### OOM Score Priorities

| Process Type | OOM Score | Protection Level |
|--------------|-----------|------------------|
| SSH Daemon | -1000 | Never kill |
| Desktop Environment | -500 | High protection |
| **Training Processes** | **-300** | **Protected** |
| TensorBoard | -100 | Low protection |
| Caption/Cluster | +200-300 | Can be killed |
| Browsers | +500 | Kill first |

**Lower score = Less likely to be killed**

---

### Memory Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| RAM Usage | 80% | 90% |
| Swap Usage | 50% | 80% |
| GPU Memory | 85% | 95% |
| GPU Temperature | 70°C | 85°C |

---

## 📊 Monitoring & Logs

### View Watchdog Logs

```bash
# Memory alerts
tail -f /tmp/memory_watchdog.log

# GPU alerts
tail -f /tmp/gpu_watchdog.log

# Training logs (if using safe wrapper)
ls -lth /tmp/safe_training_logs/
```

---

### Check Protection Status

```bash
# Check if watchdogs are running
pgrep -f watchdog

# Check OOM scores of training processes
ps aux | grep python.*train
cat /proc/<PID>/oom_score_adj
```

---

## ⚙️ Configuration

### Adjust Memory Thresholds

Edit `memory_watchdog.sh`:

```bash
RAM_CRITICAL=90    # % - critical threshold (default: 90)
RAM_WARNING=80     # % - warning threshold (default: 80)
SWAP_CRITICAL=80   # % - swap usage critical (default: 80)
CHECK_INTERVAL=10  # seconds between checks (default: 10)
```

### Adjust OOM Scores

Edit `set_oom_priorities.sh` to customize process priorities:

```bash
set_oom_score "python.*train.*" -300 "Training Processes"
# Change -300 to lower number for more protection
```

---

## 🔧 Troubleshooting

### Desktop Still Crashes

1. **Increase swap size:**
   ```bash
   sudo scripts/system_protection/increase_swap.sh
   ```

2. **Lower swappiness** (prefer RAM over swap):
   ```bash
   sudo sysctl vm.swappiness=10
   # Make permanent: echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
   ```

3. **Set stricter memory limits** in training config:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use 8-bit optimizers (AdamW8bit)

### Watchdog Not Sending Notifications

Check if `notify-send` is available:
```bash
which notify-send
# If missing: sudo apt install libnotify-bin
```

Test notification:
```bash
DISPLAY=:0 notify-send "Test" "This is a test notification"
```

### Training Still Gets Killed

1. Check actual OOM killer events:
   ```bash
   dmesg | grep -i oom
   journalctl -k | grep -i "out of memory"
   ```

2. Lower training process priority more:
   ```bash
   # Edit set_oom_priorities.sh
   set_oom_score "python.*train.*" -500 "Training Processes"
   ```

3. Consider splitting training into smaller batches or using model parallelism.

---

## 📌 Best Practices

1. **Always use safe_train_wrapper.sh** for long-running training
2. **Monitor first epoch** to establish baseline memory usage
3. **Close unnecessary applications** before training
4. **Keep swap size at least 1/4 of RAM** (e.g., 16GB swap for 64GB RAM)
5. **Run unified dashboard** in separate terminal during training
6. **Check logs after any crash** to identify root cause

---

## 🔗 Related Documentation

- [WSL Long Running Guide](../../docs/setup/WSL_LONG_RUNNING_GUIDE.md) - For WSL-specific concerns
- [Training Configuration Guide](../../docs/guides/training/) - Optimize training parameters
- Linux OOM Killer: `man oom_score_adj`

---

## 📝 Notes

- All watchdog scripts are **non-intrusive** - they only monitor and alert
- **No processes are automatically killed** unless you uncomment kill commands
- Scripts are **stateless** - can be stopped and restarted anytime
- Logs are written to `/tmp/` - cleared on reboot

---

## ⚖️ License

Part of the 3D Animation LoRA Pipeline project.
