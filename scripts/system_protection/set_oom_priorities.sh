#!/bin/bash
# Set OOM scores to protect critical processes
# Lower score = less likely to be killed
# Score range: -1000 (never kill) to 1000 (kill first)

echo "Setting OOM protection priorities..."

# Function to set OOM score
set_oom_score() {
    local pattern=$1
    local score=$2
    local description=$3

    pids=$(pgrep -f "$pattern" 2>/dev/null)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            if [ -w /proc/$pid/oom_score_adj ]; then
                echo $score > /proc/$pid/oom_score_adj 2>/dev/null
                if [ $? -eq 0 ]; then
                    echo "  ✓ Protected: $description (PID: $pid, score: $score)"
                fi
            fi
        done
    fi
}

# Critical system processes (never kill)
set_oom_score "sshd" -1000 "SSH Daemon"
set_oom_score "systemd-logind" -1000 "Login Manager"

# Desktop environment (protect from OOM, but allow if necessary)
set_oom_score "plasmashell" -500 "KDE Plasma Desktop"
set_oom_score "kwin" -500 "KDE Window Manager"
set_oom_score "gnome-shell" -500 "GNOME Shell"

# AI training processes (protect, but can be killed if system critical)
set_oom_score "python.*train.*" -300 "Training Processes"
set_oom_score "python.*kohya" -300 "Kohya Training"
set_oom_score "accelerate" -300 "Accelerate Launcher"
set_oom_score "tensorboard" -100 "TensorBoard"

# High-memory processes (kill first if needed)
set_oom_score "python.*caption" 300 "Caption Generation"
set_oom_score "python.*cluster" 200 "Clustering"
set_oom_score "chrome|chromium|firefox" 500 "Web Browsers"

echo ""
echo "✓ OOM priorities configured"
