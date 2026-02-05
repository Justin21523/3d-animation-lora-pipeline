#!/bin/bash
# Install systemd services for automatic watchdog startup

if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo to install system services"
    exit 1
fi

USER_NAME=$(logname)

# Memory Watchdog Service
cat > /etc/systemd/system/memory-watchdog.service << EOF
[Unit]
Description=Memory Watchdog for AI Training
After=multi-user.target

[Service]
Type=simple
User=$USER_NAME
ExecStart=/tmp/memory_watchdog.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# GPU Watchdog Service
cat > /etc/systemd/system/gpu-watchdog.service << EOF
[Unit]
Description=GPU Memory Watchdog
After=multi-user.target

[Service]
Type=simple
User=$USER_NAME
ExecStart=/tmp/gpu_watchdog.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable memory-watchdog.service
systemctl enable gpu-watchdog.service
systemctl start memory-watchdog.service
systemctl start gpu-watchdog.service

echo "✓ Watchdog services installed and started"
echo "  Check status: systemctl status memory-watchdog gpu-watchdog"
