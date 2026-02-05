#!/bin/bash
# Increase swap size for better OOM protection
# Recommended: 16-32GB for AI workloads with 64GB RAM

if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo"
    exit 1
fi

SWAP_SIZE_GB=24  # Adjust as needed

echo "Creating ${SWAP_SIZE_GB}GB swap file..."
echo "This may take several minutes..."

# Turn off current swap
swapoff -a

# Create new swap file
dd if=/dev/zero of=/swapfile bs=1G count=$SWAP_SIZE_GB status=progress
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

# Make permanent
if ! grep -q "/swapfile" /etc/fstab; then
    echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi

echo ""
echo "✓ Swap increased to ${SWAP_SIZE_GB}GB"
swapon --show
