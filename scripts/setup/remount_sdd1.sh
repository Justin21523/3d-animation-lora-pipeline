#!/bin/bash
# Re-mount SDD1 (3.6TB) drive to /mnt/data
# Created: 2025-11-14

set -e

echo "=========================================="
echo "SDD1 (3.6TB) Re-mount Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

# Step 1: Find the device by UUID
TARGET_UUID="03aa943d-f94a-4a9a-842f-0f980176747c"
echo -e "${YELLOW}[1/6]${NC} Searching for device with UUID: ${TARGET_UUID}..."

DEVICE=$(blkid | grep "$TARGET_UUID" | cut -d':' -f1)

if [ -z "$DEVICE" ]; then
    echo -e "${RED}ERROR: Device with UUID ${TARGET_UUID} not found!${NC}"
    echo ""
    echo "Available devices:"
    blkid
    echo ""
    echo "Please check if the drive is connected in Windows."
    exit 1
fi

echo -e "${GREEN}Found device: ${DEVICE}${NC}"

# Step 2: Check device size to confirm it's the 3.6TB drive
echo -e "${YELLOW}[2/6]${NC} Verifying device size..."
DEVICE_SIZE=$(lsblk -b -d -n -o SIZE "$DEVICE" 2>/dev/null || echo "0")
DEVICE_SIZE_GB=$((DEVICE_SIZE / 1024 / 1024 / 1024))

echo "Device size: ${DEVICE_SIZE_GB} GB"

if [ "$DEVICE_SIZE_GB" -lt 3000 ]; then
    echo -e "${YELLOW}WARNING: Device size is less than 3TB. Is this correct?${NC}"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 3: Check if already mounted
echo -e "${YELLOW}[3/6]${NC} Checking current mount status..."
if mount | grep -q "$DEVICE"; then
    CURRENT_MOUNT=$(mount | grep "$DEVICE" | awk '{print $3}')
    echo -e "${YELLOW}Device is already mounted at: ${CURRENT_MOUNT}${NC}"
    read -p "Unmount and remount? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Unmounting ${CURRENT_MOUNT}..."
        umount "$CURRENT_MOUNT"
        echo -e "${GREEN}Unmounted successfully${NC}"
    else
        echo "Keeping current mount. Exiting."
        exit 0
    fi
fi

# Step 4: Ensure mount point exists
MOUNT_POINT="/mnt/data"
echo -e "${YELLOW}[4/6]${NC} Preparing mount point: ${MOUNT_POINT}..."

if [ ! -d "$MOUNT_POINT" ]; then
    echo "Creating mount point..."
    mkdir -p "$MOUNT_POINT"
fi

# Step 5: Mount the device
echo -e "${YELLOW}[5/6]${NC} Mounting ${DEVICE} to ${MOUNT_POINT}..."
mount -t ext4 "$DEVICE" "$MOUNT_POINT"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Mount successful!${NC}"
else
    echo -e "${RED}Mount failed!${NC}"
    exit 1
fi

# Step 6: Verify mount and show info
echo -e "${YELLOW}[6/6]${NC} Verifying mount..."
echo ""
echo "=== Mount Information ==="
df -h "$MOUNT_POINT"
echo ""
echo "=== Directory Contents ==="
ls -lah "$MOUNT_POINT" | head -20
echo ""

# Check for ai_data directory
if [ -d "$MOUNT_POINT/ai_data" ]; then
    echo -e "${GREEN}✓ ai_data directory found${NC}"
    AI_DATA_SIZE=$(du -sh "$MOUNT_POINT/ai_data" 2>/dev/null | cut -f1)
    echo "  Size: ${AI_DATA_SIZE}"
else
    echo -e "${YELLOW}⚠ ai_data directory not found${NC}"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Mount completed successfully!"
echo "==========================================${NC}"
echo ""
echo "Your data is now accessible at: ${MOUNT_POINT}"
echo ""
echo "To make this mount persistent across reboots,"
echo "the /etc/fstab entry is already configured:"
echo "UUID=${TARGET_UUID}  /mnt/data  ext4  defaults,nofail  0  2"
echo ""
