#!/usr/bin/env bash
#
# ============================================================================
# Fix NVIDIA NVML "Driver/library version mismatch" (Ubuntu 24.04)
# ============================================================================
#
# Symptoms:
#   nvidia-smi -> Failed to initialize NVML: Driver/library version mismatch
#
# Common cause:
#   NVIDIA packages updated, but the old kernel module is still loaded
#   (pending reboot), OR multiple driver branches are installed.
#
# This script:
#   - Prints a concise diagnostic
#   - Removes conflicting 57x packages (optional but recommended)
#   - Reinstalls the target 58x driver branch (default: 580-open)
#   - Reminds you to reboot
#
# Usage:
#   sudo bash scripts/setup/fix_nvidia_ubuntu24_nvml_mismatch.sh
#
# Optional:
#   NVIDIA_BRANCH=580 sudo bash scripts/setup/fix_nvidia_ubuntu24_nvml_mismatch.sh
#   NVIDIA_FLAVOR=open sudo bash scripts/setup/fix_nvidia_ubuntu24_nvml_mismatch.sh
#
# Author: Codex CLI
# Date: 2026-01-26
# ============================================================================

set -euo pipefail

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "ERROR: Please run as root:"
  echo "  sudo bash $0"
  exit 1
fi

NVIDIA_BRANCH="${NVIDIA_BRANCH:-580}"
NVIDIA_FLAVOR="${NVIDIA_FLAVOR:-open}" # open | (empty/other)

is_wsl=0
if rg -qi "microsoft|wsl" /proc/version 2>/dev/null; then
  is_wsl=1
fi

echo "========================================================================"
echo "NVIDIA Fix Script (Ubuntu 24.04) - $(date -Is)"
echo "========================================================================"

echo ""
echo "[1/4] System info"
lsb_release -a 2>/dev/null || cat /etc/os-release
echo "Kernel: $(uname -r)"
if [[ "$is_wsl" -eq 1 ]]; then
  echo "WSL: detected"
  echo ""
  echo "NOTE: In WSL2, install/update the NVIDIA driver on Windows (host),"
  echo "      then ensure WSL GPU support is enabled. Linux-side driver installs"
  echo "      are typically not needed and can break things."
  echo "========================================================================"
  exit 2
fi

echo ""
echo "[2/4] GPU detection (lspci)"
if command -v lspci >/dev/null 2>&1; then
  lspci | rg -i 'nvidia|vga|3d|display' || true
else
  echo "WARN: lspci not found (install pciutils if needed)."
fi

echo ""
echo "[3/4] Current NVIDIA status"
echo "nvidia-smi:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "  (nvidia-smi not found)"
fi
echo ""
echo "Loaded kernel module version (/proc/driver/nvidia/version):"
cat /proc/driver/nvidia/version 2>/dev/null || echo "  (not available)"

pkg_ver() {
  local pkg="$1"
  dpkg-query -W -f='${Version}\n' "$pkg" 2>/dev/null || true
}

target_pkg="nvidia-utils-${NVIDIA_BRANCH}"
target_ver="$(pkg_ver "$target_pkg")"
echo ""
echo "Installed userspace package version ($target_pkg): ${target_ver:-'(not installed)'}"

echo ""
echo "[4/4] Apply fix"
echo "Target: nvidia-driver-${NVIDIA_BRANCH}-${NVIDIA_FLAVOR}"

export DEBIAN_FRONTEND=noninteractive

apt-get update

# Remove common conflicting packages from 575 branch if present (safe no-op if absent).
echo ""
echo "Removing conflicting 575 packages (if installed)..."
apt-get purge -y \
  nvidia-driver-575-open \
  nvidia-kernel-common-575 \
  'libnvidia-gl-575*' \
  'libnvidia-compute-575*' || true

echo ""
echo "Installing/reinstalling target ${NVIDIA_BRANCH} packages..."

driver_pkg="nvidia-driver-${NVIDIA_BRANCH}"
dkms_pkg="nvidia-dkms-${NVIDIA_BRANCH}"
kernel_src_pkg="nvidia-kernel-source-${NVIDIA_BRANCH}"

if [[ "$NVIDIA_FLAVOR" == "open" ]]; then
  driver_pkg="nvidia-driver-${NVIDIA_BRANCH}-open"
  dkms_pkg="nvidia-dkms-${NVIDIA_BRANCH}-open"
  kernel_src_pkg="nvidia-kernel-source-${NVIDIA_BRANCH}-open"
fi

apt-get install -y --reinstall \
  "linux-headers-$(uname -r)" \
  "$driver_pkg" \
  "$dkms_pkg" \
  "nvidia-utils-${NVIDIA_BRANCH}" \
  "nvidia-kernel-common-${NVIDIA_BRANCH}" \
  "$kernel_src_pkg" \
  "libnvidia-compute-${NVIDIA_BRANCH}" \
  "libnvidia-gl-${NVIDIA_BRANCH}" \
  "libnvidia-extra-${NVIDIA_BRANCH}" \
  "nvidia-compute-utils-${NVIDIA_BRANCH}"

echo ""
echo "Cleaning up..."
apt-get autoremove -y
update-initramfs -u

echo ""
echo "========================================================================"
echo "✅ Done. Reboot is REQUIRED to load the matching kernel module."
echo "Run:"
echo "  sudo reboot"
echo ""
echo "After reboot verify:"
echo "  nvidia-smi"
echo "  conda run -n ai_env python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.device_count())\""
echo "========================================================================"

