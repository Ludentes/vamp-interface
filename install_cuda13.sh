#!/usr/bin/env bash
# Install CUDA 13.0 toolkit on Ubuntu 24.04 alongside existing CUDA 12.8.
# Needed so nvdiffrast can build against torch 2.11.0+cu130 in the project .venv.
# Run with:  bash install_cuda13.sh
set -euo pipefail

KEYRING_DEB=/tmp/cuda-keyring_1.1-1_all.deb

echo "[1/4] Downloading NVIDIA CUDA keyring..."
wget -q -O "$KEYRING_DEB" \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb

echo "[2/4] Installing keyring (sudo)..."
sudo dpkg -i "$KEYRING_DEB"

echo "[3/4] apt-get update..."
sudo apt-get update

echo "[4/4] Installing cuda-toolkit-13-0 (~5 GB)..."
sudo apt-get install -y cuda-toolkit-13-0

echo
echo "Done. Verify:"
echo "  /usr/local/cuda-13.0/bin/nvcc --version"
