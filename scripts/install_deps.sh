#!/bin/bash
set -e

# Install all dependencies for GateANN.
# Tested on Ubuntu 22.04 LTS.
#
# Usage: ./scripts/install_deps.sh

echo "=== Installing GateANN dependencies ==="

# System packages
echo "[1/3] Installing system packages..."
sudo apt-get update
sudo apt-get install -y build-essential cmake g++ libmkl-dev \
    libomp-dev libgoogle-perftools-dev python3 python3-pip

# Python packages
echo "[2/3] Installing Python packages..."
pip3 install matplotlib numpy

# liburing (included in third_party/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIBURING_DIR="$REPO_ROOT/third_party/liburing"

echo "[3/3] Building liburing..."
if [ -d "$LIBURING_DIR" ]; then
    cd "$LIBURING_DIR"
    ./configure
    make -j$(nproc)
    cd "$REPO_ROOT"
    echo "liburing built successfully."
else
    echo "WARNING: $LIBURING_DIR not found. Install liburing manually."
fi

echo "=== All dependencies installed ==="
