#!/bin/bash

# Shell script to set up ExDark dataset training environment for TACC
# This script creates a virtual environment and installs all required packages
# 
# Usage on TACC:
# 1. Load required modules: module load python3/3.11.8 gcc/13.2.0 cuda/12.8
# 2. Run this script: bash setup_exdark_dataset.sh

set -e  # Exit on error

echo "=== ExDark Dataset Setup Script for TACC ==="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    echo "Please load Python module first:"
    echo "  module load python3/3.11.8"
    echo "  module load gcc/13.2.0"
    echo "  module load cuda/12.8"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Python version: $PYTHON_VERSION"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install required packages
echo "Installing required packages..."
echo "This may take a few minutes..."
pip install --quiet ultralytics>=8.0.0 opencv-python tqdm lxml albumentations gdown matplotlib pandas numpy neptune

# Install PyTorch with CUDA support for TACC
echo "Installing PyTorch with CUDA support..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=== Environment Setup Complete ==="
echo ""
echo "Virtual environment is ready at: $SCRIPT_DIR/venv"
echo ""
echo "To activate the environment manually, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run training, use the run_YOLOv5_ExDark_Dataset_train.sh script with sbatch"
echo ""
