#!/bin/bash

# Shell script to set up YOLOS (ViT) training environment for TACC
# This script updates the existing venv or creates a new one with all required packages
# 
# Usage on TACC:
# 1. Load required modules in this order:
#    module reset
#    module load gcc/13.2.0
#    module load cuda/12.8
#    module load python3/3.11.8
# 2. Run this script: bash setup_yolos_venv.sh
#    (or if venv already exists from YOLOv5 setup, this will just add transformers)

set -e  # Exit on error

echo "=== YOLOS (ViT) Training Setup Script for TACC ==="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    echo "Please load Python module first in this order:"
    echo "  module reset"
    echo "  module load gcc/13.2.0"
    echo "  module load cuda/12.8"
    echo "  module load python3/3.11.8"
    echo ""
    echo "Note: gcc and cuda must be loaded BEFORE python3/3.11.8"
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
    echo "Virtual environment already exists (from YOLOv5 setup)."
    echo "Will add YOLOS-specific packages..."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install base packages (if not already installed)
echo "Installing/updating base packages..."
echo "This may take a few minutes..."
pip install --quiet ultralytics>=8.0.0 opencv-python tqdm lxml albumentations gdown matplotlib pandas numpy neptune

# Install PyTorch with CUDA support for TACC (if not already installed)
# Try to match CUDA 12.8 on TACC - use cu124 (closest available) or default
echo "Installing/updating PyTorch with CUDA support..."
echo "Note: TACC uses CUDA 12.8, installing PyTorch with CUDA 12.4+ support..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install transformers library (required for YOLOS)
echo "Installing transformers library (required for YOLOS)..."
pip install --quiet transformers>=4.42.0

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
import transformers
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
" || echo "Warning: Could not verify installation"

echo ""
echo "=== Environment Setup Complete ==="
echo ""
echo "Virtual environment is ready at: $SCRIPT_DIR/venv"
echo ""
echo "To activate the environment manually, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run YOLOS training, use the run_ViT_YOLOS_train.sh script with sbatch:"
echo "  sbatch run_ViT_YOLOS_train.sh"
echo ""

