#!/bin/bash

# Script to fix venv issues on TACC
# This recreates the venv with proper module loading

set -e

echo "=== Fixing Virtual Environment on TACC ==="
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
    echo "  module reset"
    echo "  module load gcc/13.2.0"
    echo "  module load cuda/12.8"
    echo "  module load python3/3.11.8"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Python version: $PYTHON_VERSION"
echo ""

# Remove existing venv if it exists
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
    echo "Old venv removed."
fi

# Create new virtual environment
echo "Creating new virtual environment with loaded Python module..."
python3 -m venv venv
echo "Virtual environment created."
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install base packages (required for both YOLOv5_ExDark_Dataset_train.py and ViT_YOLOS_train.py)
echo "Installing base packages..."
echo "This may take a few minutes..."
pip install --quiet \
    ultralytics>=8.0.0 \
    opencv-python \
    pillow \
    tqdm \
    lxml \
    albumentations \
    gdown \
    matplotlib \
    pandas \
    numpy \
    neptune

# Install PyTorch with CUDA support (required for both scripts)
echo "Installing PyTorch with CUDA support..."
echo "Note: TACC uses CUDA 12.8, installing PyTorch with CUDA 12.4+ support..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install transformers (required only for ViT_YOLOS_train.py)
echo "Installing transformers library (for YOLOS/ViT training)..."
pip install --quiet transformers>=4.42.0

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
import transformers
from ultralytics import YOLO
from PIL import Image
import cv2
print('✓ All packages imported successfully')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
else:
    print('⚠️  CUDA not available (this is normal on login nodes)')
    print('   CUDA will be available when job runs on GPU compute nodes')
print('✓ Environment ready for both YOLOv5 and YOLOS training')
" || echo "Warning: Could not verify installation"

echo ""
echo "=== Virtual Environment Fixed ==="
echo ""
echo "Virtual environment is ready at: $SCRIPT_DIR/venv"
echo ""
echo "To activate the environment, run:"
echo "  module load python3/3.11.8  # Load module first!"
echo "  source venv/bin/activate"
echo ""
echo "This venv supports both:"
echo "  - YOLOv5 training: sbatch run_YOLOv5_ExDark_Dataset_train.sh"
echo "  - YOLOS training: sbatch run_ViT_YOLOS_train.sh"
echo ""

