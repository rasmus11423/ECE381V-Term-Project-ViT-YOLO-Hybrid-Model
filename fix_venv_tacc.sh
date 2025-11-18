#!/bin/bash

# Script to fix PyTorch CUDA installation in existing venv on TACC
# Only recreates venv if there's a serious problem (like shared library errors)

set -e

echo "=== Fixing PyTorch CUDA Installation on TACC ==="
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

# Check if venv exists and is usable
VENV_EXISTS=false
VENV_BROKEN=false

if [ -d "venv" ]; then
    VENV_EXISTS=true
    echo "Virtual environment found. Checking if it's usable..."
    
    # Try to activate and test if Python works
    if source venv/bin/activate 2>/dev/null; then
        # Test if Python can run (check for shared library errors)
        if python3 -c "import sys" 2>/dev/null; then
            echo "✓ Existing venv is usable"
        else
            echo "✗ Venv exists but Python has shared library errors"
            VENV_BROKEN=true
        fi
    else
        echo "✗ Venv exists but cannot be activated"
        VENV_BROKEN=true
    fi
fi

# Only recreate venv if it doesn't exist or is broken
if [ "$VENV_EXISTS" = false ] || [ "$VENV_BROKEN" = true ]; then
    if [ "$VENV_BROKEN" = true ]; then
        echo ""
        echo "Removing broken virtual environment..."
        rm -rf venv
    fi
    
    echo "Creating new virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
    echo ""
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip --quiet
    
    # Install all base packages
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
        neptune \
        transformers>=4.42.0
else
    # Venv exists and works - just activate it
    echo ""
    echo "Activating existing virtual environment..."
    source venv/bin/activate
    
    # Check if PyTorch needs to be fixed
    echo "Checking PyTorch installation..."
    PYTORCH_HAS_CUDA=$(python3 -c "import torch; print('True' if torch.version.cuda is not None else 'False')" 2>/dev/null || echo "False")
    
    if [ "$PYTORCH_HAS_CUDA" = "False" ]; then
        echo "⚠️  PyTorch is installed but without CUDA support"
        echo "   Fixing PyTorch installation..."
    else
        echo "✓ PyTorch already has CUDA support"
        echo "   PyTorch version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
        echo ""
        echo "If you want to reinstall PyTorch anyway, use:"
        echo "  bash fix_venv_tacc.sh --force-reinstall-pytorch"
        echo ""
        
        # Check if --force-reinstall-pytorch flag was passed
        if [ "$1" != "--force-reinstall-pytorch" ]; then
            echo "Skipping PyTorch reinstallation. All done!"
            exit 0
        else
            echo "Force reinstalling PyTorch as requested..."
        fi
    fi
fi

# Install/fix PyTorch with CUDA support (required for both scripts)
echo "Installing PyTorch with CUDA support..."
echo "Note: TACC uses CUDA 12.8, installing PyTorch with CUDA 12.4+ support..."
echo "Force installing CUDA-enabled PyTorch (even if CUDA not detected on login node)..."

# First, uninstall any existing PyTorch (CPU or CUDA) to avoid conflicts
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Force install CUDA-enabled PyTorch using ONLY the CUDA index URL
# Use --index-url (not --extra-index-url) to ONLY use CUDA repo, not PyPI
# Use --no-cache-dir to avoid cached CPU versions
echo "Installing PyTorch from CUDA wheel repository (CUDA 12.4)..."
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# Verify we got CUDA version by checking Python
echo "Verifying PyTorch has CUDA support..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA built-in:', torch.version.cuda is not None); print('CUDA version:', torch.version.cuda if torch.version.cuda else 'None (CPU version!)')" || echo "Could not verify PyTorch CUDA support"

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

# Check if PyTorch was built with CUDA support
print(f'PyTorch built with CUDA: {torch.version.cuda is not None}')
if torch.version.cuda:
    print(f'PyTorch CUDA version: {torch.version.cuda}')
else:
    print('⚠️  WARNING: PyTorch was installed without CUDA support!')
    print('   This means the CPU version was installed instead of CUDA version.')
    print('   Training will NOT use GPUs even on GPU nodes!')

print(f'CUDA available (on this node): {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Number of GPUs: {torch.cuda.device_count()}')
else:
    print('⚠️  CUDA not available on login node (this is normal)')
    print('   CUDA will be available when job runs on GPU compute nodes')
    print('   IMPORTANT: PyTorch must be CUDA-enabled for GPU training to work!')

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

