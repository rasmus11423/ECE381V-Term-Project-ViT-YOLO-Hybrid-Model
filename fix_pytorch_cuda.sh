#!/bin/bash
# Script to fix PyTorch CUDA installation on TACC
# This script loads required modules and reinstalls PyTorch with CUDA support

set -e

echo "=== Fixing PyTorch CUDA Installation ==="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"
echo ""

# Load required modules in correct order
echo "Loading required modules..."
module reset
module load gcc/13.2.0
module load cuda/12.8
module load python3/3.11.8

echo "Modules loaded:"
echo "  Python: $(python3 --version)"
echo "  CUDA: $(module list 2>&1 | grep cuda || echo 'cuda/12.8')"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "Python in venv: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Uninstall old PyTorch
echo "Uninstalling old PyTorch installation..."
pip uninstall torch torchvision -y || echo "No existing PyTorch to uninstall"
echo ""

# Install PyTorch with CUDA 12.4+ support (compatible with TACC's CUDA 12.8)
echo "Installing PyTorch with CUDA 12.4+ support..."
echo "This may take a few minutes..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo ""

# Verify PyTorch CUDA installation
echo "Verifying PyTorch CUDA installation..."
python -c "
import torch
print('='*60)
print('PyTorch Installation Verification:')
print('='*60)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version (PyTorch): {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'    Memory: {props.total_memory / 1e9:.2f} GB')
else:
    print('⚠️  WARNING: CUDA not available!')
    print('This may be normal on login nodes (no GPUs).')
    print('CUDA should work on compute nodes during job execution.')
print('='*60)
"

echo ""
echo "=== PyTorch CUDA Fix Complete ==="
echo ""
echo "Note: If CUDA shows as unavailable, this is normal on login nodes."
echo "CUDA will be available when your job runs on compute nodes with GPUs."
echo ""
echo "You can now resubmit your training job:"
echo "  sbatch run_YOLOv5_ExDark_Dataset_train.sh"
echo ""

