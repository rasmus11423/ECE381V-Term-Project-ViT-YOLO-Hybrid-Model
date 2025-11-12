#!/bin/bash
# Quick check to verify PyTorch CUDA installation

module reset
module load gcc/13.2.0
module load cuda/12.8
module load python3/3.11.8

source venv/bin/activate

python -c "
import torch
print('='*60)
print('PyTorch CUDA Check:')
print('='*60)
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch built with CUDA: {torch.version.cuda is not None}')
if torch.version.cuda:
    print(f'✓ CUDA version in PyTorch build: {torch.version.cuda}')
    print('✓ PyTorch HAS CUDA support!')
    print('')
    print(f'CUDA available at runtime: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print('✓ CUDA is available (GPUs detected)')
    else:
        print('⚠️  CUDA not available at runtime (normal on login nodes)')
        print('   Will work on compute nodes with GPUs')
else:
    print('✗ PyTorch was installed WITHOUT CUDA support!')
    print('  Need to reinstall with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124')
print('='*60)
"

