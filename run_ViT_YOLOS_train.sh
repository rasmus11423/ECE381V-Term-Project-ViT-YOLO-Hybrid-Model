#!/bin/bash
#SBATCH -J vit_yolos_exdark_train
#SBATCH -o Output/out_%j.txt
#SBATCH -e Error/error_%j.txt
#SBATCH -p gg                                              # Grace Hopper GPU partition
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user='rl37272@my.utexas.edu'

# Reset modules to clear any conflicts
module reset

# Load dependencies in correct order (required before python3/3.11.8)
echo "Loading required modules..."
module load gcc/13.2.0
module load cuda/12.8
module load python3/3.11.8

# Activate virtual environment
source venv/bin/activate

# ==============================
# Log metadata
# ==============================
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python version: $(python --version)"
echo "=========================================="

# ==============================
# Verify required packages are installed
# ==============================
echo "Verifying required packages..."
python -c "import torch; import transformers; print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo "Warning: Some packages may be missing. The script will attempt to install them automatically."
}

# ==============================
# Run YOLOS training script
# ==============================
python --version
which python
python ViT_YOLOS_train.py

echo "=========================================="
echo "End Time: $(date)"
echo "Training completed!"
echo "=========================================="

