#!/bin/bash
#SBATCH -J hybrid_exdark_train
#SBATCH -o Output/out_%j.txt
#SBATCH -e Error/error_%j.txt
#SBATCH -p gh                                              # Grace-Hopper GPU partition (H200 GPUs)
#SBATCH -N 1                                               # Start with 1 node (faster queue time)
#SBATCH -n 1                                               # 1 task per node
#SBATCH -t 00:10:00                                        # Max 48 hours on TACC Vista
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
echo "Checking node features..."
scontrol show node $SLURM_NODELIST | grep -E "NodeName|Features|Gres" | head -20 || echo "Could not query node features"
echo "=========================================="

# ==============================
# Verify GPU and CUDA availability
# ==============================
echo "Checking CUDA environment..."
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Checking for GPUs with nvidia-smi..."
which nvidia-smi && nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || echo "nvidia-smi not available"

echo "Verifying required packages..."
python -c "import torch; import transformers; print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.version.cuda else \"None\"}'); print(f'Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" || {
    echo "Warning: Some packages may be missing. The script will attempt to install them automatically."
}

# ==============================
# Run YOLOS training script
# ==============================
python --version
which python
python Hybrid_ViT_YOLO_train.py

echo "=========================================="
echo "End Time: $(date)"
echo "Training completed!"
echo "=========================================="

