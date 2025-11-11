#!/bin/bash

#SBATCH -J yolo_exdark_train

#SBATCH -o Output/out_%j.txt

#SBATCH -e Error/error_%j.txt

#SBATCH -p gg                                              # Grace Hopper GPU partition

#SBATCH -N 6

#SBATCH -n 6

#SBATCH -t 24:00:00

#SBATCH --mail-type=ALL

#SBATCH --mail-user='rl37272@my.utexas.edu'

# Reset modules to clear any conflicts
module reset

# Load dependencies in correct order (required before python3/3.11.8)
echo "Loading required modules..."
module load gcc/13.2.0
module load cuda/12.8

# Load Python module (requires gcc and cuda to be loaded first)
echo "Loading Python module..."
module load python3/3.11.8

# Verify Python is loaded
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found after loading module"
    echo "Current modules:"
    module list
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create Output and Error directories if they don't exist
mkdir -p Output
mkdir -p Error

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run setup_exdark_dataset.sh first."
    exit 1
fi

source venv/bin/activate

# ==============================
# Log metadata
# ==============================
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"
echo "Python Path: $(which python)"
echo "=========================================="

# ==============================
# Run training script
# ==============================
echo ""
echo "Starting YOLOv5 ExDark dataset training..."
echo ""

python YOLOv5_ExDark_Dataset_train.py

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Training completed!"
echo "=========================================="

