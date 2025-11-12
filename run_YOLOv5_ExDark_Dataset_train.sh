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
module load python3/3.11.8

# Activate virtual environment
source venv/bin/activate


# ==============================
# 3️⃣ Log metadata
# ==============================
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# ==============================
# 5️⃣ Run test script
# ==============================
python --version
which python
python YOLOv5_ExDark_Dataset_train.py

echo "=========================================="
echo "End Time: $(date)"
echo "Test completed successfully!"
echo "=========================================="




