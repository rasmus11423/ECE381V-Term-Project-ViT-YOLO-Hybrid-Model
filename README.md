# ECE381V-Term-Project-ViT-YOLO-Hybrid-Model

Hybrid ViT-YOLO for Low-Light Object Detection — A research project integrating Vision Transformers with YOLO to enhance object detection under low-light conditions. Trained on the ExDARK dataset, it benchmarks performance, interpretability, and robustness using attention visualizations.

## 🚀 Quick Start on TACC

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd ECE381V-Term-Project-ViT-YOLO-Hybrid-Model
```

### 2. Setup Environment
```bash
# Load required modules
module load python3/3.11.8 gcc/13.2.0 cuda/12.8

# Create virtual environment and install packages
bash setup_exdark_dataset.sh
```

### 3. Configure Neptune.ai (Optional)
If you want to use Neptune.ai for experiment tracking, create the API token file:

```bash
# Create the token file (replace YOUR_API_TOKEN with your actual token)
echo "YOUR_API_TOKEN" > neptune_api_token.txt
```

**Note:** The `neptune_api_token.txt` file is excluded from git for security. You must create it manually on TACC. If you don't create it, training will still work but without Neptune logging.

### 4. Submit Training Job
```bash
sbatch run_YOLOv5_ExDark_Dataset_train.sh
```

### 5. Monitor Job
```bash
# Check job status
squeue -u $USER

# View output
tail -f Output/out_<jobid>.txt

# View errors
cat Error/error_<jobid>.txt
```

## 📁 Project Structure

- `YOLOv5_ExDark_Dataset_train.py` - Main training script
- `setup_exdark_dataset.sh` - Environment setup script
- `run_YOLOv5_ExDark_Dataset_train.sh` - SLURM sbatch script
- `neptune_api_token.txt` - Neptune API token (create manually, not in git)

## 📝 Notes

- The script automatically downloads the ExDark dataset (~1.5GB) on first run
- Training outputs are saved to `runs/train/exdark_yolov5/`
- Training curves plot is saved to `runs/train/exdark_yolov5/training_curves.png`
- All print statements are captured in `Output/out_<jobid>.txt`

## 🔐 Security

- `neptune_api_token.txt` is excluded from git (see `.gitignore`)
- Create this file manually on TACC with your Neptune API token
- Never commit API tokens to version control
