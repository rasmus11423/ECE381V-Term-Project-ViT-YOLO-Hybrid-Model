# ECE381V-Term-Project-ViT-YOLO-Hybrid-Model

Hybrid ViT-YOLO for Low-Light Object Detection — A research project integrating Vision Transformers with YOLO to enhance object detection under low-light conditions. This project implements two hybrid approaches: (1) ViT-Based Attention-Guided Input for YOLO, and (2) Hybrid ViT-YOLO Framework. Both methods are trained on the ExDARK dataset and benchmark performance, interpretability, and robustness using attention visualizations.

## Authors
Rasmus Laansalu, Fatima Al-Janahi, Yukai Hao

## 📊 Dataset: ExDark

The **ExDark (Extended Dark)** dataset is a low-light object detection dataset designed for evaluating object detection algorithms in challenging lighting conditions. 

### Dataset Details
- **12 Object Classes**: Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table
- **Total Images**: ~7,363 annotated images
- **Format**: Images organized by class folders, with corresponding ground truth annotations in text format

### Download Process
The dataset is automatically downloaded from Google Drive when running the training scripts or notebook:

1. **Main Dataset** (~1.5GB)
   - Google Drive File ID: `1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx`
   - Contains images organized in class-specific folders under `Data/ExDark/`

2. **Ground Truth Annotations** (~5MB)
   - Google Drive File ID: `1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i`
   - Contains annotation text files organized by class under `GroundTruth/ExDark_Annno/`

### Labeling Process
The annotation files are in text format with the following structure:
- Each line contains: `Class_Name x y width height` (absolute pixel coordinates)
- The functions `Exdark_Annotation_Line()` and `Exdark_Annotation_File()` parse these annotations
- Annotations are converted from absolute coordinates to YOLO format (normalized center coordinates: `class_id cx cy w h`)
- The conversion ensures bounding boxes are properly normalized and clipped to image boundaries
- Images are split into train/val/test sets and organized in `Output/images/` and `Output/labels/` directories

## 🔬 Methods

### Method 1: ViT-Based Attention-Guided Input for YOLO
**File**: `ExDark_Dataset_Download.ipynb`

This method uses a Vision Transformer (ViT) to generate attention maps that guide the input preprocessing for YOLO:

- **Architecture**: Uses `google/vit-base-patch16-224` to extract attention weights
- **Process**: 
  - ViT processes images and outputs attention maps via attention rollout
  - Attention maps are upsampled to original image resolution
  - Attention-guided masks are applied to enhance important regions in low-light images
  - The enhanced images are then used as input for YOLO detection
- **Output**: Generates attention-masked images that highlight important regions for object detection
- **Environment**: Designed to run in Jupyter notebook environments (e.g., Google Colab)

### Method 2: Hybrid ViT-YOLO Framework
**Files**: `Hybrid_ViT_YOLO_train.py`, `run_hybrid_YOLOSS_train.sh`

This method integrates a ViT backbone directly with a YOLO detection head:

- **Architecture**: 
  - Backbone: YOLOS-small model (`hustvl/yolos-small`) - a Vision Transformer pre-trained for object detection
  - Detection Head: Custom YOLO-style detection head with anchor-based predictions
- **Training**: 
  - Uses YOLOv5-style loss function (classification, objectness, and bounding box regression)
  - Supports multi-scale training with image size 640x640
  - Includes Exponential Moving Average (EMA) for model weights
- **Features**:
  - End-to-end trainable hybrid architecture
  - Supports multi-GPU training
  - Comprehensive metrics: mAP, precision, recall, IoU
- **Deployment**: Designed for TACC cluster execution via SLURM

## 🚀 Quick Start on TACC

### 1. Access TACC
Connect to TACC via SSH:
```bash
ssh <your-username>@vista.tacc.utexas.edu
```

Navigate to your project directory:
```bash
cd $WORK/<your-project-path>/ECE381V-Term-Project-ViT-YOLO-Hybrid-Model
```

### 2. Clone Repository (if not already present)
```bash
git clone <your-repo-url>
cd ECE381V-Term-Project-ViT-YOLO-Hybrid-Model
```

### 3. Setup Environment

#### Load Required Modules
```bash
module reset
module load gcc/13.2.0
module load cuda/12.8
module load python3/3.11.8
```

#### Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (required first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all other required packages from requirements.txt
pip install -r requirements.txt
```

### 4. Configure Neptune.ai (Optional)

Neptune.ai is used for experiment tracking and monitoring. To enable it:

1. **Obtain API Token**:
   - Sign up/login at [neptune.ai](https://neptune.ai)
   - Navigate to your profile settings
   - Copy your API token

2. **Create Token File**:
   ```bash
   # Create the token file (replace YOUR_API_TOKEN with your actual token)
   echo "YOUR_API_TOKEN" > neptune_api_token.txt
   ```

3. **Project Configuration**:
   - Project name: `ECE381V-Deep-Learning/ECE381V-Term-Project-ViT-YOLO-Hybrid-Model`
   - The training scripts automatically read the token from `neptune_api_token.txt`
   - If the file doesn't exist, training will continue without Neptune logging

**Note:** The `neptune_api_token.txt` file is excluded from git for security (see `.gitignore`). You must create it manually on TACC. Training works without Neptune but won't log metrics to the dashboard.

### 5. Submit Training Jobs

#### Method 2: Hybrid ViT-YOLO Framework
```bash
sbatch run_hybrid_YOLOSS_train.sh
```

#### Alternative: ViT-YOLOS Training
```bash
sbatch run_ViT_YOLOS_train.sh
```

### 6. Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View output (replace <jobid> with your actual job ID)
tail -f Output/out_<jobid>.txt

# View errors
cat Error/error_<jobid>.txt

# Cancel a job if needed
scancel <jobid>
```

## 📁 Project Structure

```
ECE381V-Term-Project-ViT-YOLO-Hybrid-Model/
│
├── Method 1: ViT-Based Attention-Guided Input for YOLO
│   └── ExDark_Dataset_Download.ipynb          # Jupyter notebook implementing Method 1
│
├── Method 2: Hybrid ViT-YOLO Framework
│   ├── Hybrid_ViT_YOLO_train.py               # Main training script for Method 2
│   └── run_hybrid_YOLOSS_train.sh              # SLURM sbatch script to run Method 2 on TACC
│
├── Additional Training Scripts
│   ├── ViT_YOLOS_train.py                     # Alternative ViT-YOLOS training script
│   └── run_ViT_YOLOS_train.sh                 # SLURM sbatch script for ViT_YOLOS_train.py
│
├── Configuration Files
│   ├── requirements.txt                       # Python package dependencies
│   ├── neptune_api_token.txt                  # Neptune.ai API token (create manually, not in git)
│   └── exdark.yaml                            # Dataset configuration file
│
├── Output Directories
│   ├── Output/                                # Training output logs from SLURM jobs
│   │   ├── out_<jobid>.txt                    # Standard output logs
│   │   ├── images/                            # Processed images (train/val/test splits)
│   │   └── labels/                            # YOLO-format labels (train/val/test splits)
│   ├── Error/                                 # Error logs from SLURM jobs
│   │   └── error_<jobid>.txt                  # Error output logs
│   ├── Data/                                  # Dataset storage (auto-created)
│   │   └── ExDark/                            # ExDark dataset images (organized by class)
│   ├── GroundTruth/                           # Ground truth annotations (auto-created)
│   │   └── ExDark_Annno/                      # Annotation files (organized by class)
│   └── runs/                                  # Training runs and checkpoints
│       └── train/                             # Training outputs and model checkpoints
│
└── README.md                                  # This file
```

## 📝 Notes

### Dataset Download
- Both training scripts automatically download the ExDark dataset (~1.5GB) on first run
- The dataset is downloaded from Google Drive using the `gdown` library
- Ground truth annotations are also automatically downloaded and processed
- Images are automatically split into train/val/test sets and converted to YOLO format

### Training Outputs
- **Method 2 (Hybrid_ViT_YOLO_train.py)**:
  - Training outputs saved to `runs/train/`
  - Training curves: `training_curves_hybrid_vit_yolo_exdark.png`
  - Validation metrics: `validation_metrics_hybrid_vit_yolo_exdark.png`
  - Model checkpoints saved during training

- **ViT_YOLOS_train.py**:
  - Training outputs saved to `runs/train_exdark/`
  - Model checkpoints and training logs

### Logging
- All print statements are captured in `Output/out_<jobid>.txt` for SLURM jobs
- Errors are logged to `Error/error_<jobid>.txt`
- Neptune.ai logs metrics, losses, and hyperparameters (if configured)

### Performance
- Method 2 uses image size 640x640 for training
- Supports multi-GPU training for faster convergence
- Batch size is automatically adjusted based on available GPU memory

## 🔐 Security

- `neptune_api_token.txt` is excluded from git (see `.gitignore`)
- Create this file manually on TACC with your Neptune API token
- Never commit API tokens to version control
- Keep your API tokens secure and do not share them publicly

## 📚 References

- **ExDark Dataset**: Low-light object detection dataset with 12 classes
- **YOLOS**: Vision Transformer for Object Detection (https://github.com/hustvl/YOLOS)
- **Vision Transformer (ViT)**: An Image is Worth 16x16 Words (https://arxiv.org/abs/2010.11929)
- **YOLOv5**: Ultralytics YOLOv5 (https://github.com/ultralytics/yolov5)
