#!/usr/bin/env python3
"""
YOLOv5 ExDark Dataset Training Script
Compatible with TACC SSH/SLURM execution via .sh scripts

This script:
1. Downloads and prepares the ExDark dataset
2. Converts annotations to YOLO format
3. Trains YOLOv5 model on the dataset
4. Validates the trained model
5. Generates training curves plots
6. Logs metrics to Neptune.ai (optional)

Usage on TACC:
1. Run setup_exdark_dataset.sh to create venv and install packages
2. Run sbatch run_YOLOv5_ExDark_Dataset_train.sh to submit training job
"""

# %%
import subprocess
import sys
from pathlib import Path
import io, zipfile, shutil, random, re, os
from urllib.request import urlopen, Request
from PIL import Image
import xml.etree.ElementTree as ET

# Install required packages if not already installed
def install_package(package, import_name=None):
    """Install package if not already installed.
    
    Args:
        package: Package name for pip (e.g., 'opencv-python')
        import_name: Name to import (e.g., 'cv2'). If None, derives from package name.
    """
    if import_name is None:
        # Map common package names to import names
        pkg_name = package.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0]
        import_map = {
            'opencv-python': 'cv2',
            'pillow': 'PIL',
            'scikit-learn': 'sklearn',
        }
        import_name = import_map.get(pkg_name, pkg_name.replace('-', '_'))
    
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# Install required packages
required_packages = [
    ("ultralytics>=8.0.0", "ultralytics"),
    ("opencv-python", "cv2"),
    ("tqdm", "tqdm"),
    ("lxml", "lxml"),
    ("albumentations", "albumentations"),
    ("gdown", "gdown"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("neptune", "neptune"),
]
for pkg, import_name in required_packages:
    install_package(pkg, import_name)

# Check for GPU (nvidia-smi) and CUDA environment
print("="*60)
print("GPU/CUDA Diagnostics:")
print("="*60)
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ nvidia-smi available:")
        print(result.stdout[:500])  # First 500 chars
    else:
        print("✗ nvidia-smi not available")
except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
    print(f"✗ nvidia-smi error: {e}")

# Check CUDA environment variables
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

# Check CUDA library paths
cuda_path = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "Not set"))
print(f"CUDA_HOME/CUDA_PATH: {cuda_path}")
print("="*60)

import gdown

# %%
# Use current working directory as project root (or set to notebook directory)
# For local environment, use the directory where this notebook is located
import os
notebook_dir = Path(os.getcwd())  # Current working directory
# Alternative: If you want to use a specific directory, uncomment and modify:
# notebook_dir = Path("/path/to/your/project")

Root_Folder = notebook_dir
Project_Path = Path(Root_Folder)
Data_Path = Project_Path / "Data"
GroundTruth_Path  = Project_Path / "GroundTruth"
ZIP_Path = Data_Path / "exdark.zip"
GroundTruth_ZIP = GroundTruth_Path / "ExDark_Groundtruth.zip"
Images_Path = Project_Path / "Data" / "ExDark"
GroundTruthAnnotation_Path = Project_Path / "GroundTruth" / "ExDark_Annno"
Output = Project_Path / "Output"

Data_Path.mkdir(parents=True, exist_ok=True)
GroundTruth_Path.mkdir(parents=True, exist_ok=True)

print("Project root:", Project_Path.absolute())
print("Data dir:", Data_Path.absolute())
print("GroundTruth dir:", GroundTruth_Path.absolute())

# %%
# Downloading data for local environment
if not (Data_Path / "ExDark").exists():
    file_id = "1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx"
    gdrive_url = f"https://drive.google.com/uc?id={file_id}"

    print("Downloading ExDARK dataset ...")
    print(f"Destination: {ZIP_Path.absolute()}")
    
    # Use gdown library to download
    try:
        gdown.download(gdrive_url, str(ZIP_Path), quiet=False, fuzzy=True)
    except Exception as e:
        print(f"Error downloading with gdown: {e}")
        print("Trying alternative method...")
        # Alternative: use subprocess
        subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", gdrive_url, "-O", str(ZIP_Path)], check=True)

    if not ZIP_Path.exists():
        raise FileNotFoundError(f"Download failed! File not found at {ZIP_Path}")

    print("Extracting files...")
    with zipfile.ZipFile(ZIP_Path, 'r') as zip_ref:
        zip_ref.extractall(Data_Path)

    print("Extraction complete!")

    ZIP_Path.unlink()

    print("Done! ExDARK extracted to:", Data_Path.absolute())
else:
    print("ExDARK already present, skipping download.")

# %%
if not (GroundTruth_Path / "ExDark_Annno").exists():
    file_id = "1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i"
    gdrive_url = f"https://drive.google.com/uc?id={file_id}"

    print("Downloading ExDARK Groundtruth...")
    print(f"Destination: {GroundTruth_ZIP.absolute()}")
    
    # Use gdown library to download
    try:
        gdown.download(gdrive_url, str(GroundTruth_ZIP), quiet=False)
    except Exception as e:
        print(f"Error downloading with gdown: {e}")
        print("Trying alternative method...")
        # Alternative: use subprocess
        subprocess.run([sys.executable, "-m", "gdown", gdrive_url, "-O", str(GroundTruth_ZIP)], check=True)

    if not GroundTruth_ZIP.exists():
        raise FileNotFoundError(f"Download failed! File not found at {GroundTruth_ZIP}")
    else:
        print("Download complete:", GroundTruth_ZIP.absolute())

    print("Extracting files...")
    with zipfile.ZipFile(GroundTruth_ZIP, "r") as zip_ref:
        zip_ref.extractall(GroundTruth_Path)

    print("Extraction complete!")
    print("Groundtruth extracted to:", GroundTruth_Path.absolute())

    GroundTruth_ZIP.unlink(missing_ok=True)
else:
    print("ExDARK annotation already present, skipping download.")

# %%
Images_Format = {".jpg",".jpeg",".png",".bmp"}
Images_Per_Class = {}
Total_Images = 0
Classes = []

for p in sorted(Images_Path.iterdir()):
    if p.is_dir():
        Classes.append(p.name)
        n = len([f for f in p.iterdir() if f.suffix.lower() in Images_Format])
        Images_Per_Class[p.name] = n
        Total_Images += n
Classes_ID = {c:i for i,c in enumerate(Classes)}

print("Classes found under Data/ExDark:", Classes)
print("Total images in Data/ExDark:", Total_Images)
print("Per-class image counts:", Images_Per_Class)

# %%
Annotations_Per_Class = {}
Total_Annotations = 0

for c in Classes:
    gdir = GroundTruthAnnotation_Path / c
    if gdir.exists():
        n = len([f for f in gdir.iterdir() if f.is_file() and f.suffix==".txt"])
        Annotations_Per_Class[c] = n
        Total_Annotations += n
    else:
        Annotations_Per_Class[c] = 0

print("GroundTruth root:", GroundTruthAnnotation_Path)
print("Total GT .txt files for listed classes:", Total_Annotations)
print("Per-class GT counts (txt files):", Annotations_Per_Class)

# %%
def Exdark_Annotation_Line(Line, W, H, Folder_Class):
  Line = Line.strip()
  if not Line or Line.startswith('%'):
    return None

  Parts = re.split(r'[,\s]+', Line)
  if len(Parts) < 5:
    return None

  Class_Name = Parts[0]
  try:
    x, y, w, h = map(float, Parts[1:5])
  except:
    return None

  xmin, ymin = x, y
  xmax, ymax = x + max(0.0, w), y + max(0.0, h)

  xmin = max(0.0, min(float(W), xmin)); ymin = max(0.0, min(float(H), ymin))
  xmax = max(0.0, min(float(W), xmax)); ymax = max(0.0, min(float(H), ymax))
  if xmax <= xmin or ymax <= ymin:
    return None
  cx = ((xmin + xmax)/2.0) / W
  cy = ((ymin + ymax)/2.0) / H
  ww = (xmax - xmin) / W
  hh = (ymax - ymin) / H
  if ww <= 0 or hh <= 0:
    return None

  Class = (Class_Name or Folder_Class).strip()
  return (Class, cx, cy, ww, hh)

def Exdark_Annotation_File(txt_Path, W, H, Folder_Class):
  Out = []
  with open(txt_Path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
      row = Exdark_Annotation_Line(line, W, H, Folder_Class)
      if row:
        Out.append(row)
  return Out

# %%
Records, Missing, Bad = [], 0, 0

for Class in Classes:
  Images_Directory = Images_Path/Class
  GroundTruth_Directory  = GroundTruthAnnotation_Path/Class

  if not GroundTruth_Directory.exists():
    print(f"No groundtruth directory for class '{Class}': {GroundTruth_Directory}")
    continue

  for Img in sorted(Images_Directory.iterdir()):
    if Img.suffix.lower() not in Images_Format:
      continue

    GroundTruth_txt = GroundTruth_Directory / f"{Img.name}.txt"

    if not GroundTruth_txt.exists():
      Missing += 1
      continue

    try:
      with Image.open(Img) as Im:
        W, H = Im.size
    except Exception:
      Bad += 1
      continue

    Annotations = Exdark_Annotation_File(GroundTruth_txt, W, H, Folder_Class=Class)
    if Annotations:
      Records.append((Img, Annotations))

print(f"Annotated Images: {len(Records)}, Missing GroundTruth.txt: {Missing}, Bad Images: {Bad}")

# %%
random.shuffle(Records)
N = len(Records)
N_Train = int(0.70*N)
N_Val = int(0.15*N)
Splits = {"train": Records[:N_Train], "val": Records[N_Train:N_Train+N_Val], "test": Records[N_Train+N_Val:]}
print({k: len(v) for k,v in Splits.items()}, f"Total: {sum(len(v) for v in Splits.values())}")

# %%
for Sp, Items in Splits.items():
    (Output/"images"/Sp).mkdir(parents=True, exist_ok=True)
    (Output/"labels"/Sp).mkdir(parents=True, exist_ok=True)
    for (Img, Anns) in Items:
        Img2 = Output/"images"/Sp/Img.name
        if not Img2.exists():
            shutil.copy2(Img, Img2)
        Lables_Out = Output/"labels"/Sp/(Img.stem + ".txt")
        with open(Lables_Out, "w") as f:
            for (cls, cx, cy, w, h) in Anns:
                f.write(f"{Classes_ID[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

print("YOLO dataset built at:", Output)

# %%
# Use absolute paths in YAML for compatibility
yaml_text = f"""path: {Output.absolute()}
train: {Output.absolute()}/images/train
val: {Output.absolute()}/images/val
test: {Output.absolute()}/images/test
names:
"""
for i,c in enumerate(Classes):
    yaml_text += f"  {i}: {c}\n"
yaml_path = Project_Path / "exdark.yaml"
yaml_path.write_text(yaml_text)
print("Wrote", yaml_path.absolute())
print(yaml_text)

# %%
# Simple YOLOv5 training script using ExDark dataset
from ultralytics import YOLO
import torch
import neptune
from pathlib import Path

# Lists to store metrics for plotting (will be populated during training)
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Initialize Neptune.ai logging (optional - will skip if token file not found)
neptune_run = None
try:
    # Try to read API token from file
    token_file = Project_Path / 'neptune_api_token.txt'
    if token_file.exists():
        with open(token_file, 'r') as f:
            api_token = f.read().strip()
        
        neptune_run = neptune.init_run(
            project="ECE381V-Deep-Learning/ECE381V-Term-Project-ViT-YOLO-Hybrid-Model",
            api_token=api_token,
        )
        print("Neptune.ai logging initialized successfully!")
    else:
        print(f"Neptune token file not found at {token_file}. Skipping Neptune logging.")
        print("To enable Neptune logging, create 'neptune_api_token.txt' in the project root with your API token.")
except Exception as e:
    print(f"Warning: Could not initialize Neptune logging: {e}")
    print("Continuing without Neptune logging...")

# Custom callback to track metrics during training
class MetricsCallback:
    def __init__(self, neptune_run=None):
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.neptune_run = neptune_run
    
    def on_fit_epoch_end(self, trainer):
        """Called at the end of each training epoch"""
        try:
            # Get metrics from trainer - ultralytics stores metrics in trainer.metrics
            if hasattr(trainer, 'metrics'):
                metrics = trainer.metrics
            elif hasattr(trainer, 'results_dict'):
                metrics = trainer.results_dict
            else:
                metrics = {}
            
            # Extract train loss (sum of box_loss + obj_loss + cls_loss)
            train_loss = 0.0
            if 'train/box_loss' in metrics:
                train_loss = metrics.get('train/box_loss', 0) + metrics.get('train/obj_loss', 0) + metrics.get('train/cls_loss', 0)
            elif 'train_loss' in metrics:
                train_loss = metrics.get('train_loss', 0.0)
            
            # Extract validation loss
            val_loss = 0.0
            if 'val/box_loss' in metrics:
                val_loss = metrics.get('val/box_loss', 0) + metrics.get('val/obj_loss', 0) + metrics.get('val/cls_loss', 0)
            elif 'val_loss' in metrics:
                val_loss = metrics.get('val_loss', 0.0)
            else:
                val_loss = train_loss if train_loss > 0 else 0.0  # Fallback to train loss
            
            # Extract accuracy (mAP50) - validation mAP50
            val_acc = metrics.get('metrics/mAP50(B)', metrics.get('metrics/mAP50', metrics.get('mAP50', 0.0)))
            train_acc = val_acc  # For object detection, we typically use val mAP50 for both
            
            # Store metrics
            self.train_losses.append(float(train_loss))
            self.test_losses.append(float(val_loss))
            self.train_accuracies.append(float(train_acc))
            self.test_accuracies.append(float(val_acc))
            
            # Log to Neptune if available
            if self.neptune_run is not None:
                try:
                    epoch = len(self.train_losses)
                    self.neptune_run["train/loss"].append(train_loss)
                    self.neptune_run["train/accuracy"].append(train_acc)
                    self.neptune_run["val/loss"].append(val_loss)
                    self.neptune_run["val/accuracy"].append(val_acc)
                except Exception as e:
                    print(f"Warning: Error logging to Neptune: {e}")
            
            # Print metrics (will be captured in out_jobid.txt when run via .sh)
            epoch = len(self.train_losses)
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}")
        except Exception as e:
            # If callback fails, continue training but log warning
            print(f"Warning: Error in metrics callback: {e}")
            # Add placeholder values to keep lists in sync
            self.train_losses.append(0.0)
            self.test_losses.append(0.0)
            self.train_accuracies.append(0.0)
            self.test_accuracies.append(0.0)

# Check if CUDA is available with detailed diagnostics
print("\n" + "="*60)
print("PyTorch CUDA Diagnostics:")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
else:
    print("⚠️  WARNING: CUDA not available in PyTorch!")
    print("This could be due to:")
    print("  1. PyTorch not installed with CUDA support")
    print("  2. CUDA libraries not properly linked")
    print("  3. CUDA_VISIBLE_DEVICES set incorrectly")
    print("\nAttempting to check CUDA libraries...")
    try:
        import subprocess
        result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=5)
        if "libcuda.so" in result.stdout or "libcudart.so" in result.stdout:
            print("✓ CUDA libraries found in system")
        else:
            print("✗ CUDA libraries not found")
    except:
        pass
print("="*60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")
if device == 'cpu':
    print("⚠️  WARNING: Training will be VERY SLOW on CPU!")
    print("   Consider checking PyTorch CUDA installation.")

# Verify the YAML file exists
yaml_path = Project_Path / "exdark.yaml"
if not yaml_path.exists():
    raise FileNotFoundError(f"YAML file not found at {yaml_path}. Please run the previous cells first.")

print(f"\nUsing dataset YAML: {yaml_path.absolute()}")
print(f"Output directory: {Output.absolute()}")

# Initialize YOLOv5 model
# Options: 'yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt'
# Start with 'yolov5s.pt' (small) for faster training, or 'yolov5m.pt' (medium) for better accuracy
model_size = 'yolov5s.pt'  # Change to 'yolov5m.pt' or 'yolov5l.pt' for larger models
print(f"\nInitializing YOLOv5 model: {model_size}")

model = YOLO(model_size)

# Training parameters (define before Neptune logging)
training_config = {
    'data': str(yaml_path.absolute()),  # Path to dataset YAML
    'epochs': 100,  # Number of training epochs
    'imgsz': 640,  # Image size (pixels)
    'batch': 16,  # Batch size (adjust based on GPU memory: 8, 16, 32, 64)
    'device': device,  # 'cuda' or 'cpu'
    'project': str(Project_Path / 'runs' / 'train'),  # Project directory
    'name': 'exdark_yolov5',  # Experiment name
    'exist_ok': True,  # Overwrite existing experiment
    'save': True,  # Save checkpoints
    'save_period': 10,  # Save checkpoint every N epochs
    'val': True,  # Validate during training
    'plots': False,  # Disable built-in plots (we'll create our own)
    'verbose': True,  # Verbose output
}

# Adjust batch size if using CPU or limited GPU memory
if device == 'cpu':
    training_config['batch'] = 4
    print("Warning: Using CPU. Batch size reduced to 4 for compatibility.")
elif torch.cuda.is_available():
    # Auto-detect batch size based on GPU memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    if gpu_memory < 4:
        training_config['batch'] = 8
        print(f"GPU memory: {gpu_memory:.1f}GB. Using batch size 8.")
    elif gpu_memory < 8:
        training_config['batch'] = 16
        print(f"GPU memory: {gpu_memory:.1f}GB. Using batch size 16.")
    else:
        training_config['batch'] = 32
        print(f"GPU memory: {gpu_memory:.1f}GB. Using batch size 32.")

# Initialize metrics callback with Neptune run
metrics_callback = MetricsCallback(neptune_run=neptune_run)

# Log training parameters to Neptune (after training_config is defined)
if neptune_run is not None:
    try:
        neptune_run["parameters"] = {
            "model_size": model_size,
            "epochs": training_config['epochs'],
            "batch_size": training_config['batch'],
            "imgsz": training_config['imgsz'],
            "device": device,
            "dataset": "ExDark",
            "num_classes": len(Classes),
        }
        # Convert list to string for Neptune compatibility
        neptune_run["parameters/classes"] = ", ".join(Classes)
        print("Training parameters logged to Neptune.")
    except Exception as e:
        print(f"Warning: Error logging parameters to Neptune: {e}")

print("\n" + "="*60)
print("Training Configuration:")
print("="*60)
for key, value in training_config.items():
    print(f"  {key}: {value}")
print("="*60)

# Start training
print("\nStarting training...")
print("This may take a while depending on your hardware and number of epochs.")
print("Progress will be displayed below.\n")

try:
    # Add callback to model using ultralytics callback system
    model.add_callback("on_fit_epoch_end", metrics_callback.on_fit_epoch_end)
    
    results = model.train(**training_config)
    
    # Store metrics in global lists for plotting
    train_losses = metrics_callback.train_losses
    train_accuracies = metrics_callback.train_accuracies
    test_losses = metrics_callback.test_losses
    test_accuracies = metrics_callback.test_accuracies
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print(f"\nResults saved to: {Project_Path / 'runs' / 'train' / 'exdark_yolov5'}")
    print(f"Best weights: {Project_Path / 'runs' / 'train' / 'exdark_yolov5' / 'weights' / 'best.pt'}")
    print(f"Last weights: {Project_Path / 'runs' / 'train' / 'exdark_yolov5' / 'weights' / 'last.pt'}")
    
    # Print final metrics summary
    if len(train_losses) > 0:
        print(f"\nFinal Train Loss: {train_losses[-1]:.4f}, Final Train Acc: {train_accuracies[-1]:.4f}")
        print(f"Final Test Loss: {test_losses[-1]:.4f}, Final Test Acc: {test_accuracies[-1]:.4f}")
        
        # Log final metrics to Neptune
        if neptune_run is not None:
            try:
                neptune_run["final/train_loss"] = train_losses[-1]
                neptune_run["final/train_accuracy"] = train_accuracies[-1]
                neptune_run["final/val_loss"] = test_losses[-1]
                neptune_run["final/val_accuracy"] = test_accuracies[-1]
            except Exception as e:
                print(f"Warning: Error logging final metrics to Neptune: {e}")
    
except Exception as e:
    print(f"\nError during training: {e}")
    print("Please check the error message above and ensure:")
    print("  1. The dataset YAML file exists and is correct")
    print("  2. The dataset paths in the YAML are valid")
    print("  3. You have sufficient disk space and memory")
    print("  4. All required packages are installed")
    raise
finally:
    # Stop Neptune run if it was initialized
    if neptune_run is not None:
        try:
            neptune_run.stop()
            print("Neptune run stopped successfully.")
        except Exception as e:
            print(f"Warning: Error stopping Neptune run: {e}")

# %%
# Optional: Validate the trained model on test set
# Run this cell after training completes to evaluate the model

from ultralytics import YOLO
import torch
import neptune

# Verify the YAML file exists
yaml_path = Project_Path / "exdark.yaml"
if not yaml_path.exists():
    raise FileNotFoundError(f"YAML file not found at {yaml_path}. Please run the previous cells first.")

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Check if training completed and model exists
best_weights = Project_Path / 'runs' / 'train' / 'exdark_yolov5' / 'weights' / 'best.pt'
last_weights = Project_Path / 'runs' / 'train' / 'exdark_yolov5' / 'weights' / 'last.pt'

if not best_weights.exists() and not last_weights.exists():
    print("No trained model found. Please run the training cell first.")
else:
    # Use best weights if available, otherwise use last weights
    weights_path = best_weights if best_weights.exists() else last_weights
    print(f"Loading model from: {weights_path}")
    
    # Load the trained model
    model = YOLO(str(weights_path))
    
    # Initialize Neptune for validation logging (if not already initialized)
    neptune_run_val = None
    try:
        token_file = Project_Path / 'neptune_api_token.txt'
        if token_file.exists():
            with open(token_file, 'r') as f:
                api_token = f.read().strip()
            neptune_run_val = neptune.init_run(
                project="ECE381V-Deep-Learning/ECE381V-Term-Project-ViT-YOLO-Hybrid-Model",
                api_token=api_token,
            )
    except Exception as e:
        pass  # Skip Neptune if not available
    
    # Validate on validation set
    print("\nValidating on validation set...")
    val_results = model.val(
        data=str(yaml_path.absolute()),
        imgsz=640,
        batch=16,
        device=device,
        plots=True,
        save_json=False,
        save_hybrid=False,
    )
    
    print("\n" + "="*60)
    print("Validation Results:")
    print("="*60)
    if hasattr(val_results, 'box'):
        print(f"mAP50: {val_results.box.map50:.4f}")
        print(f"mAP50-95: {val_results.box.map:.4f}")
        print(f"Precision: {val_results.box.mp:.4f}")
        print(f"Recall: {val_results.box.mr:.4f}")
        
        # Log validation results to Neptune
        if neptune_run_val is not None:
            try:
                neptune_run_val["eval/mAP50"] = val_results.box.map50
                neptune_run_val["eval/mAP50-95"] = val_results.box.map
                neptune_run_val["eval/precision"] = val_results.box.mp
                neptune_run_val["eval/recall"] = val_results.box.mr
            except Exception as e:
                print(f"Warning: Error logging validation to Neptune: {e}")
    
    # Optional: Test on test set if available
    test_images_path = Output / "images" / "test"
    if test_images_path.exists() and len(list(test_images_path.glob("*.jpg"))) > 0:
        print("\n" + "="*60)
        print("Testing on test set...")
        print("="*60)
        test_results = model.val(
            data=str(yaml_path.absolute()),
            imgsz=640,
            batch=16,
            device=device,
            plots=True,
        )
        
        if hasattr(test_results, 'box'):
            print(f"\nTest Results:")
            print(f"mAP50: {test_results.box.map50:.4f}")
            print(f"mAP50-95: {test_results.box.map:.4f}")
            print(f"Precision: {test_results.box.mp:.4f}")
            print(f"Recall: {test_results.box.mr:.4f}")
            
            # Log test results to Neptune
            if neptune_run_val is not None:
                try:
                    neptune_run_val["test/mAP50"] = test_results.box.map50
                    neptune_run_val["test/mAP50-95"] = test_results.box.map
                    neptune_run_val["test/precision"] = test_results.box.mp
                    neptune_run_val["test/recall"] = test_results.box.mr
                except Exception as e:
                    print(f"Warning: Error logging test results to Neptune: {e}")
    
    print("\nValidation complete!")
    
    # Stop Neptune run if it was initialized
    if neptune_run_val is not None:
        try:
            neptune_run_val.stop()
            print("Neptune validation run stopped successfully.")
        except Exception as e:
            print(f"Warning: Error stopping Neptune run: {e}")


# %%
# Plot training and validation loss and accuracy over epochs
# Run this cell after training completes to visualize training progress

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for TACC/SSH compatibility
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Plotting - only if we have data from training
if len(train_losses) > 0 and len(test_losses) > 0:
    try:
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss', marker='o')
        plt.plot(epochs, test_losses, label='Test Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
        plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        results_dir = Project_Path / 'runs' / 'train' / 'exdark_yolov5'
        results_dir.mkdir(parents=True, exist_ok=True)
        plot_path = results_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nTraining curves saved to '{plot_path.absolute()}'")
        
        # Upload plot to Neptune if available
        try:
            token_file = Project_Path / 'neptune_api_token.txt'
            if token_file.exists():
                with open(token_file, 'r') as f:
                    api_token = f.read().strip()
                neptune_run_plot = neptune.init_run(
                    project="ECE381V-Deep-Learning/ECE381V-Term-Project-ViT-YOLO-Hybrid-Model",
                    api_token=api_token,
                )
                neptune_run_plot["plots/training_curves"].upload(str(plot_path))
                neptune_run_plot.stop()
                print("Training curves uploaded to Neptune.ai")
        except Exception as e:
            print(f"Warning: Could not upload plot to Neptune: {e}")
        
        plt.close()  # Close the figure to free memory
        
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Warning: No training data collected, skipping plot generation")
    print("Please run the training cell first.")



