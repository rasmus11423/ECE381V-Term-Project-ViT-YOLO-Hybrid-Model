#!/usr/bin/env python3
"""
Fine-tune YOLOS (ViT detector) on ExDark using existing YOLO-format dataset.

Requirements:
- ExDark already downloaded and converted as in your YOLOv5 script:
  - Data/ExDark/<class_name>/*.jpg
  - Output/images/{train,val,test}
  - Output/labels/{train,val,test} with YOLO txt labels
- neptune_api_token.txt in project root (optional, for Neptune logging)

Model:
- Pretrained ViT detector: hustvl/yolos-small (YOLOS)
"""

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
        # Check if we're in a virtual environment (TACC venv)
        _in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        if _in_venv:
            print(f"Package {package} not found. Installing in virtual environment...")
        else:
            print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {package}: {e}")
            print("If running on TACC, ensure packages are pre-installed in venv using setup_yolos_venv.sh")
            raise

# Install required packages
required_packages = [
    ("transformers>=4.42.0", "transformers"),  # Added for YOLOS
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
# Use current working directory as project root
# This works for both local development and TACC SSH execution
import os
notebook_dir = Path(os.getcwd())  # Current working directory
# On TACC, this will be the directory where the script is executed from
# Make sure to run from the project root directory

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

print("="*60)
print("Environment Check:")
print("="*60)
print("Project root:", Project_Path.absolute())
print("Data dir:", Data_Path.absolute())
print("GroundTruth dir:", GroundTruth_Path.absolute())
print("Working directory:", os.getcwd())
print("Python executable:", sys.executable)
# Check if running in venv
in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
print("Virtual environment:", "Yes" if in_venv else "No")
if in_venv:
    print("Venv path:", sys.prefix)
print("="*60)

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

# Filter out 'images' and 'labels' folders (they shouldn't be classes)
exclude_folders = {'images', 'labels'}

for p in sorted(Images_Path.iterdir()):
    if p.is_dir() and p.name not in exclude_folders:
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

# ----------------------------------------------------------------------
# YOLOS-specific imports (after dataset preparation)
# ----------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    YolosForObjectDetection,
)

import neptune

# Determine device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Using device: {device}")
if device == 'cuda':
    print(f"Found {num_gpus} GPU(s)")

# Get class mappings for YOLOS
num_classes = len(Classes)
id2label = {i: c for i, c in enumerate(Classes)}
label2id = {c: i for i, c in enumerate(Classes)}

print("\n" + "=" * 60)
print("YOLOS Training Setup")
print("=" * 60)
print("Detected classes:", Classes)
print("Number of classes:", num_classes)
print("=" * 60)

# ----------------------------------------------------------------------
# 4. Dataset: read YOLO labels and convert to COCO-style for YOLOS
# ----------------------------------------------------------------------
class ExDarkYoloDetectionDataset(Dataset):
    """
    Uses the YOLO-style labels produced by your previous script:
    Each label file line: <class_id> <cx> <cy> <w> <h>  (all normalized 0..1)
    """

    def __init__(self, images_dir: Path, labels_dir: Path):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.samples = []

        for img_path in sorted(self.images_dir.glob("*.jpg")):
            lbl_path = self.labels_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                self.samples.append((img_path, lbl_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.images_dir} with labels in {self.labels_dir}")

        print(f"Loaded {len(self.samples)} samples from {self.images_dir.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        # Load image
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Parse YOLO txt
        annotations = []
        with open(lbl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])

                # Convert from normalized cx,cy,w,h -> COCO x,y,w,h (absolute)
                bw = w * W
                bh = h * H
                x = (cx * W) - bw / 2.0
                y = (cy * H) - bh / 2.0

                # Clamp to image boundaries
                x = max(0.0, min(float(W - 1), x))
                y = max(0.0, min(float(H - 1), y))
                bw = max(1.0, min(float(W - x), bw))
                bh = max(1.0, min(float(H - y), bh))

                annotations.append(
                    {
                        "category_id": cls_id,
                        "bbox": [x, y, bw, bh],  # COCO format: [x, y, width, height] in absolute pixels
                        "area": bw * bh,
                        "iscrowd": 0,
                    }
                )

        # YOLOS image processor expects COCO-style target dict
        # It will convert this to the format it needs (class_labels, boxes)
        target = {
            "image_id": idx,
            "annotations": annotations,
        }

        return {"image": img, "target": target}


# ----------------------------------------------------------------------
# 5. Image processor & collate function (uses YOLOS preprocessor)
# ----------------------------------------------------------------------
MODEL_NAME = "hustvl/yolos-small"

print(f"Loading image processor and config from '{MODEL_NAME}'...")
sys.stdout.flush()
try:
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    print("✓ Image processor loaded successfully")
    sys.stdout.flush()
except Exception as e:
    print(f"✗ Error loading image processor: {e}")
    print("This might be due to network issues or Hugging Face Hub access.")
    import traceback
    traceback.print_exc()
    raise

def collate_fn(batch):
    """
    Collate function that processes images and annotations using YOLOS image processor.
    The image processor converts COCO-style annotations to the format YOLOS expects:
    - Input: COCO-style dict with 'annotations' list containing 'category_id' and 'bbox'
    - Output: 'labels' list with dicts containing 'class_labels' and 'boxes' tensors
    """
    images = [b["image"] for b in batch]
    targets = [b["target"] for b in batch]

    encoding = image_processor(
        images=images,
        annotations=targets,
        return_tensors="pt",
    )
    # encoding contains:
    # - pixel_values: tensor of shape (batch_size, 3, height, width)
    # - labels: list of dicts, each with 'class_labels' and 'boxes' tensors
    return encoding

# ----------------------------------------------------------------------
# 6. Build datasets & dataloaders
# ----------------------------------------------------------------------
train_dataset = ExDarkYoloDetectionDataset(
    Output / "images" / "train",
    Output / "labels" / "train",
)
val_dataset = ExDarkYoloDetectionDataset(
    Output / "images" / "val",
    Output / "labels" / "val",
)

# ----------------------------------------------------------------------
# 6.5. Set batch size
# ----------------------------------------------------------------------
BATCH_SIZE = 64
print(f"Batch size set to {BATCH_SIZE}.")
if device == "cuda" and num_gpus > 1:
    print(f"   With {num_gpus} GPUs, effective batch size will be {BATCH_SIZE * num_gpus}.")

# Set num_workers for data loading
# Use 0 workers on SLURM to avoid multiprocessing hangs (single-threaded is safer)
# For local development, you can use multiple workers
if os.environ.get('SLURM_JOB_ID') is not None:
    NUM_WORKERS = 0  # Single-threaded data loading - safest for SLURM
    print("  Using single-threaded data loading (NUM_WORKERS=0) for SLURM compatibility")
else:
    NUM_WORKERS = min(8, os.cpu_count() or 8)

# Create DataLoaders with explicit error handling
print(f"\nCreating DataLoaders...")
print(f"  Train dataset size: {len(train_dataset)}")
print(f"  Val dataset size: {len(val_dataset)}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Number of workers: {NUM_WORKERS}")

print("Creating train DataLoader...")
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True if device == "cuda" else False,  # Only pin memory for GPU
)
print("✓ Train DataLoader created")

print("Creating val DataLoader...")
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True if device == "cuda" else False,  # Only pin memory for GPU
)
print("✓ Val DataLoader created")

# ----------------------------------------------------------------------
# 7. Model loading and setup
# ----------------------------------------------------------------------

print(f"Loading YOLOS model '{MODEL_NAME}' with {num_classes} classes...")
model = YolosForObjectDetection.from_pretrained(MODEL_NAME)

# Now update the config and resize the classification head
config = model.config
config.num_labels = num_classes
config.id2label = id2label
config.label2id = label2id

# Resize the classification head to match our number of classes
# YOLOS uses a linear layer for classification: model.class_labels_classifier
# The number of classes is num_labels + 1 (for background/no-object class)
old_num_labels = model.config.num_labels
new_num_labels = num_classes
new_out_features = new_num_labels + 1  # +1 for background class - define outside if block

if old_num_labels != new_num_labels:
    print(f"Resizing classification head from {old_num_labels} to {new_num_labels} classes...")
    # Get the classifier layer
    classifier = model.class_labels_classifier
    # Get input features from the classifier
    in_features = classifier.weight.shape[1]
    old_out_features = classifier.weight.shape[0]  # Should be old_num_labels + 1
    
    # Create new classifier with correct size
    new_classifier = nn.Linear(in_features, new_out_features)
    
    # Initialize new classifier weights (copy from old if possible, otherwise random)
    with torch.no_grad():
        # Copy overlapping weights
        copy_size = min(old_out_features, new_out_features)
        new_classifier.weight[:copy_size] = classifier.weight[:copy_size]
        new_classifier.bias[:copy_size] = classifier.bias[:copy_size]
        
        # Initialize remaining weights with small random values if we're expanding
        if new_out_features > old_out_features:
            nn.init.normal_(new_classifier.weight[copy_size:], mean=0.0, std=0.02)
            nn.init.zeros_(new_classifier.bias[copy_size:])
    
    # Replace the classifier
    model.class_labels_classifier = new_classifier
    print(f"Classification head resized successfully from {old_out_features} to {new_out_features} outputs.")
    
# Update model config
model.config = config

# Fix empty_weight issue: patch cross_entropy to handle size mismatch
# The loss function has empty_weight with wrong size (COCO's 92 vs our 13 classes)
import torch.nn.functional as F
original_cross_entropy = F.cross_entropy

def patched_cross_entropy(input, target, weight=None, *args, **kwargs):
    if weight is not None and isinstance(weight, torch.Tensor):
        if len(input.shape) >= 2 and weight.shape[0] != input.shape[1]:
            weight = None  # Ignore weight if size mismatch
    return original_cross_entropy(input, target, weight=weight, *args, **kwargs)

F.cross_entropy = patched_cross_entropy
torch.nn.functional.cross_entropy = patched_cross_entropy

# Move model to device
model.to(device)

# Enable multi-GPU training if multiple GPUs are available
if device == "cuda" and num_gpus > 1:
    print(f"\nEnabling multi-GPU training with {num_gpus} GPUs using DataParallel...")
    model = nn.DataParallel(model)
    print("✓ Model wrapped with DataParallel for multi-GPU training")

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 3

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)


# ----------------------------------------------------------------------
# 8. Neptune logging (optional)
# ----------------------------------------------------------------------
neptune_run = None
token_file = Project_Path / "neptune_api_token.txt"
if token_file.exists():
    try:
        with open(token_file, "r") as f:
            api_token = f.read().strip()
        # Try to initialize with async mode for better performance
        try:
            neptune_run = neptune.init_run(
                project="ECE381V-Deep-Learning/ECE381V-Term-Project-ViT-YOLO-Hybrid-Model",
                api_token=api_token,
                mode="async",  # Use async mode for better performance and real-time updates
            )
        except TypeError:
            # If mode parameter is not supported, initialize without it
            neptune_run = neptune.init_run(
                project="ECE381V-Deep-Learning/ECE381V-Term-Project-ViT-YOLO-Hybrid-Model",
                api_token=api_token,
            )
        neptune_run["parameters"] = {
            "model_name": MODEL_NAME,
            "num_classes": num_classes,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "device": device,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dataset": "ExDark",
        }
        if device == "cuda" and num_gpus > 1:
            neptune_run["parameters/num_gpus"] = num_gpus
            neptune_run["parameters/effective_batch_size"] = BATCH_SIZE * num_gpus
        neptune_run["parameters/classes"] = ", ".join(Classes)
        print("Neptune monitoring initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize Neptune monitoring: {e}")
        print("Continuing without Neptune monitoring...")

# ----------------------------------------------------------------------
# 9. Training + validation loop
# ----------------------------------------------------------------------
# Disable tqdm progress bars for cleaner output (consistent with YOLOv5 script)
os.environ['TQDM_DISABLE'] = '1'

train_losses = []
val_losses = []

def compute_accuracy(outputs, labels):
    """
    Compute classification accuracy from model outputs.
    For object detection, we compute accuracy based on predicted vs ground truth class labels.
    """
    try:
        # Get predicted class labels from logits
        # outputs.logits shape: [batch_size, num_queries, num_classes + 1]
        logits = outputs.logits
        pred_classes = torch.argmax(logits, dim=-1)  # [batch_size, num_queries]
        
        # Get ground truth class labels
        # labels is a list of dicts, each with 'class_labels' tensor
        total_correct = 0
        total_labels = 0
        
        for i, label_dict in enumerate(labels):
            if 'class_labels' in label_dict:
                gt_classes = label_dict['class_labels']  # [num_objects]
                if len(gt_classes) > 0:
                    # For each ground truth object, find the best matching prediction
                    # Simple approach: check if any prediction matches the ground truth class
                    batch_pred = pred_classes[i]  # [num_queries]
                    for gt_class in gt_classes:
                        total_labels += 1
                        gt_class_val = gt_class.item() if isinstance(gt_class, torch.Tensor) else gt_class
                        # Check if gt_class is in the predicted classes using tensor operations
                        if (batch_pred == gt_class_val).any():
                            total_correct += 1
        
        if total_labels > 0:
            accuracy = total_correct / total_labels
        else:
            accuracy = 0.0
        return accuracy
    except Exception as e:
        # If accuracy computation fails, return 0.0
        return 0.0

def run_epoch(dataloader, training=True, epoch_num=0, neptune_run=None):
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_accuracy = 0.0
    n_batches = 0
    total_batches = len(dataloader)
    
    # Track time for progress reporting
    import time
    start_time = time.time()
    
    # Use tqdm but it will be disabled by environment variable
    loop = tqdm(dataloader, desc="train" if training else "val", leave=False, disable=True)
    for batch_idx, batch in enumerate(loop):
        try:
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch["labels"]]

            if training:
                optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                # Check for NaN before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf detected at batch {batch_idx}, stopping training")
                    return None, None, 0
                
                # Compute accuracy
                accuracy = compute_accuracy(outputs, labels)
                
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            running_loss += loss.item()
            running_accuracy += accuracy
            n_batches += 1
            
            # Compute averages
            avg_loss = running_loss / n_batches
            avg_accuracy = running_accuracy / n_batches
            
            # Print progress every 10 batches or at the end
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx+1}/{total_batches} | Loss: {avg_loss:.4f} | Acc: {avg_accuracy:.4f} | Time: {elapsed:.1f}s", flush=True)
            
            # Calculate ETA
            elapsed = time.time() - start_time
            avg_time_per_batch = elapsed / n_batches
            remaining_batches = total_batches - n_batches
            eta_seconds = avg_time_per_batch * remaining_batches
            eta_hours = eta_seconds / 3600
            
            
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            raise

    epoch_loss = running_loss / max(1, n_batches)
    epoch_accuracy = running_accuracy / max(1, n_batches)
    total_time = time.time() - start_time
    return epoch_loss, epoch_accuracy, total_time

# Wrap main execution in __main__ guard for Windows multiprocessing compatibility
if __name__ == '__main__':
    import sys
    sys.stdout.flush()  # Ensure output is flushed immediately
    
    print("=" * 60)
    print("Starting YOLOS training on ExDark...")
    print("=" * 60)
    sys.stdout.flush()

    print("\n" + "=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Device: {device}")
    if device == "cuda" and num_gpus > 1:
        print(f"  Number of GPUs: {num_gpus}")
        print(f"  Effective batch size (total): {BATCH_SIZE * num_gpus}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  DataLoader workers: {NUM_WORKERS}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print("=" * 60)

    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        sys.stdout.flush()

        train_loss, train_accuracy, train_time = run_epoch(train_loader, training=True, epoch_num=epoch, neptune_run=neptune_run)
        
        # Check if training returned None (NaN detected)
        if train_loss is None or train_accuracy is None:
            print(f"Training stopped due to NaN/Inf at epoch {epoch}")
            break
        
        val_loss, val_accuracy, val_time = run_epoch(val_loader, training=False, epoch_num=epoch, neptune_run=neptune_run)

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"NaN detected at epoch {epoch}! Training stopped.")
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | "
              f"LR: {current_lr:.6f}")

        if neptune_run is not None:
            try:
                neptune_run["train/loss"].append(train_loss)
                neptune_run["train/accuracy"].append(train_accuracy)
                neptune_run["val/loss"].append(val_loss)
                neptune_run["val/accuracy"].append(val_accuracy)
                neptune_run["train/lr"].append(optimizer.param_groups[0]["lr"])
                neptune_run["epoch"].append(epoch)
                # Sync to ensure data is sent to Neptune
                neptune_run.sync()
            except Exception as e:
                print(f"Warning: Neptune logging error: {e}")

    print("\nTraining complete!")

    # ----------------------------------------------------------------------
    # 10. Save model + processor
    # ----------------------------------------------------------------------
    save_dir = Project_Path / "runs" / "detectors" / "yolos_exdark"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving model and image processor to {save_dir} ...")
    # If using DataParallel, unwrap the model before saving
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_dir)
    image_processor.save_pretrained(save_dir)

    # ----------------------------------------------------------------------
    # 11. Plot training curves and (optionally) upload to Neptune
    # ----------------------------------------------------------------------
    plot_path_loss = save_dir / "training_curves_yolos_exdark_loss.png"
    plot_path_acc = save_dir / "training_curves_yolos_exdark_accuracy.png"

    epochs = np.arange(1, len(train_losses) + 1)
    
    # Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train loss")
    plt.plot(epochs, val_losses, marker="s", label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("YOLOS (ViT) on ExDark - Training/Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path_loss, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot accuracy curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracies, marker="o", label="Train accuracy")
    plt.plot(epochs, val_accuracies, marker="s", label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("YOLOS (ViT) on ExDark - Training/Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path_acc, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Training curves saved to: {plot_path_loss} and {plot_path_acc}")

    if neptune_run is not None:
        try:
            neptune_run["plots/loss_curves"].upload(str(plot_path_loss))
            neptune_run["plots/accuracy_curves"].upload(str(plot_path_acc))
            neptune_run["final/train_loss"] = train_losses[-1]
            neptune_run["final/val_loss"] = val_losses[-1]
            neptune_run["final/train_accuracy"] = train_accuracies[-1]
            neptune_run["final/val_accuracy"] = val_accuracies[-1]
            neptune_run.stop()
            print("Neptune run closed.")
        except Exception as e:
            print(f"Warning: could not upload plot/close Neptune: {e}")
