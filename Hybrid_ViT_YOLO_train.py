#!/usr/bin/env python3
"""
Hybrid ViT-YOLO Model Training Script
Combines YOLOS (ViT-based detector) and YOLOv5 architectures for improved performance.

Architecture:
- Uses YOLOS ViT backbone for global feature extraction
- Adds YOLOv5-style detection head for anchor-based detection
- Fuses features from both architectures

Requirements:
- ExDark dataset (same as YOLOv5 and YOLOS scripts)
- neptune_api_token.txt in project root (optional, for Neptune logging)
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
    ("transformers>=4.42.0", "transformers"),  # For YOLOS
    ("ultralytics>=8.0.0", "ultralytics"),  # For YOLOv5
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
# Hybrid Model Architecture: ViT-YOLO
# ----------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optimize PyTorch memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from transformers import (
    AutoImageProcessor,
    YolosModel,  # Use base model without detection head
    YolosConfig,
)
from ultralytics import YOLO

import neptune

# Determine device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Using device: {device}")
if device == 'cuda':
    print(f"Found {num_gpus} GPU(s)")
    torch.cuda.empty_cache()

# Get class mappings
num_classes = len(Classes)
id2label = {i: c for i, c in enumerate(Classes)}
label2id = {c: i for i, c in enumerate(Classes)}

print("\n" + "=" * 60)
print("Hybrid ViT-YOLO Training Setup")
print("=" * 60)
print("Detected classes:", Classes)
print("Number of classes:", num_classes)
print("=" * 60)

# ----------------------------------------------------------------------
# Hybrid Model Definition
# ----------------------------------------------------------------------
class HybridViTYOLO(nn.Module):
    """
    Hybrid model combining:
    - YOLOS ViT backbone for global feature extraction
    - YOLOv5-style detection head for anchor-based detection
    """
    
    def __init__(self, num_classes, vit_model_name="hustvl/yolos-small", img_size=640):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Load YOLOS ViT backbone (without detection head)
        print(f"Loading ViT backbone from '{vit_model_name}'...")
        self.vit_backbone = YolosModel.from_pretrained(vit_model_name)
        vit_config = self.vit_backbone.config
        
        # Get ViT feature dimension
        self.vit_hidden_size = vit_config.hidden_size
        
        # Image processor for ViT
        self.image_processor = AutoImageProcessor.from_pretrained(vit_model_name)
        
        # Feature projection: Convert ViT sequence output to feature maps
        # ViT outputs [batch, num_patches+1, hidden_size]
        # We need to reshape to feature maps for YOLO head
        # For 640x640 input, ViT typically has ~400 patches (20x20 grid)
        self.patch_size = vit_config.patch_size  # Usually 16
        self.num_patches_per_side = img_size // self.patch_size  # 640 // 16 = 40
        
        # Project ViT features to feature map dimensions using 1x1 convs
        # Use multiple scales like YOLOv5 (P3, P4, P5)
        self.proj_convs = nn.ModuleList([
            nn.Conv2d(self.vit_hidden_size, 256, 1),
            nn.Conv2d(self.vit_hidden_size, 512, 1),
            nn.Conv2d(self.vit_hidden_size, 1024, 1),
        ])
        
        # Reshape modules to refine spatial feature maps
        self.reshape_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.GELU(),
            ),
        ])
        
        # YOLOv5-style detection heads (3 scales: P3, P4, P5)
        # Each head predicts: [x, y, w, h, obj, cls1, cls2, ...]
        # All features now have 256 channels after FPN/PAN
        num_anchors = 3  # 3 anchors per scale
        self.detection_heads = nn.ModuleList([
            # P3 (large objects) - 80x80
            nn.Conv2d(256, num_anchors * (5 + num_classes), 1),
            # P4 (medium objects) - 40x40
            nn.Conv2d(256, num_anchors * (5 + num_classes), 1),
            # P5 (small objects) - 20x20
            nn.Conv2d(256, num_anchors * (5 + num_classes), 1),
        ])
        
        # Feature pyramid network (FPN) for multi-scale feature fusion
        # Lateral connections: project bottom-up features to common dimension
        self.fpn_lateral = nn.ModuleList([
            nn.Conv2d(256, 256, 1),  # P3 -> 256
            nn.Conv2d(512, 256, 1),  # P4 -> 256
            nn.Conv2d(1024, 256, 1),  # P5 -> 256
        ])
        
        # Top-down pathway: refine after fusion
        self.fpn_topdown = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1),  # P3
            nn.Conv2d(256, 256, 3, padding=1),  # P4
            nn.Conv2d(256, 256, 3, padding=1),  # P5
        ])
        
        # Path aggregation network (PAN) for bottom-up feature fusion
        self.pan = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1),  # P3
            nn.Conv2d(256, 256, 3, padding=1),  # P4
            nn.Conv2d(256, 256, 3, padding=1),  # P5
        ])
        
        # Upsampling layers for FPN
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, pixel_values):
        """
        Forward pass through hybrid model.
        
        Args:
            pixel_values: Preprocessed images [batch, 3, H, W]
        
        Returns:
            List of detection outputs at 3 scales
        """
        batch_size = pixel_values.shape[0]
        
        # Extract features using ViT backbone
        vit_outputs = self.vit_backbone(pixel_values=pixel_values)
        # vit_outputs.last_hidden_state: [batch, num_patches+1, hidden_size]
        vit_features = vit_outputs.last_hidden_state
        
        # Remove CLS token (first token) and keep only patch tokens
        patch_features = vit_features[:, 1:, :]  # [batch, num_patches, hidden_size]
        
        # Reshape patch features to spatial grid
        # ViT patches are arranged in a grid: for 640x640 with patch_size=16, we get 40x40 patches
        num_patches = patch_features.shape[1]  # Should be ~1600 for 640x640 (40*40)
        grid_size = int(np.sqrt(num_patches))  # Should be 40 for 640x640
        
        # Ensure we have a perfect square
        if grid_size * grid_size != num_patches:
            # If not perfect square, pad or crop
            target_patches = grid_size * grid_size
            if num_patches > target_patches:
                patch_features = patch_features[:, :target_patches, :]
            else:
                # Pad with zeros
                padding = torch.zeros(batch_size, target_patches - num_patches, self.vit_hidden_size, 
                                    device=patch_features.device, dtype=patch_features.dtype)
                patch_features = torch.cat([patch_features, padding], dim=1)
        
        # Reshape to [batch, grid_size, grid_size, hidden_size]
        patch_features = patch_features.view(batch_size, grid_size, grid_size, self.vit_hidden_size)
        # Permute to [batch, hidden_size, grid_size, grid_size]
        patch_features = patch_features.permute(0, 3, 1, 2)
        
        # Project to different scales and create feature maps
        features = []
        for i in range(len(self.proj_convs)):
            # Project features using 1x1 conv (more efficient than linear + reshape)
            # Apply 1x1 conv directly on spatial features
            proj_output = self.proj_convs[i](patch_features)
            
            # Apply conv to refine features (includes BN and activation)
            proj_output = self.reshape_modules[i](proj_output)
            
            # Upsample/downsample to target sizes
            if i == 0:  # P3 - largest (80x80)
                target_size = 80
            elif i == 1:  # P4 - medium (40x40)
                target_size = 40
            else:  # P5 - smallest (20x20)
                target_size = 20
            
            if proj_output.shape[-1] != target_size:
                proj_output = F.interpolate(proj_output, size=(target_size, target_size), mode='bilinear', align_corners=False)
            
            features.append(proj_output)
        
        # Apply FPN (top-down): all features now have 256 channels after lateral connections
        # Step 1: Apply lateral connections to normalize channels
        lateral_features = []
        for i in range(len(features)):
            lateral_features.append(self.fpn_lateral[i](features[i]))
        
        # Step 2: Top-down pathway with upsampling
        fpn_features = []
        for i in range(len(lateral_features) - 1, -1, -1):
            if i == len(lateral_features) - 1:
                # Start from top (P5)
                fpn_feat = lateral_features[i]
            else:
                # Upsample higher-level feature and add to current level
                upsampled = self.upsample(fpn_features[0])
                # Add lateral feature and upsampled feature
                fpn_feat = lateral_features[i] + upsampled
            # Apply top-down refinement
            fpn_feat = self.fpn_topdown[i](fpn_feat)
            fpn_features.insert(0, fpn_feat)
        
        # Apply PAN (bottom-up): all features are now 256 channels
        pan_features = []
        for i in range(len(fpn_features)):
            if i == 0:
                pan_feat = self.pan[i](fpn_features[i])
            else:
                # Downsample previous feature and add
                downsampled = F.avg_pool2d(pan_features[-1], kernel_size=2, stride=2)
                pan_feat = self.pan[i](fpn_features[i]) + downsampled
            pan_features.append(pan_feat)
        
        # Detection heads - need to project to different channel sizes
        # Update detection heads to accept 256 channels and project internally
        outputs = []
        for i, head in enumerate(self.detection_heads):
            out = head(pan_features[i])
            outputs.append(out)
        
        return outputs

# ----------------------------------------------------------------------
# Dataset: YOLO format labels
# ----------------------------------------------------------------------
class ExDarkYoloDetectionDataset(Dataset):
    """Dataset for YOLO-format labels."""

    def __init__(self, images_dir: Path, labels_dir: Path, image_processor, img_size=640):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_processor = image_processor
        self.img_size = img_size
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
        
        # Parse YOLO labels
        targets = []
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
                targets.append([cls_id, cx, cy, w, h])
        
        # Process image with ViT processor
        processed = self.image_processor(img, return_tensors="pt", size={"height": self.img_size, "width": self.img_size})
        pixel_values = processed["pixel_values"].squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "image": img,
            "targets": targets,
            "image_id": idx,
        }

def collate_fn(batch):
    """Collate function for batching."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    images = [b["image"] for b in batch]
    targets = [b["targets"] for b in batch]
    image_ids = [b["image_id"] for b in batch]
    
    return {
        "pixel_values": pixel_values,
        "images": images,
        "targets": targets,
        "image_ids": image_ids,
    }

# ----------------------------------------------------------------------
# Loss Function: YOLOv5-style loss
# ----------------------------------------------------------------------
class YOLOLoss(nn.Module):
    """YOLOv5-style loss function for object detection."""
    
    def __init__(self, num_classes, img_size=640):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Anchor boxes (YOLOv5 default anchors for 3 scales)
        # Format: [w, h] in absolute pixels - will be normalized when used
        # Original anchors are for 640x640 images
        anchors_abs = [
            [[10, 13], [16, 30], [33, 23]],  # P3
            [[30, 61], [62, 45], [59, 119]],  # P4
            [[116, 90], [156, 198], [373, 326]],  # P5
        ]
        # Normalize anchors to [0, 1] range by dividing by image size
        self.anchors = [[[a[0]/img_size, a[1]/img_size] for a in scale_anchors] 
                        for scale_anchors in anchors_abs]
        
        # Grid sizes for each scale
        self.grid_sizes = [80, 40, 20]  # P3, P4, P5
        
    def forward(self, predictions, targets):
        """
        Compute YOLO loss.
        
        Args:
            predictions: List of 3 tensors [P3, P4, P5], each [batch, anchors*(5+num_classes), H, W]
            targets: List of target annotations per image
        """
        device = predictions[0].device
        total_loss = torch.tensor(0.0, device=device)
        box_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        
        batch_size = predictions[0].shape[0]
        
        # Process each scale
        for scale_idx, pred in enumerate(predictions):
            B, C, H, W = pred.shape
            num_anchors = len(self.anchors[scale_idx])
            
            # Reshape prediction: [B, anchors*(5+num_classes), H, W] -> [B, anchors, 5+num_classes, H, W]
            pred = pred.view(B, num_anchors, 5 + self.num_classes, H, W)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # [B, anchors, H, W, 5+num_classes]
            
            # Split predictions
            pred_xy = torch.sigmoid(pred[..., 0:2])  # x, y (0-1 offset within grid cell)
            pred_wh_raw = pred[..., 2:4]  # w, h raw predictions (will be converted to log-space)
            pred_obj = pred[..., 4:5]  # objectness
            pred_cls = pred[..., 5:]  # class logits
            
            # Build target tensors
            target_obj = torch.zeros(B, num_anchors, H, W, device=device)
            target_cls = torch.zeros(B, num_anchors, H, W, self.num_classes, device=device)
            target_xy = torch.zeros(B, num_anchors, H, W, 2, device=device)
            target_wh = torch.zeros(B, num_anchors, H, W, 2, device=device)
            
            # Match targets to anchors and grids
            # CRITICAL FIX: Only assign targets to the BEST scale for each object
            # Small objects -> P3 (80x80), Medium -> P4 (40x40), Large -> P5 (20x20)
            for b in range(batch_size):
                img_targets = targets[b]
                if len(img_targets) == 0:
                    continue
                
                for cls_id, cx, cy, w, h in img_targets:
                    # Determine best scale based on object size
                    # Object area in normalized coordinates
                    obj_area = w * h
                    
                    # Assign to scale based on size:
                    # Small objects (area < 0.01) -> P3 (80x80, finest grid)
                    # Medium objects (0.01 <= area < 0.1) -> P4 (40x40)
                    # Large objects (area >= 0.1) -> P5 (20x20, coarsest grid)
                    if obj_area < 0.01:
                        best_scale = 0  # P3
                    elif obj_area < 0.1:
                        best_scale = 1  # P4
                    else:
                        best_scale = 2  # P5
                    
                    # Only assign to this scale if it matches current scale_idx
                    if scale_idx != best_scale:
                        continue
                    
                    # Find best anchor using IoU-like metric (consider both size and aspect ratio)
                    best_score = -1
                    best_anchor = 0
                    for a_idx, anchor in enumerate(self.anchors[scale_idx]):
                        anchor_w, anchor_h = anchor
                        # Compute IoU-like score: consider both size match and aspect ratio
                        # Size match: how close are the dimensions?
                        size_score = 1.0 / (1.0 + abs(w - anchor_w) + abs(h - anchor_h))
                        # Aspect ratio match
                        obj_ratio = w / (h + 1e-6)
                        anchor_ratio = anchor_w / (anchor_h + 1e-6)
                        aspect_score = 1.0 / (1.0 + abs(obj_ratio - anchor_ratio))
                        # Combined score
                        score = size_score * aspect_score
                        if score > best_score:
                            best_score = score
                            best_anchor = a_idx
                    
                    # Find grid cell
                    grid_x = int(cx * W)
                    grid_y = int(cy * H)
                    grid_x = max(0, min(W - 1, grid_x))
                    grid_y = max(0, min(H - 1, grid_y))
                    
                    # Set targets
                    target_obj[b, best_anchor, grid_y, grid_x] = 1.0
                    target_cls[b, best_anchor, grid_y, grid_x, int(cls_id)] = 1.0
                    # target_xy: offset within grid cell (0-1)
                    target_xy[b, best_anchor, grid_y, grid_x, 0] = cx * W - grid_x
                    target_xy[b, best_anchor, grid_y, grid_x, 1] = cy * H - grid_y
                    # target_wh: width/height in log-space relative to anchor (YOLOv5 format)
                    # Formula: target_wh = log(target_box_wh / anchor_wh)
                    anchor_w, anchor_h = self.anchors[scale_idx][best_anchor]
                    # Convert normalized w,h to absolute pixels, then to log-space relative to anchor
                    target_w_abs = w * self.img_size
                    target_h_abs = h * self.img_size
                    anchor_w_abs = anchor_w * self.img_size
                    anchor_h_abs = anchor_h * self.img_size
                    # Avoid log(0) by adding small epsilon
                    eps = 1e-6
                    ratio_w = target_w_abs / (anchor_w_abs + eps)
                    ratio_h = target_h_abs / (anchor_h_abs + eps)
                    target_wh[b, best_anchor, grid_y, grid_x, 0] = np.log(max(eps, ratio_w))
                    target_wh[b, best_anchor, grid_y, grid_x, 1] = np.log(max(eps, ratio_h))
            
            # Compute losses
            # Box loss (only for positive anchors)
            pos_mask = target_obj > 0.5
            if pos_mask.sum() > 0:
                pred_xy_pos = pred_xy[pos_mask]
                # pred_wh is in log-space, same as target_wh
                pred_wh_pos = pred_wh_raw[pos_mask]
                target_xy_pos = target_xy[pos_mask]
                target_wh_pos = target_wh[pos_mask]
                
                # Use smooth L1 loss for better training stability
                box_loss_xy = F.smooth_l1_loss(pred_xy_pos, target_xy_pos, beta=0.1)
                box_loss_wh = F.smooth_l1_loss(pred_wh_pos, target_wh_pos, beta=0.1)
                box_loss += box_loss_xy + box_loss_wh
            
            # Objectness loss (weighted by positive/negative ratio)
            obj_loss += self.bce_loss(pred_obj.squeeze(-1), target_obj)
            
            # Classification loss (only for positive anchors)
            if pos_mask.sum() > 0:
                pred_cls_pos = pred_cls[pos_mask]
                target_cls_pos = target_cls[pos_mask]
                cls_loss += self.bce_loss(pred_cls_pos, target_cls_pos)
        
        # Combine losses with proper weighting (YOLOv5 style)
        # Adjusted weights to improve detection learning
        # Box loss needs higher weight to learn localization better
        box_weight = 0.1  # Increased from 0.05 to emphasize box regression
        obj_weight = 1.0  # Keep objectness at 1.0 (critical for detection)
        cls_weight = 0.5  # Classification weight
        total_loss = box_weight * box_loss + obj_weight * obj_loss + cls_weight * cls_loss
        
        return {
            "loss": total_loss,
            "box_loss": box_loss,
            "obj_loss": obj_loss,
            "cls_loss": cls_loss,
        }

# ----------------------------------------------------------------------
# Build datasets & dataloaders
# ----------------------------------------------------------------------
MODEL_NAME = "hustvl/yolos-small"

print(f"Loading image processor from '{MODEL_NAME}'...")
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
print("✓ Image processor loaded successfully")

IMG_SIZE = 640
train_dataset = ExDarkYoloDetectionDataset(
    Output / "images" / "train",
    Output / "labels" / "train",
    image_processor,
    img_size=IMG_SIZE,
)
val_dataset = ExDarkYoloDetectionDataset(
    Output / "images" / "val",
    Output / "labels" / "val",
    image_processor,
    img_size=IMG_SIZE,
)

BATCH_SIZE = 4  # Smaller batch size for hybrid model (memory intensive)
print(f"Batch size set to {BATCH_SIZE}.")
if device == "cuda" and num_gpus > 1:
    print(f"   With {num_gpus} GPUs, effective batch size will be {BATCH_SIZE * num_gpus}.")

if os.environ.get('SLURM_JOB_ID') is not None:
    NUM_WORKERS = 0
    print("  Using single-threaded data loading (NUM_WORKERS=0) for SLURM compatibility")
else:
    NUM_WORKERS = min(4, os.cpu_count() or 4)

print(f"\nCreating DataLoaders...")
print(f"  Train dataset size: {len(train_dataset)}")
print(f"  Val dataset size: {len(val_dataset)}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Number of workers: {NUM_WORKERS}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True if device == "cuda" else False,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True if device == "cuda" else False,
)

# ----------------------------------------------------------------------
# Model initialization
# ----------------------------------------------------------------------
print(f"\nInitializing Hybrid ViT-YOLO model with {num_classes} classes...")
model = HybridViTYOLO(num_classes=num_classes, vit_model_name=MODEL_NAME, img_size=IMG_SIZE)
model.to(device)

# Enable multi-GPU training if available
if device == "cuda" and num_gpus > 1:
    print(f"\nEnabling multi-GPU training with {num_gpus} GPUs using DataParallel...")
    model = nn.DataParallel(model)
    print("✓ Model wrapped with DataParallel for multi-GPU training")

# Loss function
criterion = YOLOLoss(num_classes=num_classes, img_size=IMG_SIZE)

# Optimizer and scheduler
# Learning rate for fine-tuning: conservative for pretrained backbone, higher for new head
LEARNING_RATE = 1e-5  # Base LR: 1e-4 for head, 1e-5 for pretrained backbone
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100  # Extended training for potentially higher accuracy

# Use different learning rates for backbone and head
# Get parameters from the model (works with both regular and DataParallel)
backbone_params = [p for n, p in model.named_parameters() if 'vit_backbone' in n]
head_params = [p for n, p in model.named_parameters() if 'vit_backbone' not in n]

if len(backbone_params) > 0 and len(head_params) > 0:
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},  # Lower LR for pretrained backbone
        {'params': head_params, 'lr': LEARNING_RATE}  # Higher LR for new head
    ], weight_decay=WEIGHT_DECAY)
else:
    # Fallback: use same LR for all parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# ----------------------------------------------------------------------
# Neptune logging
# ----------------------------------------------------------------------
neptune_run = None
token_file = Project_Path / "neptune_api_token.txt"
if token_file.exists():
    try:
        with open(token_file, "r") as f:
            api_token = f.read().strip()
        try:
            neptune_run = neptune.init_run(
                project="ECE381V-Deep-Learning/ECE381V-Term-Project-ViT-YOLO-Hybrid-Model",
                api_token=api_token,
                mode="async",
            )
        except TypeError:
            neptune_run = neptune.init_run(
                project="ECE381V-Deep-Learning/ECE381V-Term-Project-ViT-YOLO-Hybrid-Model",
                api_token=api_token,
            )
        neptune_run["parameters"] = {
            "model_name": "Hybrid ViT-YOLO",
            "vit_backbone": MODEL_NAME,
            "num_classes": num_classes,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "device": device,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dataset": "ExDark",
            "img_size": IMG_SIZE,
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
# Training loop
# ----------------------------------------------------------------------
os.environ['TQDM_DISABLE'] = '1'

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# ----------------------------------------------------------------------
# Accuracy computation functions (simplified mAP50)
# ----------------------------------------------------------------------
def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format."""
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)

def decode_predictions(predictions, anchors, num_classes, img_size=640, conf_threshold=0.1):
    """
    Decode YOLO predictions to bounding boxes.
    
    Returns:
        List of detections per image: [[x1, y1, x2, y2, conf, cls], ...]
    """
    device = predictions[0].device
    batch_size = predictions[0].shape[0]
    all_detections = [[] for _ in range(batch_size)]
    
    for scale_idx, pred in enumerate(predictions):
        B, C, H, W = pred.shape
        num_anchors = len(anchors[scale_idx])
        
        # Reshape: [B, anchors*(5+num_classes), H, W] -> [B, anchors, H, W, 5+num_classes]
        pred = pred.view(B, num_anchors, 5 + num_classes, H, W)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        
        # Decode predictions
        pred_xy = torch.sigmoid(pred[..., 0:2])  # [B, anchors, H, W, 2]
        pred_wh = torch.exp(pred[..., 2:4])  # [B, anchors, H, W, 2] (log-space)
        pred_obj = torch.sigmoid(pred[..., 4:5])  # [B, anchors, H, W, 1]
        pred_cls = torch.softmax(pred[..., 5:], dim=-1)  # [B, anchors, H, W, num_classes]
        
        # Get best class and confidence
        pred_conf = pred_obj.squeeze(-1)  # [B, anchors, H, W]
        pred_cls_conf, pred_cls_id = pred_cls.max(dim=-1)  # [B, anchors, H, W]
        final_conf = pred_conf * pred_cls_conf  # [B, anchors, H, W]
        
        # Filter by confidence threshold
        mask = final_conf > conf_threshold
        
        for b in range(B):
            for a in range(num_anchors):
                for y in range(H):
                    for x in range(W):
                        if mask[b, a, y, x]:
                            # Decode box coordinates
                            cx = (x + pred_xy[b, a, y, x, 0]) / W
                            cy = (y + pred_xy[b, a, y, x, 1]) / H
                            
                            anchor_w, anchor_h = anchors[scale_idx][a]
                            w = pred_wh[b, a, y, x, 0] * anchor_w
                            h = pred_wh[b, a, y, x, 1] * anchor_h
                            
                            # Convert to xyxy format (absolute pixels)
                            x1 = (cx - w/2) * img_size
                            y1 = (cy - h/2) * img_size
                            x2 = (cx + w/2) * img_size
                            y2 = (cy + h/2) * img_size
                            
                            conf = final_conf[b, a, y, x].item()
                            cls_id = pred_cls_id[b, a, y, x].item()
                            
                            all_detections[b].append([x1, y1, x2, y2, conf, cls_id])
    
    return all_detections

def apply_nms(detections, iou_threshold=0.45, max_det=300):
    """Apply Non-Maximum Suppression to detections."""
    if len(detections) == 0:
        return []
    
    # Convert to list of lists for easier manipulation
    if isinstance(detections, torch.Tensor):
        detections = detections.tolist()
    
    if len(detections) == 0:
        return []
    
    # Sort by confidence (index 4 is confidence)
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    detections = detections[:max_det]
    
    # Simple NMS - greedy selection
    keep = []
    while len(detections) > 0:
        # Keep highest confidence box
        keep.append(detections[0])
        if len(detections) == 1:
            break
        
        # Remove boxes with high IoU
        box0 = detections[0][:4]
        remaining = []
        for det in detections[1:]:
            iou = compute_iou(box0, det[:4])
            if iou < iou_threshold:
                remaining.append(det)
        detections = remaining
    
    return keep

def compute_simple_accuracy(predictions_list, targets_list, anchors, num_classes, img_size=640, iou_threshold=0.5):
    """
    Compute simplified accuracy metric: F1 score at IoU threshold.
    Only computes during validation to avoid slowing training.
    
    Args:
        predictions_list: List of batches, each batch is list of [P3, P4, P5] predictions
        targets_list: Flattened list of targets, one per image
    """
    try:
        total_correct = 0
        total_predictions = 0
        total_targets = 0
        
        img_idx = 0  # Track image index across batches
        
        # Process each batch
        for batch_idx, predictions in enumerate(predictions_list):
            # Get batch size from first prediction
            batch_size = predictions[0].shape[0]
            
            # Decode predictions for this batch (lower threshold for early training)
            all_detections = decode_predictions(predictions, anchors, num_classes, img_size, conf_threshold=0.1)
            
            # Process each image in batch
            for b in range(batch_size):
                if img_idx >= len(targets_list):
                    break
                    
                detections = all_detections[b]
                
                # Apply NMS
                detections = apply_nms(detections, iou_threshold=0.45)
                
                # Get ground truth boxes for this image
                gt_boxes = []
                for cls_id, cx, cy, w, h in targets_list[img_idx]:
                    # Convert normalized cx,cy,w,h to xyxy (absolute pixels)
                    x1 = (cx - w/2) * img_size
                    y1 = (cy - h/2) * img_size
                    x2 = (cx + w/2) * img_size
                    y2 = (cy + h/2) * img_size
                    gt_boxes.append([x1, y1, x2, y2, int(cls_id)])
                
                total_targets += len(gt_boxes)
                total_predictions += len(detections)
                
                # Match predictions to ground truth
                matched_gt = set()
                for det in detections:
                    if len(det) < 6:
                        continue
                    x1, y1, x2, y2, conf, pred_cls = det[:6]
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt in enumerate(gt_boxes):
                        if gt_idx in matched_gt:
                            continue
                        if int(gt[4]) != int(pred_cls):
                            continue
                        
                        iou = compute_iou([x1, y1, x2, y2], gt[:4])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= iou_threshold:
                        total_correct += 1
                        if best_gt_idx >= 0:
                            matched_gt.add(best_gt_idx)
                
                img_idx += 1
        
        # Compute precision (correct predictions / total predictions)
        if total_predictions > 0:
            precision = total_correct / total_predictions
        else:
            precision = 0.0
        
        # Compute recall (correct predictions / total targets)
        if total_targets > 0:
            recall = total_correct / total_targets
        else:
            recall = 0.0
        
        # F1 score as accuracy metric
        if precision + recall > 0:
            accuracy = 2 * (precision * recall) / (precision + recall)
        else:
            accuracy = 0.0
        
        return accuracy
    except Exception as e:
        # If computation fails, return 0
        import traceback
        traceback.print_exc()
        return 0.0

def compute_map50_simple(predictions, targets, num_classes):
    """Compute simplified accuracy metric based on loss."""
    # For a more accurate metric, we'd need to:
    # 1. Apply NMS to predictions
    # 2. Match predictions to ground truth using IoU
    # 3. Compute mAP50
    # For now, use loss-based proxy metric
    try:
        # Count total targets
        total_targets = sum(len(t) for t in targets)
        if total_targets == 0:
            return 0.0
        
        # Simple metric: use inverse of normalized loss as accuracy proxy
        # This gives us a rough estimate that correlates with actual performance
        return 0.0  # Will be computed from loss in run_epoch
    except:
        return 0.0

def run_epoch(dataloader, training=True, epoch_num=0):
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_box_loss = 0.0
    running_obj_loss = 0.0
    running_cls_loss = 0.0
    n_batches = 0
    total_batches = len(dataloader)
    
    # Store predictions and targets for accuracy computation
    # Sample every Nth batch to avoid excessive computation (both train and val)
    all_predictions = []
    all_targets = []
    sample_interval = max(1, total_batches // 5)  # Sample ~5 batches for accuracy
    
    import time
    start_time = time.time()
    
    loop = tqdm(dataloader, desc="train" if training else "val", leave=False, disable=True)
    for batch_idx, batch in enumerate(loop):
        try:
            pixel_values = batch["pixel_values"].to(device)
            targets = batch["targets"]

            if training:
                optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                predictions = model(pixel_values)
                loss_dict = criterion(predictions, targets)
                loss = loss_dict["loss"]
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf detected at batch {batch_idx}, stopping training")
                    return None, None, 0
                
                # Store predictions for accuracy computation (sample batches for both train and val)
                if batch_idx % sample_interval == 0:  # Sample batches
                    all_predictions.append([p.detach() for p in predictions])  # Keep on device for now
                    all_targets.extend(targets)  # Flatten targets list
                
                if training:
                    loss.backward()
                    # More aggressive gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()

            running_loss += loss.item()
            running_box_loss += loss_dict["box_loss"].item()
            running_obj_loss += loss_dict["obj_loss"].item()
            running_cls_loss += loss_dict["cls_loss"].item()
            n_batches += 1
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - start_time
                avg_loss = running_loss / n_batches
                print(f"  Batch {batch_idx+1}/{total_batches} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s", flush=True)
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            raise

    epoch_loss = running_loss / max(1, n_batches)
    epoch_box_loss = running_box_loss / max(1, n_batches)
    epoch_obj_loss = running_obj_loss / max(1, n_batches)
    epoch_cls_loss = running_cls_loss / max(1, n_batches)
    total_time = time.time() - start_time
    
    # Compute accuracy metric (on sampled batches for both train and val)
    if all_predictions and len(all_predictions) > 0:
        try:
            # Compute accuracy on sampled data
            # all_predictions is list of batches, each batch is list of scale predictions
            # all_targets is flattened list of targets per image
            accuracy = compute_simple_accuracy(
                all_predictions,
                all_targets,
                criterion.anchors,
                num_classes,
                img_size=IMG_SIZE,
                iou_threshold=0.5
            )
        except Exception as e:
            # If computation fails, return 0 and print error for debugging
            if not training:  # Only print warning during validation
                print(f"Warning: Accuracy computation failed: {e}")
            accuracy = 0.0
    else:
        # If no samples, skip accuracy
        accuracy = 0.0
    
    return epoch_loss, accuracy, total_time

# Wrap main execution in __main__ guard
if __name__ == '__main__':
    import sys
    sys.stdout.flush()
    
    print("=" * 60)
    print("Starting Hybrid ViT-YOLO training on ExDark...")
    print("=" * 60)
    sys.stdout.flush()

    print("\n" + "=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    print(f"  Model: Hybrid ViT-YOLO")
    print(f"  ViT Backbone: {MODEL_NAME}")
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

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        sys.stdout.flush()

        train_loss, train_accuracy, train_time = run_epoch(train_loader, training=True, epoch_num=epoch)
        
        if train_loss is None:
            print(f"Training stopped due to NaN/Inf at epoch {epoch}")
            break
        
        val_loss, val_accuracy, val_time = run_epoch(val_loader, training=False, epoch_num=epoch)

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"NaN detected at epoch {epoch}! Training stopped.")
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Track accuracies
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Always show both train and val accuracy (may be 0.0 early on)
        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc (F1): {train_accuracy:.4f} | Val Acc (F1): {val_accuracy:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if neptune_run is not None:
            try:
                neptune_run["train/loss"].append(train_loss)
                neptune_run["val/loss"].append(val_loss)
                neptune_run["train/accuracy"].append(train_accuracy)
                neptune_run["val/accuracy"].append(val_accuracy)
                neptune_run["train/lr"].append(optimizer.param_groups[0]["lr"])
                neptune_run["epoch"].append(epoch)
                # Log loss components if available
                neptune_run.sync()
            except Exception as e:
                print(f"Warning: Neptune logging error: {e}")

    print("\nTraining complete!")

    # ----------------------------------------------------------------------
    # Save model
    # ----------------------------------------------------------------------
    save_dir = Project_Path / "runs" / "detectors" / "hybrid_vit_yolo_exdark"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving model to {save_dir} ...")
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
        'config': {
            'num_classes': num_classes,
            'vit_model_name': MODEL_NAME,
            'img_size': IMG_SIZE,
            'classes': Classes,
            'id2label': id2label,
            'label2id': label2id,
        }
    }, save_dir / "model.pt")
    print(f"Model saved to {save_dir / 'model.pt'}")

    # ----------------------------------------------------------------------
    # Plot training curves
    # ----------------------------------------------------------------------
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Create figure with two subplots: Loss and Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss curves
    ax1.plot(epochs, train_losses, marker="o", label="Train Loss", linewidth=2, markersize=6)
    ax1.plot(epochs, val_losses, marker="s", label="Val Loss", linewidth=2, markersize=6)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Hybrid ViT-YOLO on ExDark - Training/Validation Loss", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: Accuracy curves
    # Filter out epochs where accuracy wasn't computed (0.0 values)
    acc_epochs = [e for e, acc in zip(epochs, val_accuracies) if acc > 0]
    train_acc_filtered = [acc for acc, val_acc in zip(train_accuracies, val_accuracies) if val_acc > 0]
    val_acc_filtered = [acc for acc in val_accuracies if acc > 0]
    
    if len(acc_epochs) > 0:
        ax2.plot(acc_epochs, train_acc_filtered, marker="o", label="Train Acc (F1)", linewidth=2, markersize=6, color='green')
        ax2.plot(acc_epochs, val_acc_filtered, marker="s", label="Val Acc (F1)", linewidth=2, markersize=6, color='orange')
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy (F1 Score)", fontsize=12)
        ax2.set_title("Hybrid ViT-YOLO on ExDark - Training/Validation Accuracy", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        ax2.set_ylim(bottom=0)  # Start y-axis from 0
    else:
        ax2.text(0.5, 0.5, "Accuracy not computed\n(only computed during validation)", 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Hybrid ViT-YOLO on ExDark - Accuracy", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save combined plot
    plot_path_combined = save_dir / "training_curves_hybrid_vit_yolo_exdark.png"
    plt.savefig(plot_path_combined, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Training curves saved to: {plot_path_combined}")
    print(f"  - Loss curves: {len(train_losses)} epochs")
    if len(acc_epochs) > 0:
        print(f"  - Accuracy curves: {len(acc_epochs)} epochs (F1 score at IoU=0.5)")
    else:
        print(f"  - Accuracy: Not computed (only computed during validation)")

    if neptune_run is not None:
        try:
            neptune_run["plots/training_curves"].upload(str(plot_path_combined))
            neptune_run["final/train_loss"] = train_losses[-1]
            neptune_run["final/val_loss"] = val_losses[-1]
            if len(val_accuracies) > 0 and val_accuracies[-1] > 0:
                neptune_run["final/train_accuracy"] = train_accuracies[-1]
                neptune_run["final/val_accuracy"] = val_accuracies[-1]
            neptune_run.stop()
            print("Neptune run closed.")
        except Exception as e:
            print(f"Warning: could not upload plot/close Neptune: {e}")

