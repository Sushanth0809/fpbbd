# Fingerprint-Based Blood Group Detection

A non-invasive framework for predicting blood groups from fingerprint images using a hybrid multi-modal deep learning approach with feature fusion and multi-task learning.

## Overview

This project implements a novel system that combines:
- **CNN Features**: Extracted from fingerprint images using a pretrained ResNet-18 backbone
- **Handcrafted Features**: Ridge density, orientation maps, and minutiae points
- **Feature Fusion**: Attention-based fusion using Squeeze-and-Excitation blocks
- **Multi-Task Learning**: Simultaneous prediction of ABO groups (A, B, AB, O) and Rh factor (+, -)

## Features

- Hybrid multi-modal architecture combining deep learning and traditional computer vision
- Domain-specific preprocessing for fingerprint analysis
- Multi-task loss with weighted cross-entropy
- Comprehensive evaluation with per-task metrics
- Training visualization and model checkpointing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Fingerprint-Blood-Group-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset should be organized as follows:
```
dataset/dataset_blood_group/
├── A+/
├── A-/
├── B+/
├── B-/
├── AB+/
├── AB-/
├── O+/
└── O-/
```

The dataset is automatically downloaded and organized during setup.

## Usage

### Training

Train the model:
```bash
python train.py --epochs 50 --batch_size 32
```

For quick testing:
```bash
python train.py --quick_test
```

### Prediction

Predict blood group from a fingerprint image:
```bash
python predict.py path/to/fingerprint.jpg
```

### Evaluation

Evaluate on test set (modify train.py to include evaluation after training).

## Project Structure

```
Fingerprint-Blood-Group-Detection/
├── config.py                     # Hyperparameters & paths
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
├── train.py                      # Main training entry point
├── predict.py                    # Inference on single image
│
├── data/
│   ├── dataset.py                # Custom Dataset class
│   └── transforms.py             # Image augmentations
│
├── features/
│   ├── ridge_density.py          # Ridge density extraction
│   ├── ridge_orientation.py      # Ridge orientation maps
│   ├── minutiae.py               # Minutiae detection
│   └── handcrafted.py            # Feature orchestration
│
├── models/
│   ├── backbone.py               # ResNet18 CNN backbone
│   ├── fusion.py                 # Feature fusion module
│   ├── heads.py                  # Classification heads
│   └── hybrid_model.py           # Complete model
│
├── training/
│   ├── trainer.py                # Training loop
│   ├── losses.py                 # Multi-task loss
│   └── scheduler.py              # LR scheduler
│
├── evaluation/
│   ├── evaluate.py               # Evaluation pipeline
│   └── visualize.py              # Training curves & charts
│
├── utils/
│   └── helpers.py                # Utilities
│
├── outputs/
│   ├── checkpoints/              # Model checkpoints
│   └── graphs/                   # Training visualizations
│
└── notebooks/
    └── demo.ipynb                # Interactive demo
```

## Key Components

### Handcrafted Features
- **Ridge Density**: Gabor filter responses at multiple orientations
- **Ridge Orientation**: Gradient-based orientation histograms
- **Minutiae**: Ridge endings and bifurcations via crossing number method

### Model Architecture
- **Backbone**: Pretrained ResNet-18 with frozen early layers
- **Fusion**: SE-block attention for feature combination
- **Heads**: Separate classifiers for ABO (4-class) and Rh (2-class)

### Training
- Multi-task loss with configurable weights
- Early stopping and model checkpointing
- Stratified train/val/test splits

## Results

After training, the model provides:
- Per-task accuracy, precision, recall, F1-score
- Confusion matrices for ABO and Rh classification
- Combined 8-class accuracy
- Training curves and performance visualizations

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for faster training)
- 8GB+ RAM

**Note**: Training on CPU is supported but significantly slower. For CPU training, consider reducing batch size to 8 and epochs to 10-20.

## License

[Add license information]

## Citation

[Add citation if applicable]