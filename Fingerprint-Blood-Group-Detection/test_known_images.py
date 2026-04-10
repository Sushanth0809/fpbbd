#!/usr/bin/env python3
"""
Test predictions on known dataset images
"""

import torch
import os
from PIL import Image
from models.hybrid_model import HybridMultiModalNet
from features.handcrafted import extract_all
from data.transforms import val_test_transforms
from config import CHECKPOINTS_DIR, ABO_CLASSES, RH_CLASSES

# Reverse mappings
ABO_LABELS = {v: k for k, v in ABO_CLASSES.items()}
RH_LABELS = {v: k for k, v in RH_CLASSES.items()}

def load_model():
    """Load the trained model"""
    checkpoints = [f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith('.pth')]
    checkpoints.sort(key=lambda x: float(x.split('_loss_')[1].split('.pth')[0]))
    best_checkpoint = os.path.join(CHECKPOINTS_DIR, checkpoints[0])

    model = HybridMultiModalNet()
    checkpoint = torch.load(best_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_image(model, image_path, true_label=None):
    """Predict on a single image"""
    device = torch.device('cpu')
    model.to(device)

    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_test_transforms(image).unsqueeze(0).to(device)

    # Extract handcrafted features
    handcrafted = extract_all(image_path)
    handcrafted_tensor = torch.tensor(handcrafted, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        abo_logits, rh_logits = model(image_tensor, handcrafted_tensor)

        abo_probs = torch.softmax(abo_logits, dim=1)
        rh_probs = torch.softmax(rh_logits, dim=1)

        abo_pred_idx = torch.argmax(abo_probs, dim=1).item()
        rh_pred_idx = torch.argmax(rh_probs, dim=1).item()

        abo_conf = abo_probs[0, abo_pred_idx].item()
        rh_conf = rh_probs[0, rh_pred_idx].item()

        abo_pred = ABO_LABELS[abo_pred_idx]
        rh_pred = RH_LABELS[rh_pred_idx]

    blood_group = f"{abo_pred}{rh_pred}"
    true_blood_group = true_label if true_label else "Unknown"

    print(f"Image: {os.path.basename(image_path)}")
    print(f"True: {true_blood_group} | Predicted: {blood_group}")
    print(f"ABO: {abo_pred} ({abo_conf:.3f}) | Rh: {rh_pred} ({rh_conf:.3f})")
    print(f"Correct: {blood_group == true_blood_group}")
    print("-" * 50)

def main():
    model = load_model()

    # Test a few images from different classes
    test_images = [
        ('dataset/dataset_blood_group/A+/cluster_0_1001.BMP', 'A+'),
        ('dataset/dataset_blood_group/B-/cluster_3_1017.BMP', 'B-'),
        ('dataset/dataset_blood_group/AB+/cluster_4_100.BMP', 'AB+'),
        ('dataset/dataset_blood_group/O-/cluster_7_1002.BMP', 'O-'),
    ]

    for image_path, true_label in test_images:
        if os.path.exists(image_path):
            predict_image(model, image_path, true_label)
        else:
            print(f"Image not found: {image_path}")

if __name__ == "__main__":
    main()