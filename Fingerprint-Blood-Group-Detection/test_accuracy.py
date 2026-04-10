#!/usr/bin/env python3
"""
Test Accuracy Evaluation Script
Evaluates the trained model on the test dataset and prints accuracy metrics.
"""

import torch
import os
import numpy as np
from data.dataset import create_datasets
from models.hybrid_model import HybridMultiModalNet
from config import CHECKPOINTS_DIR

def main():
    print("🔍 Loading best trained model...")

    # Find the best checkpoint (lowest loss)
    checkpoints = [f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith('.pth')]
    if not checkpoints:
        print("❌ No model checkpoints found!")
        return

    checkpoints.sort(key=lambda x: float(x.split('_loss_')[1].split('.pth')[0]))
    best_checkpoint = os.path.join(CHECKPOINTS_DIR, checkpoints[0])

    print(f"📁 Loading model: {os.path.basename(best_checkpoint)}")

    # Load model
    model = HybridMultiModalNet()
    checkpoint = torch.load(best_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device('cpu')
    model.to(device)

    print("📊 Creating test dataset...")
    _, _, test_dataset = create_datasets()

    print(f"🧪 Evaluating on {len(test_dataset)} test samples...")
    print("⏳ This may take a few minutes due to feature extraction...")

    # Initialize counters
    abo_correct = 0
    rh_correct = 0
    combined_correct = 0
    total_samples = len(test_dataset)

    # ABO and Rh class names for display
    abo_classes = ['A', 'B', 'AB', 'O']
    rh_classes = ['+', '-']

    with torch.no_grad():
        for i, (image, handcrafted, abo_label, rh_label) in enumerate(test_dataset):
            if (i + 1) % 100 == 0:
                print(f"🔄 Processed {i + 1}/{total_samples} samples...")

            # Move to device
            image = image.unsqueeze(0).to(device)
            handcrafted = handcrafted.unsqueeze(0).to(device)

            # Forward pass
            abo_logits, rh_logits = model(image, handcrafted)

            # Get predictions
            _, abo_pred = torch.max(abo_logits, 1)
            _, rh_pred = torch.max(rh_logits, 1)

            # Check accuracy
            if abo_pred.item() == abo_label.item():
                abo_correct += 1
            if rh_pred.item() == rh_label.item():
                rh_correct += 1
            if abo_pred.item() == abo_label.item() and rh_pred.item() == rh_label.item():
                combined_correct += 1

    # Calculate accuracies
    abo_accuracy = (abo_correct / total_samples) * 100
    rh_accuracy = (rh_correct / total_samples) * 100
    combined_accuracy = (combined_correct / total_samples) * 100

    # Print results
    print("\n" + "="*70)
    print("🎯 MODEL TEST ACCURACY RESULTS")
    print("="*70)
    print(f"Total Test Samples:     {total_samples}")
    print()
    print("📈 ACCURACY METRICS:")
    print(f"ABO Blood Group:        {abo_accuracy:.2f}% ({abo_correct}/{total_samples})")
    print(f"Rh Factor:              {rh_accuracy:.2f}% ({rh_correct}/{total_samples})")
    print(f"Combined 8-Class:       {combined_accuracy:.2f}% ({combined_correct}/{total_samples})")
    print()
    print("🏆 PERFORMANCE SUMMARY:")
    if combined_accuracy >= 90:
        print("   ⭐ Excellent! Model performs exceptionally well!")
    elif combined_accuracy >= 80:
        print("   ✅ Very Good! Model shows strong performance!")
    elif combined_accuracy >= 70:
        print("   👍 Good! Model performs adequately!")
    else:
        print("   📈 Fair! Consider additional training or data augmentation!")
    print("="*70)

if __name__ == "__main__":
    main()