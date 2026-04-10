#!/usr/bin/env python3
"""
Save Model in Pickle Format
Loads the best trained model and saves it in pickle format for deployment.
"""

import torch
import pickle
import os
from models.hybrid_model import HybridMultiModalNet
from config import CHECKPOINTS_DIR, MODELS_DIR

def save_model_pickle():
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

    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save model state_dict using pickle
    model_path = os.path.join(MODELS_DIR, 'best_model_state_dict.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model.state_dict(), f)

    print(f"✅ Model saved to: {model_path}")
    print(f"📏 File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    save_model_pickle()