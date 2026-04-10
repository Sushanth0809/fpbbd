#!/usr/bin/env python3
"""
Display all saved model checkpoints with their details
"""

import os
from pathlib import Path
from config import CHECKPOINTS_DIR

def show_all_checkpoints():
    """Display all saved checkpoints"""
    
    print("\n" + "="*80)
    print("📦 ALL MODEL CHECKPOINTS")
    print("="*80 + "\n")
    
    if not os.path.exists(CHECKPOINTS_DIR):
        print(f"❌ Checkpoints directory not found: {CHECKPOINTS_DIR}")
        return
    
    # Get all checkpoints
    checkpoints = [f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith('.pth')]
    
    if not checkpoints:
        print(f"ℹ️  No checkpoints found in {CHECKPOINTS_DIR}")
        print("💡 Run 'python train.py' to train and create checkpoints\n")
        return
    
    # Sort by loss (ascending - lower loss first)
    checkpoints.sort(key=lambda x: float(x.split('_loss_')[1].split('.pth')[0]))
    
    print(f"Found {len(checkpoints)} checkpoint(s):\n")
    
    for i, checkpoint_name in enumerate(checkpoints, 1):
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_name)
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        
        # Extract epoch and loss
        parts = checkpoint_name.replace('model_', '').replace('.pth', '').split('_loss_')
        epoch_str = parts[0].replace('epoch_', '')
        loss_str = parts[1]
        
        # Determine ranking
        if i == 1:
            rank = "🏆 BEST"
        elif i <= 3:
            rank = f"#{i}"
        else:
            rank = f" {i}"
        
        print(f"{rank} | Epoch: {epoch_str:>2} | Loss: {loss_str:>8} | Size: {file_size_mb:>6.2f}MB")
        print(f"     Path: {checkpoint_name}")
        print()
    
    print("="*80)
    print(f"✅ Total: {len(checkpoints)} checkpoint(s)")
    print(f"📁 Location: {CHECKPOINTS_DIR}")
    print(f"🏆 Best model: {checkpoints[0]}")
    print("="*80 + "\n")

if __name__ == "__main__":
    show_all_checkpoints()
