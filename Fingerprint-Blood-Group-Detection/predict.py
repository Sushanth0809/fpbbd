import torch
import argparse
from PIL import Image
import os
from models.hybrid_model import HybridMultiModalNet
from features.handcrafted import extract_all
from data.transforms import val_test_transforms
from utils.helpers import get_device
from config import CHECKPOINTS_DIR, ABO_CLASSES, RH_CLASSES

# Reverse mappings
ABO_LABELS = {v: k for k, v in ABO_CLASSES.items()}
RH_LABELS = {v: k for k, v in RH_CLASSES.items()}

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    model = HybridMultiModalNet()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_blood_group(image_path, model, device):
    """
    Predict blood group from fingerprint image.

    Args:
        image_path: Path to fingerprint image
        model: Trained model
        device: Device

    Returns:
        abo_pred: Predicted ABO group
        rh_pred: Predicted Rh factor
        abo_conf: Confidence for ABO
        rh_conf: Confidence for Rh
    """
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

    return abo_pred, rh_pred, abo_conf, rh_conf

def main():
    parser = argparse.ArgumentParser(description="Predict blood group from fingerprint")
    parser.add_argument('image_path', help='Path to fingerprint image')
    parser.add_argument('--checkpoint', default=None, help='Path to model checkpoint')

    args = parser.parse_args()

    device = get_device()

    # Find latest checkpoint if not specified
    if args.checkpoint is None:
        checkpoints = [f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith('.pth')]
        if not checkpoints:
            print("No checkpoints found in", CHECKPOINTS_DIR)
            return
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINTS_DIR, x)))
        args.checkpoint = os.path.join(CHECKPOINTS_DIR, checkpoints[-1])

    print(f"Loading model from: {args.checkpoint}")

    model = load_model(args.checkpoint, device)

    abo, rh, abo_conf, rh_conf = predict_blood_group(args.image_path, model, device)

    print("Predicted Blood Group:")
    print(f"ABO Group: {abo} (confidence: {abo_conf:.4f})")
    print(f"Rh Factor: {rh} (confidence: {rh_conf:.4f})")
    print(f"Full Blood Group: {abo}{rh}")

if __name__ == "__main__":
    main()