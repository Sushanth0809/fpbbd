import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
from config import BATCH_SIZE, NUM_CLASSES_ABO, NUM_CLASSES_RH

def evaluate_model(model, dataset, device):
    """
    Evaluate the model on a dataset.

    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        device: Device to use

    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    abo_preds = []
    rh_preds = []
    abo_targets = []
    rh_targets = []

    with torch.no_grad():
        for images, handcrafted, abo_labels, rh_labels in loader:
            images = images.to(device)
            handcrafted = handcrafted.to(device)

            abo_logits, rh_logits = model(images, handcrafted)

            _, abo_pred = torch.max(abo_logits, 1)
            _, rh_pred = torch.max(rh_logits, 1)

            abo_preds.extend(abo_pred.cpu().numpy())
            rh_preds.extend(rh_pred.cpu().numpy())
            abo_targets.extend(abo_labels.numpy())
            rh_targets.extend(rh_labels.numpy())

    abo_preds = np.array(abo_preds)
    rh_preds = np.array(rh_preds)
    abo_targets = np.array(abo_targets)
    rh_targets = np.array(rh_targets)

    # ABO metrics
    abo_acc = accuracy_score(abo_targets, abo_preds)
    abo_precision, abo_recall, abo_f1, _ = precision_recall_fscore_support(abo_targets, abo_preds, average='weighted')
    abo_cm = confusion_matrix(abo_targets, abo_preds)

    # Rh metrics
    rh_acc = accuracy_score(rh_targets, rh_preds)
    rh_precision, rh_recall, rh_f1, _ = precision_recall_fscore_support(rh_targets, rh_preds, average='weighted')
    rh_cm = confusion_matrix(rh_targets, rh_preds)

    # Combined 8-class accuracy
    combined_targets = abo_targets * 2 + rh_targets  # 0-7
    combined_preds = abo_preds * 2 + rh_preds
    combined_acc = accuracy_score(combined_targets, combined_preds)

    metrics = {
        'abo_accuracy': abo_acc,
        'abo_precision': abo_precision,
        'abo_recall': abo_recall,
        'abo_f1': abo_f1,
        'abo_confusion_matrix': abo_cm,
        'rh_accuracy': rh_acc,
        'rh_precision': rh_precision,
        'rh_recall': rh_recall,
        'rh_f1': rh_f1,
        'rh_confusion_matrix': rh_cm,
        'combined_accuracy': combined_acc
    }

    return metrics

def print_evaluation_report(metrics):
    """Print evaluation report."""
    logging.info("=== Evaluation Report ===")
    logging.info(f"ABO Accuracy: {metrics['abo_accuracy']:.4f}")
    logging.info(f"ABO Precision: {metrics['abo_precision']:.4f}")
    logging.info(f"ABO Recall: {metrics['abo_recall']:.4f}")
    logging.info(f"ABO F1-Score: {metrics['abo_f1']:.4f}")
    logging.info("ABO Confusion Matrix:")
    logging.info(metrics['abo_confusion_matrix'])

    logging.info(f"Rh Accuracy: {metrics['rh_accuracy']:.4f}")
    logging.info(f"Rh Precision: {metrics['rh_precision']:.4f}")
    logging.info(f"Rh Recall: {metrics['rh_recall']:.4f}")
    logging.info(f"Rh F1-Score: {metrics['rh_f1']:.4f}")
    logging.info("Rh Confusion Matrix:")
    logging.info(metrics['rh_confusion_matrix'])

    logging.info(f"Combined 8-Class Accuracy: {metrics['combined_accuracy']:.4f}")

if __name__ == "__main__":
    import os
    from data.dataset import create_datasets
    from models.hybrid_model import HybridMultiModalNet
    from config import CHECKPOINTS_DIR

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("🔍 Loading best trained model...")

    # Find the best checkpoint (lowest loss)
    checkpoints = [f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith('.pth')]
    if not checkpoints:
        print("❌ No model checkpoints found!")
        exit(1)

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

    metrics = evaluate_model(model, test_dataset, device)
    print_evaluation_report(metrics)