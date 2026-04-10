import matplotlib.pyplot as plt
import os
from config import GRAPHS_DIR

def plot_training_curves(train_losses, val_losses, train_abo_accs, val_abo_accs, train_rh_accs, val_rh_accs):
    """
    Plot training curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_abo_accs: List of training ABO accuracies
        val_abo_accs: List of validation ABO accuracies
        train_rh_accs: List of training Rh accuracies
        val_rh_accs: List of validation Rh accuracies
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # ABO Accuracy
    ax2.plot(epochs, train_abo_accs, label='Train ABO Acc')
    ax2.plot(epochs, val_abo_accs, label='Val ABO Acc')
    ax2.set_title('ABO Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Rh Accuracy
    ax3.plot(epochs, train_rh_accs, label='Train Rh Acc')
    ax3.plot(epochs, val_rh_accs, label='Val Rh Acc')
    ax3.set_title('Rh Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()

    # Combined Accuracy (if available)
    if len(train_abo_accs) == len(train_rh_accs):
        train_combined = [(a + r) / 2 for a, r in zip(train_abo_accs, train_rh_accs)]
        val_combined = [(a + r) / 2 for a, r in zip(val_abo_accs, val_rh_accs)]
        ax4.plot(epochs, train_combined, label='Train Combined Acc')
        ax4.plot(epochs, val_combined, label='Val Combined Acc')
        ax4.set_title('Combined Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'training_curves.png'))
    plt.close()

def plot_comparison_bars(metrics):
    """
    Plot comparison bar charts.

    Args:
        metrics: Dictionary with evaluation metrics
    """
    tasks = ['ABO', 'Rh']
    accuracies = [metrics['abo_accuracy'], metrics['rh_accuracy']]
    precisions = [metrics['abo_precision'], metrics['rh_precision']]
    recalls = [metrics['abo_recall'], metrics['rh_recall']]
    f1s = [metrics['abo_f1'], metrics['rh_f1']]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    ax1.bar(tasks, accuracies, color=['blue', 'green'])
    ax1.set_title('Accuracy')
    ax1.set_ylim(0, 1)

    ax2.bar(tasks, precisions, color=['blue', 'green'])
    ax2.set_title('Precision')
    ax2.set_ylim(0, 1)

    ax3.bar(tasks, recalls, color=['blue', 'green'])
    ax3.set_title('Recall')
    ax3.set_ylim(0, 1)

    ax4.bar(tasks, f1s, color=['blue', 'green'])
    ax4.set_title('F1-Score')
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'comparison_bars.png'))
    plt.close()