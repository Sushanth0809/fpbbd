import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging
from tqdm import tqdm
from .losses import MultiTaskLoss
from .scheduler import get_scheduler
from evaluation.evaluate import evaluate_model
from config import EPOCHS, LEARNING_RATE, BATCH_SIZE, CHECKPOINTS_DIR
from utils.helpers import get_device

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, device=None):
        """
        Trainer for the hybrid model.

        Args:
            model: The model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            device: Device to use
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device or get_device()

        self.model.to(self.device)

        # Data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Scheduler
        self.scheduler = get_scheduler(self.optimizer, scheduler_type='step', step_size=10, gamma=0.5)

        # Loss
        self.criterion = MultiTaskLoss()

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0

        # Logging
        logging.info(f"Training on device: {self.device}")
        logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # History for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_abo_accs = []
        self.val_abo_accs = []
        self.train_rh_accs = []
        self.val_rh_accs = []

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        abo_correct = 0
        rh_correct = 0
        total_samples = 0

        for images, handcrafted, abo_labels, rh_labels in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            handcrafted = handcrafted.to(self.device)
            abo_labels = abo_labels.to(self.device)
            rh_labels = rh_labels.to(self.device)

            self.optimizer.zero_grad()

            abo_logits, rh_logits = self.model(images, handcrafted)

            loss, _, _ = self.criterion(abo_logits, rh_logits, abo_labels, rh_labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Accuracy
            _, abo_pred = torch.max(abo_logits, 1)
            _, rh_pred = torch.max(rh_logits, 1)
            abo_correct += (abo_pred == abo_labels).sum().item()
            rh_correct += (rh_pred == rh_labels).sum().item()
            total_samples += abo_labels.size(0)

        avg_loss = total_loss / len(self.train_loader)
        abo_acc = abo_correct / total_samples
        rh_acc = rh_correct / total_samples

        self.train_losses.append(avg_loss)
        self.train_abo_accs.append(abo_acc)
        self.train_rh_accs.append(rh_acc)

        return avg_loss, abo_acc, rh_acc

        return avg_loss, abo_acc, rh_acc

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        abo_correct = 0
        rh_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, handcrafted, abo_labels, rh_labels in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                handcrafted = handcrafted.to(self.device)
                abo_labels = abo_labels.to(self.device)
                rh_labels = rh_labels.to(self.device)

                abo_logits, rh_logits = self.model(images, handcrafted)

                loss, _, _ = self.criterion(abo_logits, rh_logits, abo_labels, rh_labels)
                total_loss += loss.item()

                _, abo_pred = torch.max(abo_logits, 1)
                _, rh_pred = torch.max(rh_logits, 1)
                abo_correct += (abo_pred == abo_labels).sum().item()
                rh_correct += (rh_pred == rh_labels).sum().item()
                total_samples += abo_labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        abo_acc = abo_correct / total_samples
        rh_acc = rh_correct / total_samples

        self.val_losses.append(avg_loss)
        self.val_abo_accs.append(abo_acc)
        self.val_rh_accs.append(rh_acc)

        return avg_loss, abo_acc, rh_acc

    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'model_epoch_{epoch}_loss_{val_loss:.4f}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")

    def train(self, num_epochs=EPOCHS):
        """Full training loop."""
        for epoch in range(1, num_epochs + 1):
            logging.info(f"\n{'='*70}")
            logging.info(f"Epoch {epoch}/{num_epochs}")
            logging.info(f"{'='*70}")

            train_loss, train_abo_acc, train_rh_acc = self.train_epoch()
            val_loss, val_abo_acc, val_rh_acc = self.validate()

            # Print formatted results
            logging.info(f"\n📊 TRAINING RESULTS:")
            logging.info(f"   Loss: {train_loss:.6f} | ABO Acc: {train_abo_acc*100:.2f}% | Rh Acc: {train_rh_acc*100:.2f}%")
            logging.info(f"\n📊 VALIDATION RESULTS:")
            logging.info(f"   Loss: {val_loss:.6f} | ABO Acc: {val_abo_acc*100:.2f}% | Rh Acc: {val_rh_acc*100:.2f}%")
            logging.info(f"   Combined: {(val_abo_acc + val_rh_acc) / 2 * 100:.2f}%")

            self.scheduler.step()

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
                logging.info(f"✅ New best model saved! (Val Loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                logging.info(f"⚠️  No improvement ({self.patience_counter}/{self.patience})")
                if self.patience_counter >= self.patience:
                    logging.info("🛑 Early stopping triggered")
                    break

        logging.info("\n✅ Training completed!")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_abo_accs': self.train_abo_accs,
            'val_abo_accs': self.val_abo_accs,
            'train_rh_accs': self.train_rh_accs,
            'val_rh_accs': self.val_rh_accs
        }