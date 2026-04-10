import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from features.handcrafted import extract_all
from .transforms import train_transforms, val_test_transforms
from config import DATASET_DIR, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED, ABO_CLASSES, RH_CLASSES

class FingerprintDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, extract_features=True):
        """
        Custom dataset for fingerprint blood group classification.

        Args:
            image_paths: List of image file paths
            labels: List of (abo_label, rh_label) tuples
            transform: Image transformations
            extract_features: Whether to extract handcrafted features
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.extract_features = extract_features

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        abo_label, rh_label = self.labels[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Extract handcrafted features
        if self.extract_features:
            handcrafted = extract_all(image_path)
            handcrafted = torch.tensor(handcrafted, dtype=torch.float32)
        else:
            handcrafted = torch.zeros(64, dtype=torch.float32)  # Placeholder

        return image, handcrafted, abo_label, rh_label

def parse_label_from_path(path):
    """
    Parse blood group label from folder path.

    Args:
        path: Folder name like 'A+', 'B-', etc.

    Returns:
        abo: ABO class index
        rh: Rh class index
    """
    folder = os.path.basename(os.path.dirname(path))
    if len(folder) >= 2:
        abo_part = folder[:-1]
        rh_part = folder[-1]
        abo = ABO_CLASSES.get(abo_part, -1)
        rh = RH_CLASSES.get(rh_part, -1)
        return abo, rh
    return -1, -1

def create_datasets():
    """
    Create train, validation, and test datasets with stratified split.

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    image_paths = []
    labels = []

    # Collect all images and labels
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                path = os.path.join(root, file)
                abo, rh = parse_label_from_path(path)
                if abo != -1 and rh != -1:
                    image_paths.append(path)
                    labels.append((abo, rh))

    # Stratified split
    labels_array = np.array(labels)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels_array,
        test_size=VAL_SPLIT + TEST_SPLIT,
        stratify=labels_array,
        random_state=RANDOM_SEED
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
        stratify=temp_labels,
        random_state=RANDOM_SEED
    )

    # Create datasets
    train_dataset = FingerprintDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = FingerprintDataset(val_paths, val_labels, transform=val_test_transforms)
    test_dataset = FingerprintDataset(test_paths, test_labels, transform=val_test_transforms)

    return train_dataset, val_dataset, test_dataset