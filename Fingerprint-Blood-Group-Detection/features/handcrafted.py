import numpy as np
from .ridge_density import compute_ridge_density
from .ridge_orientation import compute_ridge_orientation
from .minutiae import extract_minutiae

def extract_all(image_path):
    """
    Extract all handcrafted features from a fingerprint image.

    Args:
        image_path: Path to the fingerprint image

    Returns:
        feature_vector: Concatenated and normalized handcrafted features
    """
    # Load image
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Extract features
    ridge_density = compute_ridge_density(image)
    ridge_orientation = compute_ridge_orientation(image)
    minutiae_features = extract_minutiae(image)

    # Concatenate
    features = np.concatenate([ridge_density, ridge_orientation, minutiae_features])

    # Normalize
    if np.std(features) > 0:
        features = (features - np.mean(features)) / np.std(features)

    return features