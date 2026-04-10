import numpy as np
import cv2

def compute_ridge_orientation(image, block_size=16, bins=8):
    """
    Compute gradient-based ridge orientation maps and extract histogram features.

    Args:
        image: Grayscale image (numpy array)
        block_size: Size of blocks for local orientation
        bins: Number of bins for orientation histogram

    Returns:
        feature_vector: Orientation histogram features
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute orientation
    orientation = np.arctan2(sobely, sobelx)

    h, w = image.shape
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size

    orientations = []

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block_ori = orientation[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            if block_ori.size == 0:
                continue
            # Average orientation in the block
            mean_ori = np.mean(block_ori)
            orientations.append(mean_ori)

    # Create histogram
    hist, _ = np.histogram(orientations, bins=bins, range=(-np.pi, np.pi))
    hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist

    return hist