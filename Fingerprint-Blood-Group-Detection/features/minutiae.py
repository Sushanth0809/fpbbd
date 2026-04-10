import numpy as np
import cv2
from skimage.morphology import skeletonize

def extract_minutiae(image):
    """
    Extract minutiae points (ridge endings and bifurcations) from fingerprint image.

    Args:
        image: Grayscale image (numpy array)

    Returns:
        features: Dictionary with minutiae counts and spatial features
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Skeletonize
    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255

    # Find minutiae using crossing number
    minutiae = []
    h, w = skeleton.shape

    for i in range(1, h-1):
        for j in range(1, w-1):
            if skeleton[i, j] == 255:
                # Get 8-neighbors
                neighbors = [
                    skeleton[i-1, j-1], skeleton[i-1, j], skeleton[i-1, j+1],
                    skeleton[i, j-1], skeleton[i, j+1],
                    skeleton[i+1, j-1], skeleton[i+1, j], skeleton[i+1, j+1]
                ]
                neighbors = [1 if n > 0 else 0 for n in neighbors]

                # Crossing number
                cn = 0
                for k in range(8):
                    cn += abs(neighbors[k] - neighbors[(k+1)%8])
                cn //= 2

                if cn == 1:
                    minutiae.append(('ending', (i, j)))
                elif cn == 3:
                    minutiae.append(('bifurcation', (i, j)))

    # Extract features
    endings = [m for m in minutiae if m[0] == 'ending']
    bifurcations = [m for m in minutiae if m[0] == 'bifurcation']

    # Spatial distribution: divide into quadrants
    h_half, w_half = h // 2, w // 2
    quadrants = {
        'top_left': 0,
        'top_right': 0,
        'bottom_left': 0,
        'bottom_right': 0
    }

    for _, (y, x) in minutiae:
        if y < h_half and x < w_half:
            quadrants['top_left'] += 1
        elif y < h_half and x >= w_half:
            quadrants['top_right'] += 1
        elif y >= h_half and x < w_half:
            quadrants['bottom_left'] += 1
        else:
            quadrants['bottom_right'] += 1

    features = {
        'total_minutiae': len(minutiae),
        'endings_count': len(endings),
        'bifurcations_count': len(bifurcations),
        'endings_ratio': len(endings) / len(minutiae) if len(minutiae) > 0 else 0,
        'bifurcations_ratio': len(bifurcations) / len(minutiae) if len(minutiae) > 0 else 0,
        **quadrants
    }

    # Convert to vector
    feature_vector = np.array([
        features['total_minutiae'],
        features['endings_count'],
        features['bifurcations_count'],
        features['endings_ratio'],
        features['bifurcations_ratio'],
        features['top_left'],
        features['top_right'],
        features['bottom_left'],
        features['bottom_right']
    ])

    return feature_vector