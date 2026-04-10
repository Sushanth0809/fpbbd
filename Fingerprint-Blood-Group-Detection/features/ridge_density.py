import numpy as np
import cv2
from skimage.filters import gabor
from skimage import img_as_float

def compute_ridge_density(image, block_size=56, orientations=8):
    """
    Compute local ridge density using Gabor filter responses at multiple orientations.

    Args:
        image: Grayscale image (numpy array)
        block_size: Size of blocks to divide the image into (56 for 4x4 grid on 224x224)
        orientations: Number of orientations for Gabor filters

    Returns:
        feature_vector: Ridge density features (16,)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = img_as_float(image)
    h, w = image.shape

    # Fixed 4x4 grid
    num_blocks_h = 4
    num_blocks_w = 4

    features = np.zeros(num_blocks_h * num_blocks_w)

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            if block.size == 0:
                continue

            # Compute Gabor responses for different orientations
            max_response = 0
            for theta in np.linspace(0, np.pi, orientations, endpoint=False):
                filtered_real, filtered_imag = gabor(block, frequency=0.1, theta=theta)
                response = np.sqrt(filtered_real**2 + filtered_imag**2)
                max_response = max(max_response, np.mean(response))

            features[i*num_blocks_w + j] = max_response

    return features