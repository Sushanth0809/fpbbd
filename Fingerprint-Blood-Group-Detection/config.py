import os

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 8  # Reduced for CPU training
EPOCHS = 5      # Very quick training for testing
IMAGE_SIZE = (224, 224)

# Multi-task loss weights
ALPHA_ABO = 1.0  # Weight for ABO classification loss
BETA_RH = 0.5    # Weight for Rh classification loss

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'dataset_blood_group')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, 'checkpoints')
GRAPHS_DIR = os.path.join(OUTPUTS_DIR, 'graphs')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure output directories exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Model parameters
NUM_CLASSES_ABO = 4  # A, B, AB, O
NUM_CLASSES_RH = 2   # +, -

# Training parameters
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Label mappings
ABO_CLASSES = {'A': 0, 'B': 1, 'AB': 2, 'O': 3}
RH_CLASSES = {'+': 0, '-': 1}

# Feature dimensions
CNN_FEATURE_DIM = 512         # Dimension of CNN features
HANDCRAFTED_FEATURE_DIM = 33  # Dimension of handcrafted features
FUSED_FEATURE_DIM = 256        # Dimension after fusion