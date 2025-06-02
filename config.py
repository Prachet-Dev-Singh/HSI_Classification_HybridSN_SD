# config.py

# --- Dataset and Preprocessing ---
DATASET_NAME = "IndianPines"
DATA_PATH = "./"
INDIAN_PINES_URL_DATA = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
INDIAN_PINES_URL_GT = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
NUM_PCA_COMPONENTS = 30

PATCH_SIZE = 9
VIT_SUB_PATCH_SIZE = 3 # Results in 3x3 initial token grid
VIT_IMG_SIZE = (PATCH_SIZE, PATCH_SIZE)

TEST_RATIO = 0.8
VAL_RATIO_FROM_TRAIN = 0.1
RANDOM_SEED = 42

# --- Swin Transformer Backbone Parameters ---
SWIN_IN_CHANS = NUM_PCA_COMPONENTS
SWIN_INITIAL_PATCH_EMBED_DIM = 96
SWIN_PATCH_NORM = True

# MODIFIED for Experiment B: Enable Absolute Positional Embedding
SWIN_APE = True # Previously False

# MODIFIED for Experiment B: Drastically simplified model - 1 stage, 2 blocks
SWIN_DEPTHS = [6]      # Only 6 Swin blocks in total
SWIN_NUM_HEADS = [3]   # 96 initial_embed_dim is divisible by 3 (head_dim=32)
SWIN_WINDOW_SIZES = [3] # For the 3x3 token grid

SWIN_MLP_RATIO = 4.0
SWIN_QKV_BIAS = True
SWIN_QK_SCALE = None
SWIN_DROP_RATE = 0.0  # No dropout in MLPs/Linear for this debug phase
SWIN_ATTN_DROP_RATE = 0.0 # No attention dropout for this debug phase
SWIN_DROP_PATH_RATE = 0.1 
# --- Training Parameters ---
BATCH_SIZE = 64
EPOCHS = 150 

# MODIFIED for Experiment B (Attempt 1 - higher LR, lower WD)
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01 # Lowered from 0.05

# --- Loss Weights (MODIFIED for Experiment B - Disable all but final CE) ---
LAMBDA_CE_FINAL = 1.0
LAMBDA_CE_AUX = 0.1
LAMBDA_KL_DISTILL = 0.1
DISTILLATION_TEMP = 3.0
LAMBDA_L2_HINT = 0.001

# --- Dynamically set ---
NUM_CLASSES_ACTUAL = None