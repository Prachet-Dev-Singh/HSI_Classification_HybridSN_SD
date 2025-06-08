# config.py

# --- Dataset and Preprocessing ---
DATASET_NAME = "IndianPines"
DATA_PATH = "./"
INDIAN_PINES_URL_DATA = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
INDIAN_PINES_URL_GT = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
NUM_PCA_COMPONENTS = 30
PATCH_SIZE = 25 # Keep aligned with HybridSN paper for best results
VIT_IMG_SIZE = (PATCH_SIZE, PATCH_SIZE)
TEST_RATIO = 0.60
VAL_RATIO_FROM_TRAIN = 0.1
RANDOM_SEED = 42

# --- HybridSN Backbone Parameters ---
HYBRIDSN_INPUT_BANDS = NUM_PCA_COMPONENTS
HYBRIDSN_PATCH_SIZE = PATCH_SIZE
HYBRIDSN_CONV3D_LAYERS = [
    {"out_channels": 8,  "kernel_size": (7, 3, 3), "padding": (0, 0, 0)},
    {"out_channels": 16, "kernel_size": (5, 3, 3), "padding": (0, 0, 0)},
    {"out_channels": 32, "kernel_size": (3, 3, 3), "padding": (0, 0, 0)},
]
HYBRIDSN_USE_BATCHNORM3D = False
HYBRIDSN_ACTIVATION = "relu" # Main activation for HybridSN blocks

HYBRIDSN_CONV2D_OUT_CHANNELS = 64
HYBRIDSN_CONV2D_KERNEL_SIZE = 3
HYBRIDSN_CONV2D_PADDING = 0
HYBRIDSN_USE_BATCHNORM2D = False

HYBRIDSN_FC_HIDDEN_UNITS = [256, 128]
HYBRIDSN_DROPOUT_RATE_FC = 0.4

# --- NEW: Parameters for MLP Auxiliary Heads ---
# For a 5-layer MLP head (Input -> H1 -> H2 -> H3 -> H4 -> Output), specify 4 hidden dims
HYBRIDSN_AUX_HEAD_MLP_DIMS = [128, 128, 128, 128] # Example: 4 hidden layers of 128 units
HYBRIDSN_AUX_HEAD_MLP_DROPOUT = 0.2 # Dropout within the aux head MLP
HYBRIDSN_AUX_HEAD_ACTIVATION = "relu" # Activation for aux head MLPs (can be same as main)


# --- Training Parameters ---
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# --- Loss Weights (MODIFIED to emphasize shallow head learning) ---
LAMBDA_CE_FINAL = 1.0     # Final head is still important
LAMBDA_CE_AUX = 0.8       # INCREASED: More direct supervision for shallow heads
LAMBDA_KL_DISTILL = 0.1   # Kept moderate, or can try slightly lower if CE_AUX is high
DISTILLATION_TEMP = 5.0   # INCREASED: Softer targets for KL
LAMBDA_L2_HINT = 0.0005    # Keep small

# --- Dynamically set ---
NUM_CLASSES_ACTUAL = None

# --- NEW: Parameters for Self-Attention in STUDENT Auxiliary Head Pathway ---
# These will apply ONLY to the student aux heads (not the final/main head)
STUDENT_AUX_USE_SELF_ATTENTION = True # Master switch
STUDENT_AUX_SA_EMBED_DIM = None # This will be set by the feature dim at each aux head point (e.g., 8, 16, 32, 64, 256, 128)
                                # So, this param is not strictly needed if SA layer dynamically takes embed_dim
STUDENT_AUX_SA_NUM_HEADS = 4    # Number of heads for the self-attention layer
STUDENT_AUX_SA_DROPOUT = 0.1    # Dropout for the self-attention layer