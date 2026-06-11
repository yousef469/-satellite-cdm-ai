"""
Production configuration for satellite collision avoidance training.

All settings tunable here. No hardcoded magic numbers in other files.
"""
import os
import torch

_BASE = os.path.dirname(os.path.abspath(__file__))

# ── Model Architecture ──────────────────────────────────────────────────────
NUM_CDM_FEATURES = 53
MAX_CDMS_PER_EVENT = 15
HIDDEN_DIM = 96
TCN_BLOCKS = 4
TCN_DROPOUT = 0.30
NUM_SUBSPACES = 3
SUBSPACE_DIM = 16
PROTO_DIM = NUM_SUBSPACES * SUBSPACE_DIM  # 48
NUM_PROTOTYPES = 24
TEMPERATURE = 0.07

# ── Training Phases ────────────────────────────────────────────────────────
# Phase 1: train risk_head only (backbone frozen)
PHASE1_EPOCHS = 5
PHASE1_LR = 1e-3
PHASE1_WEIGHT_DECAY = 0.0

# Phase 2: full fine-tune
PHASE2_EPOCHS = 100
PHASE2_BACKBONE_LR = 5e-4
PHASE2_HEAD_LR = 1e-4
PHASE2_WEIGHT_DECAY = 5e-4
PHASE2_WARMUP_EPOCHS = 3

# Loss weights
BCE_POS_WEIGHT_CAP = 15.0      # cap on pos_weight to prevent saturation
CONTRASTIVE_WEIGHT = 0.05       # final contrastive loss weight
CONTRASTIVE_RAMP_EPOCHS = 10   # ramp from 0 to CONTRASTIVE_WEIGHT over N epochs
ORTHOGONALITY_WEIGHT = 0.01

# ── Data ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 128
NUM_WORKERS = 4
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10

# Synthetic data fallback (when real data unavailable)
NUM_SYNTHETIC_EVENTS = 5000
RISK_RATIO = 0.04

# ESA Kelvins dataset path (relative to script directory)
ESA_DATA_PATH = os.path.join(_BASE, "data", "esa_kelvins_full.csv")

# ── Checkpointing & Logging ────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(_BASE, "checkpoints")
CHECKPOINT_PATH = os.path.join(_BASE, "checkpoints", "best.pt")
LOG_INTERVAL = 10                # batches between logging
EVAL_INTERVAL = 1                # epochs between validation

# ── Hardware ────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Algorithm ────────────────────────────────────────────────────────────────
ALGORITHM_AMBER_THRESHOLD = 0.30
ALGORITHM_RED_THRESHOLD = 0.45
ALGORITHM_CRITICAL_THRESHOLD = 0.70
ALGORITHM_TEMPORAL_WINDOW = 5
ALGORITHM_TEMPORAL_MIN_POSITIVE = 3
ALGORITHM_FUEL_RESERVE_FRACTION = 0.15
ALGORITHM_MIN_LEAD_HOURS = 6.0
ALGORITHM_MAX_LEAD_HOURS = 48.0
