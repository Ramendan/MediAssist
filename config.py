"""
MediAssist - Central Configuration
===================================
All tuneable parameters in one place.  Change values here rather than
searching through individual source files.

Import pattern
--------------
    from config import RANDOM_STATE, MODEL_SELECTION_METRIC
"""

from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_DATA_PATH = DATA_DIR / "cardio_train.csv"

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
RANDOM_STATE = 42

# ------------------------------------------------------------------
# Dataset split ratios  (must sum to 1.0)
# train / validation / test
# Validation is used exclusively for threshold tuning.
# ------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ------------------------------------------------------------------
# Outlier removal — biological plausibility bounds
# ------------------------------------------------------------------
OUTLIER_BOUNDS = {
    "height": (100, 220),   # cm
    "weight": (30, 200),    # kg
    "ap_hi":  (60, 250),    # mmHg systolic
    "ap_lo":  (40, 160),    # mmHg diastolic
}

# ------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------
HYPERPARAMETER_SEARCH_ITER = 30   # RandomizedSearchCV iterations
CV_FOLDS                   = 5    # Cross-validation folds
MODEL_SELECTION_METRIC     = "recall"   # Scoring metric for CV search

# ------------------------------------------------------------------
# Threshold tuning
# ------------------------------------------------------------------
THRESHOLD_BETA = 2.0   # F-beta weighting: 2.0 = recall-focused (recall 2× precision)

# ------------------------------------------------------------------
# Plot output
# ------------------------------------------------------------------
PLOT_DPI = 150
