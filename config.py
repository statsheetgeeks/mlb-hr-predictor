"""
config.py – Global configuration for the MLB HR Predictor
"""
import os
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(BASE_DIR, "cache")
DOCS_DIR   = os.path.join(BASE_DIR, "docs")
MODEL_PATH = os.path.join(CACHE_DIR, "hr_model.joblib")
META_PATH  = os.path.join(CACHE_DIR, "cache_meta.json")

# ── Season settings ───────────────────────────────────────────────────────────
CURRENT_YEAR   = datetime.now().year
TRAINING_YEARS = [CURRENT_YEAR - 2, CURRENT_YEAR - 1]   # e.g. 2023, 2024
ALL_YEARS      = TRAINING_YEARS + [CURRENT_YEAR]

# Approximate MLB season windows (March 28 – Oct 1)
SEASON_START_MMDD = "03-20"
SEASON_END_MMDD   = "10-05"

# ── Model hyperparameters ────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":      400,
    "max_depth":         12,
    "min_samples_leaf":  20,
    "max_features":      "sqrt",
    "class_weight":      "balanced",   # HRs are rare events
    "random_state":      42,
    "n_jobs":           -1,
}

# ── Feature settings ─────────────────────────────────────────────────────────
# Minimum PA threshold to include a batter in predictions
MIN_PA_CURRENT_SEASON = 30

# ── Weather ──────────────────────────────────────────────────────────────────
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
# Wind within ±60° of CF direction is considered "blowing out"
WIND_OUT_THRESHOLD_DEG = 60

# ── Top-N predictions to show on the site ────────────────────────────────────
TOP_N = 30
