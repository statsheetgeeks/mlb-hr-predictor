"""
src/model.py
Random Forest model wrapper: train, evaluate, save, load.
"""

import os
import json
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config import MODEL_PATH, RF_PARAMS, CACHE_DIR


class HRModel:
    """Wraps a scikit-learn Pipeline (imputer → scaler → RandomForest)."""

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] = []
        self.feature_importances_: dict = {}
        self.train_info: dict = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        cv_folds: int = 5,
    ):
        self.feature_names = feature_names

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf",      RandomForestClassifier(**RF_PARAMS)),
        ])

        print(f"\nTraining Random Forest on {len(y):,} samples…")
        print(f"  HR rate: {y.mean():.4f}  ({int(y.sum()):,} HRs)")

        # Cross-validated ROC-AUC
        skf   = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        aucs  = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        print(f"  {cv_folds}-fold CV ROC-AUC: {aucs.mean():.4f} ± {aucs.std():.4f}")

        # Fit on full data
        pipe.fit(X, y)
        self.pipeline = pipe

        # Feature importances from the RF step
        rf = pipe.named_steps["rf"]
        imp = dict(zip(feature_names, rf.feature_importances_))
        self.feature_importances_ = dict(
            sorted(imp.items(), key=lambda x: x[1], reverse=True)
        )

        # Evaluate on training set (in-sample, optimistic but useful for sanity)
        y_proba = pipe.predict_proba(X)[:, 1]
        train_auc = roc_auc_score(y, y_proba)

        self.train_info = {
            "trained_at":       datetime.now().isoformat(),
            "n_samples":        int(len(y)),
            "hr_rate":          float(y.mean()),
            "cv_roc_auc_mean":  float(aucs.mean()),
            "cv_roc_auc_std":   float(aucs.std()),
            "train_roc_auc":    float(train_auc),
            "rf_params":        RF_PARAMS,
            "feature_names":    feature_names,
            "feature_importances": self.feature_importances_,
        }

        print(f"  Train ROC-AUC: {train_auc:.4f}")
        print("  Top 5 features:")
        for feat, val in list(self.feature_importances_.items())[:5]:
            print(f"    {feat:30s}  {val:.4f}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba_hr(self, X: np.ndarray) -> np.ndarray:
        """Return P(HR in game) for each row."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained / loaded yet.")
        return self.pipeline.predict_proba(X)[:, 1]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "pipeline":              self.pipeline,
            "feature_names":         self.feature_names,
            "feature_importances_":  self.feature_importances_,
            "train_info":            self.train_info,
        }
        joblib.dump(payload, path)

        # Also save a human-readable JSON summary
        meta_path = path.replace(".joblib", "_info.json")
        with open(meta_path, "w") as f:
            info_copy = dict(self.train_info)
            # feature_importances is already a plain dict – safe to dump
            json.dump(info_copy, f, indent=2, default=str)

        print(f"\nModel saved → {path}")

    def load(self, path: str = MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No model found at {path}.\n"
                "Run with --train first to build the model."
            )
        payload = joblib.load(path)
        self.pipeline             = payload["pipeline"]
        self.feature_names        = payload["feature_names"]
        self.feature_importances_ = payload["feature_importances_"]
        self.train_info           = payload["train_info"]
        print(f"Model loaded from {path}")

    def get_info(self) -> dict:
        return self.train_info
