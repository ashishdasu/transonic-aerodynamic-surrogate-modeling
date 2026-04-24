"""XGBoost regressor, one model per target (Cl, Cm).

XGBoost does not natively support multi-output, so we fit two independent
XGBRegressors and wrap them in a thin sklearn-compatible class whose
.predict(X) returns an (n, 2) array [Cl, Cm].

Hyperparameters selected by 5-fold CV per target:
  - learning_rate ∈ {0.01, 0.05, 0.1}
  - max_depth     ∈ {3, 5, 7}
  - n_estimators  ∈ {100, 300, 500}
  - subsample     ∈ {0.8, 1.0}

Public API
----------
train(X, y) -> (XGBoostMultiOutput, meta)
    X shape: (n, 18), y shape: (n, 2) — both already z-scored.
    meta: {"Cl": {best_params}, "Cm": {best_params}}
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

LEARNING_RATES = (0.01, 0.05, 0.1)
MAX_DEPTHS     = (3, 5, 7)
N_ESTIMATORS   = (100, 300, 500)
SUBSAMPLES     = (0.8, 1.0)


class XGBoostMultiOutput:
    """Thin wrapper that holds one XGBRegressor per target."""

    def __init__(
        self,
        models: Dict[str, XGBRegressor],
        meta: Dict[str, Dict],
        target_cols: Tuple[str, ...] = ("Cl", "Cm"),
    ) -> None:
        self.models_      = models
        self.meta_        = meta
        self.target_cols_ = target_cols

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return (n, 2) predictions in the same column order as target_cols_."""
        preds = [self.models_[t].predict(X) for t in self.target_cols_]
        return np.column_stack(preds)


def _best_params_for_target(
    X: np.ndarray,
    y_1d: np.ndarray,
    cv: int,
    learning_rates: tuple,
    max_depths: tuple,
    n_estimators: tuple,
    subsamples: tuple,
    random_state: int,
) -> Dict:
    best_score = -np.inf
    best = {
        "learning_rate": learning_rates[0],
        "max_depth":     max_depths[0],
        "n_estimators":  n_estimators[0],
        "subsample":     subsamples[0],
    }
    for lr in learning_rates:
        for d in max_depths:
            for n in n_estimators:
                for ss in subsamples:
                    xgb = XGBRegressor(
                        learning_rate=lr,
                        max_depth=d,
                        n_estimators=n,
                        subsample=ss,
                        random_state=random_state,
                        tree_method="hist",
                        verbosity=0,
                    )
                    scores = cross_val_score(
                        xgb, X, y_1d, cv=cv,
                        scoring="neg_root_mean_squared_error",
                    )
                    mean_score = float(scores.mean())
                    if mean_score > best_score:
                        best_score = mean_score
                        best = {
                            "learning_rate": lr,
                            "max_depth":     d,
                            "n_estimators":  n,
                            "subsample":     ss,
                            "cv_rmse":       -mean_score,
                        }
    return best


def train(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    learning_rates: tuple = LEARNING_RATES,
    max_depths: tuple = MAX_DEPTHS,
    n_estimators: tuple = N_ESTIMATORS,
    subsamples: tuple = SUBSAMPLES,
    random_state: int = 42,
    target_cols: Tuple[str, ...] = ("Cl", "Cm"),
) -> Tuple[XGBoostMultiOutput, Dict]:
    """CV-tune and fit one XGBRegressor per target, return wrapper + meta."""
    models: Dict[str, XGBRegressor] = {}
    meta:   Dict[str, Dict]         = {}

    for i, target in enumerate(target_cols):
        y_1d   = y[:, i]
        params = _best_params_for_target(
            X, y_1d, cv,
            learning_rates, max_depths, n_estimators, subsamples, random_state,
        )
        cv_rmse = params.pop("cv_rmse")

        xgb = XGBRegressor(
            **params,
            random_state=random_state,
            tree_method="hist",
            verbosity=0,
        )
        xgb.fit(X, y_1d)
        models[target]  = xgb
        meta[target]    = {**params, "cv_rmse": cv_rmse}

    wrapper = XGBoostMultiOutput(models=models, meta=meta, target_cols=target_cols)
    return wrapper, meta
