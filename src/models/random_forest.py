"""Multi-output Random Forest regressor (sklearn).

Hyperparameters selected by 5-fold CV on the scaled training set:
  - n_estimators     ∈ {50, 100, 200}
  - max_depth        ∈ {5, 10, None}
  - min_samples_split ∈ {2, 5}

RandomForestRegressor handles multi-output regression natively, so both
Cl and Cm share one model. CV objective: neg-RMSE (uniform_average).

Public API
----------
train(X, y) -> (model, meta)
    model : fitted RandomForestRegressor
    meta  : {"n_estimators", "max_depth", "min_samples_split", "cv_rmse"}
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

N_ESTIMATORS      = (50, 100, 200)
MAX_DEPTHS        = (5, 10, None)
MIN_SAMPLES_SPLIT = (2, 5)


def train(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    n_estimators: tuple = N_ESTIMATORS,
    max_depths: tuple = MAX_DEPTHS,
    min_samples_splits: tuple = MIN_SAMPLES_SPLIT,
    random_state: int = 42,
    n_jobs: int = 1,
) -> Tuple[RandomForestRegressor, Dict]:
    """Select best hyperparams by 5-fold CV, refit on full X/y.

    n_jobs=1 to avoid worker conflicts when running after other models.
    Returns fitted model and meta dict with best params and cv_rmse.
    """
    best_score            = -np.inf
    best_n_estimators     = n_estimators[0]
    best_max_depth        = max_depths[0]
    best_min_samples      = min_samples_splits[0]

    for n in n_estimators:
        for d in max_depths:
            for m in min_samples_splits:
                rf = RandomForestRegressor(
                    n_estimators=n,
                    max_depth=d,
                    min_samples_split=m,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )
                scores = cross_val_score(
                    rf, X, y, cv=cv,
                    scoring="neg_root_mean_squared_error",
                )
                mean_score = float(scores.mean())
                if mean_score > best_score:
                    best_score        = mean_score
                    best_n_estimators = n
                    best_max_depth    = d
                    best_min_samples  = m

    best_model = RandomForestRegressor(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_samples_split=best_min_samples,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    best_model.fit(X, y)

    meta: Dict = {
        "n_estimators":      best_n_estimators,
        "max_depth":         best_max_depth,
        "min_samples_split": best_min_samples,
        "cv_rmse":           -best_score,
    }
    return best_model, meta
