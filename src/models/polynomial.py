"""Polynomial regression baseline: degree-d basis expansion + Ridge.

Hyperparameters selected by 5-fold CV on the (scaled) training set:
  - degree ∈ {1, 2, 3}
  - ridge alpha ∈ {1e-2, 1e-1, 1, 10}

Ridge handles multi-output regression natively, so both Cl and Cm are
predicted by a single pipeline. CV objective: mean neg-RMSE averaged
over both outputs (uniform_average, sklearn default).

Public API
----------
train(X, y) -> (pipeline, meta)
    Fit and return the best pipeline + a dict with degree/alpha/cv_rmse.
    X shape: (n, 18), y shape: (n, 2) — both already z-scored.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

DEGREES = (1, 2, 3)
ALPHAS  = (1e-2, 1e-1, 1.0, 10.0)


def _make_pipeline(degree: int, alpha: float) -> Pipeline:
    return Pipeline([
        ("poly",  PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge", Ridge(alpha=alpha)),
    ])


def train(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    degrees: tuple = DEGREES,
    alphas: tuple = ALPHAS,
) -> Tuple[Pipeline, Dict]:
    """Select best (degree, alpha) by 5-fold CV, refit on full X/y.

    Returns
    -------
    pipeline : fitted sklearn Pipeline (PolynomialFeatures → Ridge)
    meta     : {"degree": int, "alpha": float, "cv_rmse": float}
               cv_rmse is the mean RMSE across folds (positive, original scale).
    """
    best_score  = -np.inf   # neg-RMSE; higher (less negative) is better
    best_degree = degrees[0]
    best_alpha  = alphas[0]

    for degree in degrees:
        for alpha in alphas:
            pipe   = _make_pipeline(degree, alpha)
            scores = cross_val_score(
                pipe, X, y, cv=cv,
                scoring="neg_root_mean_squared_error",
            )
            mean_score = float(scores.mean())
            if mean_score > best_score:
                best_score  = mean_score
                best_degree = degree
                best_alpha  = alpha

    best_pipe = _make_pipeline(best_degree, best_alpha)
    best_pipe.fit(X, y)

    meta: Dict = {
        "degree":  best_degree,
        "alpha":   best_alpha,
        "cv_rmse": -best_score,   # positive RMSE
    }
    return best_pipe, meta
