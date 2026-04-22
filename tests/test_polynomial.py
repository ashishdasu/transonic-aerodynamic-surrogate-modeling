"""Tests for the polynomial regression baseline (Phase 3)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src import data
from src.models.polynomial import ALPHAS, DEGREES, train


@pytest.fixture(scope="module")
def prepared():
    df     = data.assign_airfoil_ids(data.load_raw())
    splits = data.make_splits(df)
    x_sc, y_sc = data.fit_scalers(splits["train"])
    X_tr, y_tr = data.transform(splits["train"],  x_sc, y_sc)
    X_te, y_te = data.transform(splits["test"],   x_sc, y_sc)
    return X_tr, y_tr, X_te, y_te


def test_train_returns_pipeline_and_meta(prepared):
    X_tr, y_tr, _, _ = prepared
    pipe, meta = train(X_tr, y_tr)
    assert isinstance(pipe, Pipeline)
    assert set(meta.keys()) == {"degree", "alpha", "cv_rmse"}


def test_best_degree_in_search_space(prepared):
    X_tr, y_tr, _, _ = prepared
    _, meta = train(X_tr, y_tr)
    assert meta["degree"] in DEGREES


def test_best_alpha_in_search_space(prepared):
    X_tr, y_tr, _, _ = prepared
    _, meta = train(X_tr, y_tr)
    assert meta["alpha"] in ALPHAS


def test_predict_shape(prepared):
    X_tr, y_tr, X_te, _ = prepared
    pipe, _ = train(X_tr, y_tr)
    preds = pipe.predict(X_te)
    assert preds.shape == (len(X_te), 2)


def test_cv_rmse_positive(prepared):
    X_tr, y_tr, _, _ = prepared
    _, meta = train(X_tr, y_tr)
    assert meta["cv_rmse"] > 0.0


def test_train_rmse_below_naive_baseline(prepared):
    """Fitted model should beat predicting the mean (RMSE = 1.0 in z-score space)."""
    X_tr, y_tr, _, _ = prepared
    pipe, _ = train(X_tr, y_tr)
    preds = pipe.predict(X_tr)
    residuals = preds - y_tr
    rmse = float(np.sqrt((residuals ** 2).mean()))
    assert rmse < 1.0, f"Train RMSE {rmse:.4f} should be < 1.0 (z-score naive baseline)"


def test_deterministic(prepared):
    """Same inputs → same best params."""
    X_tr, y_tr, _, _ = prepared
    _, meta_a = train(X_tr, y_tr)
    _, meta_b = train(X_tr, y_tr)
    assert meta_a["degree"] == meta_b["degree"]
    assert meta_a["alpha"]  == meta_b["alpha"]
