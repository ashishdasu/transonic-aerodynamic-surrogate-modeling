"""Tests for the Random Forest model (Phase 4)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from src import data
from src.models.random_forest import MAX_DEPTHS, MIN_SAMPLES_SPLIT, N_ESTIMATORS, train


@pytest.fixture(scope="module")
def prepared():
    df     = data.assign_airfoil_ids(data.load_raw())
    splits = data.make_splits(df)
    x_sc, y_sc = data.fit_scalers(splits["train"])
    X_tr, y_tr = data.transform(splits["train"], x_sc, y_sc)
    X_te, y_te = data.transform(splits["test"],  x_sc, y_sc)
    return X_tr, y_tr, X_te, y_te


def test_train_returns_rf_and_meta(prepared):
    X_tr, y_tr, _, _ = prepared
    model, meta = train(X_tr, y_tr)
    assert isinstance(model, RandomForestRegressor)
    assert set(meta.keys()) == {"n_estimators", "max_depth", "min_samples_split", "cv_rmse"}


def test_best_n_estimators_in_grid(prepared):
    X_tr, y_tr, _, _ = prepared
    _, meta = train(X_tr, y_tr)
    assert meta["n_estimators"] in N_ESTIMATORS


def test_best_max_depth_in_grid(prepared):
    X_tr, y_tr, _, _ = prepared
    _, meta = train(X_tr, y_tr)
    assert meta["max_depth"] in MAX_DEPTHS


def test_best_min_samples_in_grid(prepared):
    X_tr, y_tr, _, _ = prepared
    _, meta = train(X_tr, y_tr)
    assert meta["min_samples_split"] in MIN_SAMPLES_SPLIT


def test_predict_shape(prepared):
    X_tr, y_tr, X_te, _ = prepared
    model, _ = train(X_tr, y_tr)
    preds = model.predict(X_te)
    assert preds.shape == (len(X_te), 2)


def test_cv_rmse_positive(prepared):
    X_tr, y_tr, _, _ = prepared
    _, meta = train(X_tr, y_tr)
    assert meta["cv_rmse"] > 0.0


def test_train_rmse_well_below_naive(prepared):
    """RF should beat z-score naive baseline (RMSE=1) by a clear margin."""
    X_tr, y_tr, _, _ = prepared
    model, _ = train(X_tr, y_tr)
    preds    = model.predict(X_tr)
    rmse     = float(np.sqrt(((preds - y_tr) ** 2).mean()))
    assert rmse < 0.5, f"Train RMSE {rmse:.4f} should be well below 0.5"
