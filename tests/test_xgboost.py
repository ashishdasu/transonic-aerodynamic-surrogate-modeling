"""Tests for the XGBoost model (Phase 5)."""
from __future__ import annotations

import numpy as np
import pytest

from src import data
from src.models.xgboost_model import (
    LEARNING_RATES, MAX_DEPTHS, N_ESTIMATORS, SUBSAMPLES,
    XGBoostMultiOutput, train,
)


@pytest.fixture(scope="module")
def prepared():
    df     = data.assign_airfoil_ids(data.load_raw())
    splits = data.make_splits(df)
    x_sc, y_sc = data.fit_scalers(splits["train"])
    X_tr, y_tr = data.transform(splits["train"], x_sc, y_sc)
    X_te, y_te = data.transform(splits["test"],  x_sc, y_sc)
    return X_tr, y_tr, X_te, y_te


@pytest.fixture(scope="module")
def trained(prepared):
    """Run grid search once; reuse across all tests in this module."""
    X_tr, y_tr, _, _ = prepared
    return train(X_tr, y_tr)


def test_train_returns_wrapper_and_meta(trained):
    wrapper, meta = trained
    assert isinstance(wrapper, XGBoostMultiOutput)
    assert set(meta.keys()) == {"Cl", "Cm"}


def test_meta_has_expected_keys(trained):
    _, meta = trained
    for t in ("Cl", "Cm"):
        assert set(meta[t].keys()) == {
            "learning_rate", "max_depth", "n_estimators", "subsample", "cv_rmse"
        }


def test_best_params_in_grid(trained):
    _, meta = trained
    for t in ("Cl", "Cm"):
        assert meta[t]["learning_rate"] in LEARNING_RATES
        assert meta[t]["max_depth"]     in MAX_DEPTHS
        assert meta[t]["n_estimators"]  in N_ESTIMATORS
        assert meta[t]["subsample"]     in SUBSAMPLES


def test_predict_shape(prepared, trained):
    _, _, X_te, _ = prepared
    wrapper, _    = trained
    preds = wrapper.predict(X_te)
    assert preds.shape == (len(X_te), 2)


def test_cv_rmse_positive(trained):
    _, meta = trained
    for t in ("Cl", "Cm"):
        assert meta[t]["cv_rmse"] > 0.0


def test_train_rmse_well_below_naive(prepared, trained):
    """XGBoost should beat z-score naive baseline by a clear margin."""
    X_tr, y_tr, _, _ = prepared
    wrapper, _        = trained
    preds = wrapper.predict(X_tr)
    rmse  = float(np.sqrt(((preds - y_tr) ** 2).mean()))
    assert rmse < 0.3, f"Train RMSE {rmse:.4f} should be < 0.3"
