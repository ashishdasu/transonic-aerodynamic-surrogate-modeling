"""Tests for the evaluation module (Phase 7)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src import data, evaluate
from src.models.dnn import AeroMLP
from src.models.polynomial import train as poly_train


@pytest.fixture(scope="module")
def prepared():
    df     = data.assign_airfoil_ids(data.load_raw())
    splits = data.make_splits(df)
    x_sc, y_sc = data.fit_scalers(splits["train"])
    X_tr, y_tr = data.transform(splits["train"], x_sc, y_sc)
    X_te, y_te = data.transform(splits["test"],  x_sc, y_sc)
    return X_tr, y_tr, X_te, y_te, splits, y_sc


@pytest.fixture(scope="module")
def poly_fitted(prepared):
    X_tr, y_tr, *_ = prepared
    model, _ = poly_train(X_tr, y_tr)
    return model


def test_predict_sklearn_shape(prepared, poly_fitted):
    _, _, X_te, _, _, _ = prepared
    preds = evaluate.predict(poly_fitted, X_te)
    assert preds.shape == (len(X_te), 2)


def test_predict_dnn_shape(prepared):
    _, _, X_te, _, _, _ = prepared
    model = AeroMLP(18, (64,), 0.0)
    model.eval()
    preds = evaluate.predict(model, X_te)
    assert preds.shape == (len(X_te), 2)


def test_compute_metrics_keys(prepared, poly_fitted):
    _, _, X_te, y_te, _, y_sc = prepared
    y_true = y_sc.inverse_transform(y_te)
    y_pred = y_sc.inverse_transform(evaluate.predict(poly_fitted, X_te))
    metrics = evaluate.compute_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"Cl", "Cm"}
    for t in ("Cl", "Cm"):
        assert set(metrics[t].keys()) == {"rmse", "r2"}


def test_compute_metrics_rmse_positive(prepared, poly_fitted):
    _, _, X_te, y_te, _, y_sc = prepared
    y_true = y_sc.inverse_transform(y_te)
    y_pred = y_sc.inverse_transform(evaluate.predict(poly_fitted, X_te))
    metrics = evaluate.compute_metrics(y_true, y_pred)
    for t in ("Cl", "Cm"):
        assert metrics[t]["rmse"] > 0.0


def test_plot_parity_returns_path(tmp_path, prepared, poly_fitted):
    _, _, X_te, y_te, _, y_sc = prepared
    y_true = y_sc.inverse_transform(y_te)
    y_pred = y_sc.inverse_transform(evaluate.predict(poly_fitted, X_te))
    path   = evaluate.plot_parity(y_true, y_pred, "Polynomial",
                                   save_path=tmp_path / "parity.png")
    assert path.exists()


def test_plot_residuals_by_mach_returns_path(tmp_path, prepared, poly_fitted):
    _, _, X_te, y_te, splits, y_sc = prepared
    mach   = splits["test"]["Mach"].values
    y_true = y_sc.inverse_transform(y_te)
    y_pred = y_sc.inverse_transform(evaluate.predict(poly_fitted, X_te))
    path   = evaluate.plot_residuals_by_mach(
        y_true, y_pred, mach, "Polynomial",
        save_path=tmp_path / "res_mach.png",
    )
    assert path.exists()


def test_benchmark_latency_shape(prepared, poly_fitted):
    _, _, X_te, _, _, _ = prepared
    df = evaluate.benchmark_latency({"Polynomial": poly_fitted}, X_te)
    assert list(df.columns) == ["Model", "Mean (ms)", "Std (ms)"]
    assert len(df) == 1
