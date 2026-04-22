"""Tests for the DNN model (Phase 6).

Architecture search is expensive, so these tests use a stripped-down
search space (1 hidden config, 1 dropout, 1 lr, 1 seed) — enough to
validate shape contracts and that training runs without crashing.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src import data
from src.models.dnn import AeroMLP, train, train_multi_seed


@pytest.fixture(scope="module")
def prepared():
    df     = data.assign_airfoil_ids(data.load_raw())
    splits = data.make_splits(df)
    x_sc, y_sc = data.fit_scalers(splits["train"])
    X_tr, y_tr = data.transform(splits["train"], x_sc, y_sc)
    X_va, y_va = data.transform(splits["val"],   x_sc, y_sc)
    X_te, y_te = data.transform(splits["test"],  x_sc, y_sc)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def test_aeromplp_forward_shape():
    model = AeroMLP(in_features=18, hidden=(64,), dropout=0.1)
    x     = torch.randn(10, 18)
    out   = model(x)
    assert out.shape == (10, 2)


def test_train_returns_model_and_meta(prepared):
    X_tr, y_tr, X_va, y_va, _, _ = prepared
    model, meta = train(
        X_tr, y_tr, X_va, y_va,
        seed=0,
        hidden_configs=((64, 64),),
        dropouts=(0.1,),
        learning_rates=(1e-3,),
    )
    assert isinstance(model, AeroMLP)
    assert set(meta.keys()) == {"hidden", "dropout", "lr", "seed", "val_rmse"}


def test_predict_shape(prepared):
    X_tr, y_tr, X_va, y_va, X_te, _ = prepared
    model, _ = train(
        X_tr, y_tr, X_va, y_va,
        seed=0,
        hidden_configs=((64, 64),),
        dropouts=(0.1,),
        learning_rates=(1e-3,),
    )
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_te).float()).numpy()
    assert preds.shape == (len(X_te), 2)


def test_val_rmse_positive(prepared):
    X_tr, y_tr, X_va, y_va, _, _ = prepared
    _, meta = train(
        X_tr, y_tr, X_va, y_va,
        seed=0,
        hidden_configs=((64, 64),),
        dropouts=(0.1,),
        learning_rates=(1e-3,),
    )
    assert meta["val_rmse"] > 0.0


def test_multi_seed_stats(prepared):
    X_tr, y_tr, X_va, y_va, _, _ = prepared
    models, stats = train_multi_seed(
        X_tr, y_tr, X_va, y_va,
        seeds=(0, 1),
    )
    # Run with defaults — just validate return contract, not quality
    assert len(models) == 2
    assert "mean_val_rmse" in stats
    assert "std_val_rmse"  in stats
    assert len(stats["per_seed"]) == 2
