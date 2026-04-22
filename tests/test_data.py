"""Sanity tests on the raw CSV and the Phase 1 data pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data
from src.config import (
    DATA_PATH, FEATURE_COLS, HELDOUT_AIRFOIL, SEED, TARGET_COLS,
    TEST_FRAC, VAL_FRAC,
)


# ─── Raw CSV ─────────────────────────────────────────────────────────────
def test_csv_exists():
    assert DATA_PATH.exists(), f"Data file missing at {DATA_PATH}"


def test_load_raw_schema():
    df = data.load_raw()
    assert len(df) == 1362, f"Expected 1362 rows, got {len(df)}"
    for col in FEATURE_COLS + TARGET_COLS:
        assert col in df.columns, f"Missing column: {col}"


def test_no_nans():
    df = data.load_raw()
    assert not df[FEATURE_COLS + TARGET_COLS].isna().any().any()


def test_eight_unique_airfoils():
    df = data.load_raw()
    geom = df[[c for c in df.columns if c.startswith(("y_U", "y_L"))]]
    assert geom.drop_duplicates().shape[0] == 8


# ─── Airfoil identification ──────────────────────────────────────────────
def test_assign_airfoil_ids_full_coverage():
    df = data.assign_airfoil_ids(data.load_raw())
    assert df["airfoil"].isna().sum() == 0
    assert df["airfoil"].nunique() == 8
    assert HELDOUT_AIRFOIL in set(df["airfoil"].unique())


def test_naca0012_is_symmetric():
    """NACA0012 labeling sanity: all y_U[i] == -y_L[i]."""
    df = data.assign_airfoil_ids(data.load_raw())
    sym_rows = df[df["airfoil"] == "NACA0012"]
    assert len(sym_rows) > 0
    for i in range(1, 9):
        np.testing.assert_allclose(
            sym_rows[f"y_U{i}"].values,
            -sym_rows[f"y_L{i}"].values,
            atol=1e-6,
        )


def test_rae2822_has_expected_thickness_and_camber():
    """RAE2822 sanity: ~12% thickness, ~1.3% max camber."""
    df = data.assign_airfoil_ids(data.load_raw())
    r = df[df["airfoil"] == "RAE2822"].iloc[0]
    yU = np.array([r[f"y_U{i}"] for i in range(1, 9)])
    yL = np.array([r[f"y_L{i}"] for i in range(1, 9)])
    thickness = (yU - yL).max()
    camber    = ((yU + yL) / 2).max()
    assert 0.11 < thickness < 0.13, f"RAE2822 thickness off: {thickness:.4f}"
    assert 0.010 < camber < 0.020, f"RAE2822 max camber off: {camber:.4f}"


# ─── Splits ──────────────────────────────────────────────────────────────
def test_splits_deterministic():
    df = data.assign_airfoil_ids(data.load_raw())
    a = data.make_splits(df, seed=SEED)
    b = data.make_splits(df, seed=SEED)
    for key in a:
        pd.testing.assert_frame_equal(a[key], b[key])


def test_rae2822_only_in_heldout():
    df = data.assign_airfoil_ids(data.load_raw())
    splits = data.make_splits(df)
    assert set(splits["heldout_rae2822"]["airfoil"].unique()) == {HELDOUT_AIRFOIL}
    for key in ("train", "val", "test"):
        assert HELDOUT_AIRFOIL not in set(splits[key]["airfoil"].unique())


def test_split_ratios_within_tolerance():
    df = data.assign_airfoil_ids(data.load_raw())
    splits = data.make_splits(df)
    pool = len(df) - len(splits["heldout_rae2822"])
    assert abs(len(splits["train"]) / pool - (1 - TEST_FRAC - VAL_FRAC)) < 0.02
    assert abs(len(splits["val"])   / pool - VAL_FRAC)  < 0.02
    assert abs(len(splits["test"])  / pool - TEST_FRAC) < 0.02


def test_splits_partition_dataset():
    """train + val + test + heldout = entire df, no overlap."""
    df = data.assign_airfoil_ids(data.load_raw())
    splits = data.make_splits(df)
    total = sum(len(s) for s in splits.values())
    assert total == len(df)


def test_no_row_in_multiple_splits():
    """Use the index of the original df as row identity."""
    df = data.assign_airfoil_ids(data.load_raw()).reset_index(drop=True)
    heldout_mask = df["airfoil"] == HELDOUT_AIRFOIL
    pool_df = df[~heldout_mask].reset_index(drop=True)

    splits = data.make_splits(df)
    # reconstruct identity via a stable row hash on features+targets
    def row_key(r):
        return tuple(r[c] for c in FEATURE_COLS + TARGET_COLS)

    seen = set()
    for key in ("train", "val", "test", "heldout_rae2822"):
        for _, r in splits[key].iterrows():
            k = row_key(r)
            assert k not in seen, f"Row appears in multiple splits: {key}"
            seen.add(k)


# ─── Scalers ─────────────────────────────────────────────────────────────
def test_scalers_fit_on_train_only():
    df = data.assign_airfoil_ids(data.load_raw())
    splits = data.make_splits(df)
    x_scaler, y_scaler = data.fit_scalers(splits["train"])
    np.testing.assert_allclose(
        x_scaler.mean_,
        splits["train"][FEATURE_COLS].mean().values,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        y_scaler.mean_,
        splits["train"][TARGET_COLS].mean().values,
        rtol=1e-5,
    )


def test_transform_roundtrip():
    df = data.assign_airfoil_ids(data.load_raw())
    splits = data.make_splits(df)
    x_scaler, y_scaler = data.fit_scalers(splits["train"])
    X, y = data.transform(splits["train"], x_scaler, y_scaler)

    X_inv = x_scaler.inverse_transform(X)
    y_inv = y_scaler.inverse_transform(y)
    # float32 intermediate causes ~3e-7 absolute error near zero
    np.testing.assert_allclose(
        X_inv, splits["train"][FEATURE_COLS].values, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        y_inv, splits["train"][TARGET_COLS].values, rtol=1e-5, atol=1e-6
    )


def test_transformed_train_has_zero_mean_unit_std():
    df = data.assign_airfoil_ids(data.load_raw())
    splits = data.make_splits(df)
    x_scaler, y_scaler = data.fit_scalers(splits["train"])
    X, y = data.transform(splits["train"], x_scaler, y_scaler)
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(X.std(axis=0),  1, atol=1e-5)
    np.testing.assert_allclose(y.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(y.std(axis=0),  1, atol=1e-5)
