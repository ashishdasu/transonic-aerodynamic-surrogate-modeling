"""Sanity tests on the raw CSV. Runnable today (Phase 0)."""
import pandas as pd

from src import data
from src.config import DATA_PATH, FEATURE_COLS, TARGET_COLS


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
    """There must be exactly 8 unique airfoil geometries in the data."""
    df = data.load_raw()
    unique = df[[c for c in df.columns if c.startswith(("y_U", "y_L"))]] \
        .drop_duplicates()
    assert len(unique) == 8, f"Expected 8 unique airfoils, got {len(unique)}"


# Phase 1 will add: test_splits_deterministic, test_rae2822_heldout_only,
# test_scalers_fit_on_train_only, test_no_row_in_multiple_splits.
