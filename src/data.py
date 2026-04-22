"""Data loading, airfoil identification, splitting, and scaling.

Pipeline exposed to callers:

    df         = load_raw()
    df         = assign_airfoil_ids(df)
    splits     = make_splits(df)           # dict: train/val/test/heldout
    x_sc, y_sc = fit_scalers(splits['train'])
    X, y       = transform(splits['train'], x_sc, y_sc)

The only external dependency is ``Input_Data.csv`` at the repo root.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_PATH, FEATURE_COLS, GEOMETRY_COLS, HELDOUT_AIRFOIL,
    SEED, TARGET_COLS, TEST_FRAC, VAL_FRAC,
)


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Read and schema-validate the CSV."""
    df = pd.read_csv(path)
    missing = set(FEATURE_COLS + TARGET_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    return df


def assign_airfoil_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 1: groupby on the 16 geometry columns to recover the 8 unique
    airfoils, then map each to its canonical name (NACA0012, RAE2822, ...)
    by matching reference profile coordinates. Adds an ``airfoil`` column.
    """
    raise NotImplementedError("Phase 1")


def make_splits(df: pd.DataFrame, seed: int = SEED) -> Dict[str, pd.DataFrame]:
    """Phase 1: returns {'train', 'val', 'test', 'heldout_rae2822'}.

    Invariants enforced (verified in tests/test_data.py):
      - heldout_rae2822 contains every RAE2822 row and nothing else.
      - train / val / test partition the remaining 7 airfoils.
      - split is deterministic in ``seed``.
      - no row appears in more than one split.
    """
    raise NotImplementedError("Phase 1")


def fit_scalers(train_df: pd.DataFrame) -> Tuple[StandardScaler, StandardScaler]:
    """Phase 1: z-score inputs, z-score targets, fit on training rows ONLY."""
    raise NotImplementedError("Phase 1")


def transform(df: pd.DataFrame, x_scaler, y_scaler):
    """Phase 1: apply fitted scalers and return (X, y) float32 arrays."""
    raise NotImplementedError("Phase 1")
