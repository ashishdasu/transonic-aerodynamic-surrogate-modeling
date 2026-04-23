"""Data loading, airfoil identification, splitting, and scaling.

Pipeline exposed to callers:

    df                 = load_raw()
    df                 = assign_airfoil_ids(df)
    splits             = make_splits(df)          # train/val/test/heldout
    x_scaler, y_scaler = fit_scalers(splits["train"])
    X, y               = transform(splits["train"], x_scaler, y_scaler)

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


# ─── Airfoil identification by coordinate signature ─────────────────────
#
# The CSV has no airfoil-name column. Each of the eight profiles appears
# ~170 times with identical y-values; we recover identity by matching
# each row's 6-coordinate fingerprint (y_U1, y_U4, y_U8, y_L1, y_L4, y_L8)
# against the table below.
#
# Five identifications are physics-confident:
#   NACA0012 — only symmetric profile (y_U[i] == -y_L[i] everywhere).
#   RAE2822  — supercritical, 1.3% max camber + reflexed TE (y_L8 ~ 0).
#   NACA2412 — 4-digit with 2% max camber (matches observed 0.0199).
#   NACA4412 — 4-digit with 4% max camber (matches observed 0.0399).
#   RAE5212  — only other reflexed-TE profile (y_L8 = -0.0016 ≈ 0);
#              moderate camber (~1.8%) consistent with published data.
#
# The three NACA 5-digit assignments (23012 / 24112 / 25112) are
# ordered by max-camber-peak station position: 23012 peaks earliest
# (x ≈ 0.15), 25112 latest (x ≈ 0.25). Phase 2 EDA overlays each
# sampled profile on published reference coordinates; this lookup is
# corrected there if any of the three NACA 5-digit labels are swapped.
_SIG_KEYS = ("y_U1", "y_U4", "y_U8", "y_L1", "y_L4", "y_L8")

_SIG_TO_NAME: Dict[Tuple[float, ...], str] = {
    # confident
    (0.0485, 0.0562, 0.0164, -0.0485, -0.0562, -0.0164): "NACA0012",
    (0.0401, 0.0628, 0.0215, -0.0405, -0.0559, -0.0017): "RAE2822",
    (0.0587, 0.0761, 0.0236, -0.0385, -0.0362, -0.0093): "NACA2412",
    (0.0690, 0.0961, 0.0308, -0.0289, -0.0163, -0.0024): "NACA4412",
    (0.0538, 0.0722, 0.0244, -0.0348, -0.0445, -0.0016): "RAE5212",
    # tentative (3 NACA 5-digit variants — validate in Phase 2 EDA)
    (0.0665, 0.0686, 0.0190, -0.0306, -0.0438, -0.0138): "NACA23012",
    (0.0691, 0.0760, 0.0169, -0.0283, -0.0366, -0.0160): "NACA24112",
    (0.0697, 0.0714, 0.0171, -0.0275, -0.0412, -0.0158): "NACA25112",
}


# ─── Loading ─────────────────────────────────────────────────────────────
def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Read and schema-validate the CSV."""
    df = pd.read_csv(path)
    missing = set(FEATURE_COLS + TARGET_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    return df


# ─── Airfoil identification ──────────────────────────────────────────────
def assign_airfoil_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with an ``airfoil`` column populated by
    matching each row's 6-coordinate signature against ``_SIG_TO_NAME``.
    Raises if any row fails to match."""
    df = df.copy()
    sig = df[list(_SIG_KEYS)].round(4).apply(tuple, axis=1)
    df["airfoil"] = sig.map(_SIG_TO_NAME)
    unmatched = df["airfoil"].isna().sum()
    if unmatched:
        example = df[df["airfoil"].isna()].iloc[0][list(_SIG_KEYS)].to_dict()
        raise ValueError(
            f"{unmatched} row(s) did not match any known airfoil "
            f"signature. Example: {example}"
        )
    return df


# ─── Splitting ───────────────────────────────────────────────────────────
def make_splits(df: pd.DataFrame, seed: int = SEED) -> Dict[str, pd.DataFrame]:
    """Partition into train / val / test / heldout_rae2822.

    RAE2822 rows are carved out BEFORE splitting so the model never sees
    that geometry during training. This tests extrapolation to an unseen
    airfoil family. Contrast with make_paper_splits(), which pools all 8
    airfoils and tests interpolation only (matches Elrefaie et al. 2024).

    Invariants (tests/test_data.py):
      - heldout_rae2822 contains every RAE2822 row and nothing else.
      - train / val / test partition the remaining seven airfoils.
      - ratios within 2 pp of 60/20/20.
      - deterministic in ``seed``.
      - no row appears in more than one split.
    """
    if "airfoil" not in df.columns:
        raise ValueError(
            "df must have an 'airfoil' column; call assign_airfoil_ids first"
        )

    heldout = df[df["airfoil"] == HELDOUT_AIRFOIL].copy()
    pool    = df[df["airfoil"] != HELDOUT_AIRFOIL].copy()

    train_val, test = train_test_split(
        pool, test_size=TEST_FRAC, random_state=seed,
    )
    # VAL_FRAC is fraction of the WHOLE pool; relative to train_val it's
    # VAL_FRAC / (1 - TEST_FRAC).
    val_relative = VAL_FRAC / (1.0 - TEST_FRAC)
    train, val = train_test_split(
        train_val, test_size=val_relative, random_state=seed,
    )

    return {
        "train":           train.reset_index(drop=True),
        "val":             val.reset_index(drop=True),
        "test":            test.reset_index(drop=True),
        "heldout_rae2822": heldout.reset_index(drop=True),
    }


def make_paper_splits(df: pd.DataFrame, seed: int = SEED) -> Dict[str, pd.DataFrame]:
    """Partition all 8 airfoils (including RAE2822) into train/val/test 60/20/20.

    Replicates the evaluation paradigm of Elrefaie et al. (2024), who used
    random splits across all airfoils. Because RAE2822 rows appear in both
    train and test, this evaluates interpolation within known geometries —
    not extrapolation to an unseen profile. Use make_splits() for the
    harder geometry-holdout evaluation.
    """
    train_val, test = train_test_split(df, test_size=TEST_FRAC, random_state=seed)
    val_relative = VAL_FRAC / (1.0 - TEST_FRAC)
    train, val = train_test_split(train_val, test_size=val_relative, random_state=seed)
    return {
        "train": train.reset_index(drop=True),
        "val":   val.reset_index(drop=True),
        "test":  test.reset_index(drop=True),
    }


# ─── Scaling ─────────────────────────────────────────────────────────────
def fit_scalers(
    train_df: pd.DataFrame,
) -> Tuple[StandardScaler, StandardScaler]:
    """Fit z-score scalers on the training split ONLY.

    Features and targets scaled independently. Returned scalers must be
    applied with ``transform()`` to val / test / heldout.
    """
    x_scaler = StandardScaler().fit(train_df[FEATURE_COLS].values)
    y_scaler = StandardScaler().fit(train_df[TARGET_COLS].values)
    return x_scaler, y_scaler


def transform(
    df: pd.DataFrame,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply fitted scalers. Returns ``(X, y)`` float32 arrays."""
    X = x_scaler.transform(df[FEATURE_COLS].values).astype(np.float32)
    y = y_scaler.transform(df[TARGET_COLS].values).astype(np.float32)
    return X, y
