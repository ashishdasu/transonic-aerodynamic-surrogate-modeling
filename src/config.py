"""Authoritative configuration: paths, seeds, column names.

Everything else in the project imports from here. If you need to change a
path, a seed, or the feature/target schema, change it here once.
"""
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
DATA_PATH    = ROOT / "Input_Data.csv"
RESULTS_DIR  = ROOT / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
TABLES_DIR   = RESULTS_DIR / "tables"
MODELS_DIR   = RESULTS_DIR / "models"
REPORT_DIR   = ROOT / "report"

# ─── Schema ─────────────────────────────────────────────────────────────────
GEOMETRY_COLS = (
    [f"y_U{i}" for i in range(1, 9)]
    + [f"y_L{i}" for i in range(1, 9)]
)
FLOW_COLS    = ["alpha", "Mach"]
FEATURE_COLS = GEOMETRY_COLS + FLOW_COLS
TARGET_COLS  = ["Cl", "Cm"]

# ─── Dataset metadata ───────────────────────────────────────────────────────
# 8 airfoils, per Elrefaie 2024. Order here does not determine identity;
# src/data.py matches geometry tuples against reference profiles to name
# them. RAE2822 is held out from training.
AIRFOIL_NAMES = (
    "NACA0012", "NACA2412", "NACA4412",
    "NACA23012", "NACA24112", "NACA25112",
    "RAE2822",  "RAE5212",
)
HELDOUT_AIRFOIL = "RAE2822"

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED      = 42
# DNN trains across three seeds to report mean +/- std. Classical models
# are deterministic given random_state=SEED, so single-seed is honest.
DNN_SEEDS = (0, 1, 42)

# ─── Splits (applied after removing RAE2822) ────────────────────────────────
TEST_FRAC = 0.20
VAL_FRAC  = 0.20          # 0.20 of total -> 0.25 of (total - test)
