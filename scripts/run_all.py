"""Single entry point invoked by the Makefile.

    python scripts/run_all.py --stage {eda,train,eval,all}

Kept deliberately thin: each stage delegates to src/. No logic here that
could hide in a non-obvious place.
"""
from __future__ import annotations

# Must be set before ANY numerical library is imported.
# XGBoost and PyTorch both use OpenMP; on macOS their OMP runtimes conflict
# in __kmp_join_barrier when both are loaded in the same process. Forcing
# single-threaded OMP eliminates the deadlock and segfault.
import os
os.environ["OMP_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]   = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_eda() -> None:
    """Regenerate all EDA figures under results/figures/eda/."""
    from src import data, eda

    df     = data.assign_airfoil_ids(data.load_raw())
    splits = data.make_splits(df)

    print("[eda] airfoil_shapes             ...", end=" ", flush=True)
    print(eda.plot_airfoil_shapes(df).name)

    print("[eda] target_dist_by_split       ...", end=" ", flush=True)
    print(eda.plot_target_distributions_by_split(splits).name)

    print("[eda] target_dist_by_airfoil     ...", end=" ", flush=True)
    print(eda.plot_target_distributions_by_airfoil(df).name)

    print("[eda] cl_vs_alpha                ...", end=" ", flush=True)
    print(eda.plot_coefficient_vs_alpha(df, "Cl").name)

    print("[eda] cm_vs_alpha                ...", end=" ", flush=True)
    print(eda.plot_coefficient_vs_alpha(df, "Cm").name)

    print("[eda] correlation_heatmap        ...", end=" ", flush=True)
    print(eda.plot_correlation_heatmap(df).name)


def run_train() -> None:
    """Fit all four models; saves artifacts to results/models/."""
    from src.train import run_train as _train
    _train()


def run_eval() -> None:
    """Generate all evaluation figures and LaTeX metric tables."""
    import pickle

    import numpy as np
    import torch

    from src import data, evaluate
    from src.config import MODELS_DIR, TABLES_DIR
    from src.models.dnn import AeroMLP
    from src.models.xgboost_model import XGBoostMultiOutput

    # ── Load data ──────────────────────────────────────────────────────────
    print("[eval] loading data …")
    df     = data.assign_airfoil_ids(data.load_raw())
    splits = data.make_splits(df)

    with open(MODELS_DIR / "scalers.pkl", "rb") as f:
        x_sc, y_sc = pickle.load(f)

    X_te, y_te_sc  = data.transform(splits["test"],            x_sc, y_sc)
    X_ho, y_ho_sc  = data.transform(splits["heldout_rae2822"], x_sc, y_sc)

    # Inverse-transform targets to original coefficient space
    y_te  = y_sc.inverse_transform(y_te_sc)
    y_ho  = y_sc.inverse_transform(y_ho_sc)

    # ── Load models ────────────────────────────────────────────────────────
    print("[eval] loading models …")
    with open(MODELS_DIR / "polynomial.pkl",   "rb") as f:
        poly_model = pickle.load(f)
    with open(MODELS_DIR / "random_forest.pkl", "rb") as f:
        rf_model   = pickle.load(f)
    with open(MODELS_DIR / "xgboost.pkl",       "rb") as f:
        xgb_model  = pickle.load(f)

    # Rebuild DNN from saved state_dicts; use first seed for single-model ops
    from src.config import DNN_SEEDS, FEATURE_COLS
    # Infer hidden from saved meta
    import json
    with open(MODELS_DIR / "train_meta.json") as f:
        train_meta = json.load(f)
    dnn_per_seed = train_meta["dnn"]["per_seed"]

    dnn_models = []
    for i, seed in enumerate(DNN_SEEDS):
        hidden = tuple(dnn_per_seed[i]["hidden"])
        model  = AeroMLP(len(FEATURE_COLS), hidden, dnn_per_seed[i]["dropout"])
        model.load_state_dict(torch.load(MODELS_DIR / f"dnn_seed{seed}.pt", weights_only=True))
        model.eval()
        dnn_models.append(model)

    named_models = {
        "Polynomial":    poly_model,
        "Random Forest": rf_model,
        "XGBoost":       xgb_model,
        "DNN":           dnn_models[0],   # seed-0 representative for figures
    }

    # ── Metrics tables ────────────────────────────────────────────────────
    print("[eval] computing metrics …")
    test_metrics = {}
    ho_metrics   = {}
    for name, model in named_models.items():
        y_te_pred_sc = evaluate.predict(model, X_te)
        y_ho_pred_sc = evaluate.predict(model, X_ho)
        y_te_pred    = y_sc.inverse_transform(y_te_pred_sc)
        y_ho_pred    = y_sc.inverse_transform(y_ho_pred_sc)
        test_metrics[name] = evaluate.compute_metrics(y_te, y_te_pred)
        ho_metrics[name]   = evaluate.compute_metrics(y_ho, y_ho_pred)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    p1 = evaluate.save_metrics_table(test_metrics, TABLES_DIR / "test_metrics.tex")
    p2 = evaluate.save_metrics_table(ho_metrics,   TABLES_DIR / "heldout_metrics.tex")
    print(f"[eval] metrics → {p1.name}, {p2.name}")

    # ── Parity plots (individual + combined) ─────────────────────────────
    print("[eval] parity plots …")
    te_preds_all = {}
    for name, model in named_models.items():
        y_te_pred = y_sc.inverse_transform(evaluate.predict(model, X_te))
        te_preds_all[name] = y_te_pred
        path = evaluate.plot_parity(y_te, y_te_pred, name)
        print(f"[eval]   {path.name}")
    path = evaluate.plot_parity_all(y_te, te_preds_all)
    print(f"[eval]   {path.name}")

    # ── Prediction KDE distributions ──────────────────────────────────────
    print("[eval] prediction KDE distributions …")
    path = evaluate.plot_prediction_distributions(y_te, te_preds_all)
    print(f"[eval]   {path.name}")

    # ── Residuals by Mach (individual + combined) ─────────────────────
    print("[eval] residuals by Mach …")
    mach_te = splits["test"]["Mach"].values
    mach_ho = splits["heldout_rae2822"]["Mach"].values
    for name, model in named_models.items():
        path = evaluate.plot_residuals_by_mach(y_te, te_preds_all[name], mach_te, name)
        print(f"[eval]   {path.name}")
    path = evaluate.plot_residuals_mach_all(y_te, te_preds_all, mach_te)
    print(f"[eval]   {path.name}")

    # ── Cm vs alpha overlay (RAE2822 held-out slice) ──────────────────────
    print("[eval] Cm vs alpha …")
    ho_df = splits["heldout_rae2822"].copy()
    ho_df.index = range(len(ho_df))
    ho_preds = {}
    for name, model in named_models.items():
        ho_preds[name] = y_sc.inverse_transform(evaluate.predict(model, X_ho))
    for mach in sorted(ho_df["Mach"].unique()):
        path = evaluate.plot_cm_vs_alpha(ho_preds, ho_df, mach=mach)
        print(f"[eval]   {path.name}")

    # ── DNN training curves ───────────────────────────────────────────────
    print("[eval] DNN training curves …")
    if all("val_history" in m for m in train_meta["dnn"]["per_seed"]):
        path = evaluate.plot_dnn_training_curves(train_meta["dnn"]["per_seed"])
        print(f"[eval]   {path.name}")
    else:
        print("[eval]   skipped (re-run make train to generate val_history)")

    # ── Feature importance ────────────────────────────────────────────────
    print("[eval] feature importance …")
    path = evaluate.plot_feature_importance(rf_model, xgb_model)
    print(f"[eval]   {path.name}")

    # ── Latency benchmark ─────────────────────────────────────────────────
    print("[eval] latency benchmark …")
    lat_df = evaluate.benchmark_latency(named_models, X_te)
    lat_path = evaluate.save_latency_table(lat_df, TABLES_DIR / "latency.tex")
    print(f"[eval] latency → {lat_path.name}")
    print(lat_df.to_string(index=False))


def run_analysis() -> None:
    """Run four supplementary analyses (LOAO, learning curves, Mach extrap, ablation).

    Requires trained models from make train and scalers from make eval.
    Outputs go to results/tables/ and results/figures/analysis/.
    Runtime: ~20–30 min (LOAO dominates; DNN excluded from LOAO).
    """
    import json
    import pickle

    import torch

    from src import data, analysis
    from src.config import MODELS_DIR

    print("[analysis] loading data and models …")
    df     = data.assign_airfoil_ids(data.load_raw())
    splits = data.make_splits(df)

    with open(MODELS_DIR / "scalers.pkl", "rb") as f:
        x_sc, y_sc = pickle.load(f)

    with open(MODELS_DIR / "train_meta.json") as f:
        meta = json.load(f)

    # 1. Leave-one-airfoil-out
    print("\n[analysis] === LOAO ===")
    loao_results = analysis.run_loao(df, meta)

    # 2. Learning curves
    print("\n[analysis] === Learning Curves ===")
    torch.set_num_threads(1)
    lc_results = analysis.run_learning_curves(splits, x_sc, y_sc, meta)

    # 3. Mach extrapolation
    print("\n[analysis] === Mach Extrapolation ===")
    mextrap_results = analysis.run_mach_extrap(df, meta)

    # 4. Feature ablation
    print("\n[analysis] === Feature Ablation ===")
    ablation_results = analysis.run_feature_ablation(splits, x_sc, y_sc, meta)

    print("\n[analysis] done — tables and figures saved to results/")


STAGES = {"eda": run_eda, "train": run_train, "eval": run_eval, "analysis": run_analysis}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    args = ap.parse_args()

    if args.stage == "all":
        for s in ("eda", "train", "eval"):
            STAGES[s]()
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
