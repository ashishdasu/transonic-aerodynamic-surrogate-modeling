"""Training orchestration: fit all four models and persist to results/models/.

Called by ``scripts/run_all.py --stage train``.

Artifacts saved:
  results/models/polynomial.pkl
  results/models/random_forest.pkl
  results/models/xgboost.pkl
  results/models/dnn_seed<N>.pt   (one per DNN_SEEDS)
  results/models/scalers.pkl      (x_scaler, y_scaler tuple)
  results/models/train_meta.json  (all CV results + DNN stats)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

from src import data
from src.config import DNN_SEEDS, MODELS_DIR
from src.models import polynomial, random_forest, xgboost_model


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_train() -> Dict:
    """Fit all models; return meta dict with best hyperparams + CV RMSEs."""
    print("[train] loading data …")
    df          = data.assign_airfoil_ids(data.load_raw())
    splits      = data.make_splits(df)
    x_sc, y_sc  = data.fit_scalers(splits["train"])

    X_tr, y_tr  = data.transform(splits["train"], x_sc, y_sc)
    X_va, y_va  = data.transform(splits["val"],   x_sc, y_sc)

    # Save scalers first — everything else depends on them at eval time
    scaler_path = _ensure(MODELS_DIR / "scalers.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump((x_sc, y_sc), f)
    print(f"[train] scalers → {scaler_path.name}")

    all_meta: Dict = {}

    # ── Polynomial ────────────────────────────────────────────────────────
    print("[train] polynomial … ", end="", flush=True)
    poly_model, poly_meta = polynomial.train(X_tr, y_tr)
    path = _ensure(MODELS_DIR / "polynomial.pkl")
    with open(path, "wb") as f:
        pickle.dump(poly_model, f)
    all_meta["polynomial"] = poly_meta
    print(f"degree={poly_meta['degree']} α={poly_meta['alpha']} cv_rmse={poly_meta['cv_rmse']:.4f} → {path.name}")

    # ── Random Forest ─────────────────────────────────────────────────────
    print("[train] random_forest … ", end="", flush=True)
    rf_model, rf_meta = random_forest.train(X_tr, y_tr)
    path = _ensure(MODELS_DIR / "random_forest.pkl")
    with open(path, "wb") as f:
        pickle.dump(rf_model, f)
    all_meta["random_forest"] = rf_meta
    print(
        f"n={rf_meta['n_estimators']} d={rf_meta['max_depth']} "
        f"cv_rmse={rf_meta['cv_rmse']:.4f} → {path.name}"
    )

    # ── XGBoost ───────────────────────────────────────────────────────────
    print("[train] xgboost … ", end="", flush=True)
    xgb_wrapper, xgb_meta = xgboost_model.train(X_tr, y_tr)
    path = _ensure(MODELS_DIR / "xgboost.pkl")
    with open(path, "wb") as f:
        pickle.dump(xgb_wrapper, f)
    all_meta["xgboost"] = xgb_meta
    for t, m in xgb_meta.items():
        print(
            f"\n[train]   xgb/{t}: lr={m['learning_rate']} d={m['max_depth']} "
            f"n={m['n_estimators']} cv_rmse={m['cv_rmse']:.4f}",
            end="",
        )
    print(f" → {path.name}")

    # ── DNN (multi-seed) ──────────────────────────────────────────────────
    # Imported lazily so torch initializes AFTER XGBoost finishes.
    # set_num_threads(1) prevents OMP barrier deadlock: XGBoost's OMP runtime
    # leaves thread-pool state that causes PyTorch's parallel batch_norm to
    # hang indefinitely in __kmp_join_barrier on macOS.
    import torch
    torch.set_num_threads(1)
    from src.models import dnn

    print("[train] dnn (3 seeds) … ", end="", flush=True)
    dnn_models, dnn_stats = dnn.train_multi_seed(X_tr, y_tr, X_va, y_va, seeds=DNN_SEEDS)
    for seed, model in zip(DNN_SEEDS, dnn_models):
        path = _ensure(MODELS_DIR / f"dnn_seed{seed}.pt")
        torch.save(model.state_dict(), path)
    all_meta["dnn"] = dnn_stats
    print(
        f"mean_val_rmse={dnn_stats['mean_val_rmse']:.4f} "
        f"± {dnn_stats['std_val_rmse']:.4f}"
    )

    # Persist full meta
    meta_path = _ensure(MODELS_DIR / "train_meta.json")
    # Convert any non-JSON-serializable items (None, tuples → lists)
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        if obj is None:
            return "null"
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    with open(meta_path, "w") as f:
        json.dump(_json_safe(all_meta), f, indent=2)
    print(f"[train] meta → {meta_path.name}")

    return all_meta
