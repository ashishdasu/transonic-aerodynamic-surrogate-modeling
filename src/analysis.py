"""Four supplementary analyses extending the core four-model comparison.

Run via:  python scripts/run_all.py --stage analysis

All analyses use best hyperparameters from results/models/train_meta.json
instead of re-running full CV grids, keeping total runtime under 30 min.

1. LOAO            Leave-one-airfoil-out (classical models; DNN excluded)
2. Learning curves  R² vs training-set fraction for all 4 models
3. Mach extrap     Train on M≤0.80, evaluate on M=0.85
4. Feature ablation 18-feature vs 5-feature (domain-selected) models

Outputs saved to results/tables/ and results/figures/analysis/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src import viz  # noqa: F401
from src.config import (
    FEATURE_COLS, FIGURES_DIR, HELDOUT_AIRFOIL,
    MODELS_DIR, SEED, TABLES_DIR, TARGET_COLS,
)
from src.models.xgboost_model import XGBoostMultiOutput

ANALYSIS_FIG_DIR = FIGURES_DIR / "analysis"

# Five features from the source paper's feature-selection experiment.
SELECTED_FEATURES = ["y_U1", "y_U2", "y_L8", "alpha", "Mach"]
TRAIN_FRACTIONS   = [0.10, 0.25, 0.50, 0.75, 1.00]


# ─── Utilities ───────────────────────────────────────────────────────────────

def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    return {
        t: {
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "r2":   float(r2_score(y_true[:, i], y_pred[:, i])),
        }
        for i, t in enumerate(TARGET_COLS)
    }


def _save_tex(rows: List[Dict], path: Path, header: str) -> Path:
    """Write a booktabs table from rows of {label, Cl:{rmse,r2}, Cm:{rmse,r2}}."""
    _ensure(path)
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             f"{header} \\\\ \\midrule"]
    prev = None
    for row in rows:
        label = row["label"] if row["label"] != prev else ""
        prev  = row["label"]
        for t in TARGET_COLS:
            lines.append(
                f'{label} & ${t}$ & {row[t]["rmse"]:.4f} & {row[t]["r2"]:.4f} \\\\'
            )
            label = ""
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines))
    return path


# ─── Model factories (no CV — use pre-tuned best hyperparams) ────────────────

def _meta_hp(meta: Dict) -> Dict:
    rf    = meta["random_forest"]
    depth = None if rf["max_depth"] in ("null", None) else int(rf["max_depth"])
    return {
        "poly_degree": int(meta["polynomial"]["degree"]),
        "poly_alpha":  float(meta["polynomial"]["alpha"]),
        "rf_n":        int(rf["n_estimators"]),
        "rf_depth":    depth,
        "xgb":         meta["xgboost"],
    }


def _fit_poly(X_tr, y_tr, degree, alpha) -> Pipeline:
    p = Pipeline([
        ("poly",  PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge", Ridge(alpha=alpha)),
    ])
    p.fit(X_tr, y_tr)
    return p


def _fit_rf(X_tr, y_tr, n_estimators, max_depth) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        n_jobs=1, random_state=SEED,
    )
    rf.fit(X_tr, y_tr)
    return rf


def _fit_xgb(X_tr: np.ndarray, y_tr: np.ndarray, xgb_meta: Dict) -> XGBoostMultiOutput:
    """Build and fit one XGBRegressor per target; return wrapped."""
    from xgboost import XGBRegressor
    models_dict: Dict = {}
    for i, t in enumerate(TARGET_COLS):
        hp = xgb_meta[t]
        m  = XGBRegressor(
            n_estimators=int(hp["n_estimators"]),
            max_depth=int(hp["max_depth"]),
            learning_rate=float(hp["learning_rate"]),
            subsample=float(hp.get("subsample", 1.0)),
            tree_method="hist",
            random_state=SEED,
            verbosity=0,
        )
        m.fit(X_tr, y_tr[:, i])
        models_dict[t] = m
    return XGBoostMultiOutput(models_dict, {t: {} for t in TARGET_COLS},
                              target_cols=tuple(TARGET_COLS))


def _fit_classical(X_tr, y_tr, hp) -> Dict[str, Any]:
    return {
        "Polynomial":    _fit_poly(X_tr, y_tr, hp["poly_degree"], hp["poly_alpha"]),
        "Random Forest": _fit_rf(X_tr, y_tr, hp["rf_n"], hp["rf_depth"]),
        "XGBoost":       _fit_xgb(X_tr, y_tr, hp["xgb"]),
    }


def _fit_dnn(X_tr, y_tr, X_val, y_val, dnn_meta) -> Any:
    """Train best DNN architecture (seed 0) from saved meta."""
    import torch
    torch.set_num_threads(1)
    from src.models.dnn import _train_one
    best    = dnn_meta["per_seed"][0]
    hidden  = tuple(best["hidden"])
    dropout = best["dropout"]
    lr      = best["lr"]
    model, _, _ = _train_one(
        X_tr, y_tr, X_val, y_val,
        hidden, dropout, lr, seed=SEED, device=torch.device("cpu"),
    )
    return model


def _predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Dispatch predict for sklearn, XGBoostMultiOutput, or AeroMLP."""
    try:
        import torch
        from src.models.dnn import AeroMLP
        if isinstance(model, AeroMLP):
            model.eval()
            with torch.no_grad():
                return model(torch.from_numpy(X).float()).numpy()
    except Exception:
        pass
    return model.predict(X)


# ─── 1. Leave-One-Airfoil-Out ─────────────────────────────────────────────────

def run_loao(df: pd.DataFrame, meta: Dict) -> Dict:
    """Hold out each airfoil in turn; train classical models on remaining 7.

    DNN excluded (8 full DNN training runs is too slow for interactive use).
    Classical results capture the geometry-generalization story.

    Returns {airfoil: {model_name: {target: {rmse, r2}}}}.
    Saves results/tables/loao_metrics.tex and loao_r2_bar.png.
    """
    hp       = _meta_hp(meta)
    airfoils = sorted(df["airfoil"].unique())
    results: Dict[str, Dict] = {}

    for airfoil in airfoils:
        print(f"[loao] holdout={airfoil} …", end=" ", flush=True)
        holdout = df[df["airfoil"] == airfoil]
        pool    = df[df["airfoil"] != airfoil]

        tv, _   = train_test_split(pool, test_size=0.20, random_state=SEED)
        train, _ = train_test_split(tv, test_size=0.20 / 0.80, random_state=SEED)

        x_sc = StandardScaler().fit(train[FEATURE_COLS].values)
        y_sc = StandardScaler().fit(train[TARGET_COLS].values)

        X_tr = x_sc.transform(train[FEATURE_COLS].values).astype(np.float32)
        y_tr = y_sc.transform(train[TARGET_COLS].values).astype(np.float32)
        X_ho = x_sc.transform(holdout[FEATURE_COLS].values).astype(np.float32)
        y_ho = holdout[TARGET_COLS].values

        fold: Dict[str, Dict] = {}
        for name, model in _fit_classical(X_tr, y_tr, hp).items():
            y_pred     = y_sc.inverse_transform(_predict(model, X_ho))
            fold[name] = _metrics(y_ho, y_pred)
        results[airfoil] = fold
        print(f"XGB Cl R²={fold['XGBoost']['Cl']['r2']:.3f}")

    # Table — XGBoost only (best model); bar chart shows all 3 models visually
    path = TABLES_DIR / "loao_metrics.tex"
    _ensure(path)
    lines = [
        r"\begin{tabular}{llrr}", r"\toprule",
        r"Held-out Airfoil & Target & RMSE & $R^2$ \\ \midrule",
    ]
    prev_airfoil = None
    for airfoil in airfoils:
        m = results[airfoil]["XGBoost"]
        a_cell = airfoil if airfoil != prev_airfoil else ""
        prev_airfoil = airfoil
        for ti, t in enumerate(TARGET_COLS):
            lines.append(
                f"{''.join([a_cell if ti == 0 else ''])} & ${t}$ & {m[t]['rmse']:.4f} & {m[t]['r2']:.4f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines))

    # Bar chart per target
    model_names_classical = ("Polynomial", "Random Forest", "XGBoost")
    colors = {"Polynomial": "#e45756", "Random Forest": "#54a24b", "XGBoost": "#4c78a8"}
    x = np.arange(len(airfoils))
    w = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, tname in zip(axes, TARGET_COLS):
        for j, name in enumerate(model_names_classical):
            vals = [results[a][name][tname]["r2"] for a in airfoils]
            ax.bar(x + (j - 1) * w, vals, w, label=name,
                   color=colors[name], alpha=0.85)
        ax.axhline(0, color="k", lw=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("NACA", "N") for a in airfoils], fontsize=7)
        ax.set_ylabel(f"$R^2$ (${tname}$)")
        ax.set_title(f"LOAO generalization — ${tname}$")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Leave-one-airfoil-out: $R^2$ on each held-out geometry\n"
        "(negative = worse than predicting the training mean)",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(_ensure(ANALYSIS_FIG_DIR / "loao_r2_bar.png"))
    plt.close(fig)
    return results


# ─── 2. Learning Curves ───────────────────────────────────────────────────────

def run_learning_curves(
    splits: Dict,
    x_sc: StandardScaler,
    y_sc: StandardScaler,
    meta: Dict,
) -> Dict:
    """R² on fixed test split vs training-set fraction (fixed best hyperparams).

    Saves learning_curves.png.
    """
    hp = _meta_hp(meta)

    X_tr_full = x_sc.transform(splits["train"][FEATURE_COLS].values).astype(np.float32)
    y_tr_full = y_sc.transform(splits["train"][TARGET_COLS].values).astype(np.float32)
    X_te      = x_sc.transform(splits["test"][FEATURE_COLS].values).astype(np.float32)
    y_te      = splits["test"][TARGET_COLS].values
    X_va      = x_sc.transform(splits["val"][FEATURE_COLS].values).astype(np.float32)
    y_va      = y_sc.transform(splits["val"][TARGET_COLS].values).astype(np.float32)

    N_full   = len(X_tr_full)
    mnames   = ["Polynomial", "Random Forest", "XGBoost", "DNN"]
    colors   = {"Polynomial": "#e45756", "Random Forest": "#54a24b",
                "XGBoost": "#4c78a8", "DNN": "#b279a2"}
    # curves[model][target] = [r2 at each fraction]
    curves: Dict[str, Dict[str, List[float]]] = {
        m: {t: [] for t in TARGET_COLS} for m in mnames
    }
    ns: List[int] = []

    rng = np.random.default_rng(SEED)
    for frac in TRAIN_FRACTIONS:
        n = max(int(frac * N_full), 20)
        ns.append(n)
        idx   = rng.choice(N_full, size=n, replace=False)
        X_sub = X_tr_full[idx]
        y_sub = y_tr_full[idx]

        for name, model in _fit_classical(X_sub, y_sub, hp).items():
            y_pred = y_sc.inverse_transform(_predict(model, X_te))
            for i, t in enumerate(TARGET_COLS):
                curves[name][t].append(float(r2_score(y_te[:, i], y_pred[:, i])))

        # DNN: carve a small internal val set from the subsample
        nv      = max(int(0.2 * n), 5)
        X_s_tr  = X_sub[nv:] if n - nv >= 10 else X_sub
        y_s_tr  = y_sub[nv:] if n - nv >= 10 else y_sub
        X_s_va  = X_sub[:nv] if n - nv >= 10 else X_va[:nv]
        y_s_va  = y_sub[:nv] if n - nv >= 10 else y_va[:nv]

        dnn_m = _fit_dnn(X_s_tr, y_s_tr, X_s_va, y_s_va, meta["dnn"])
        import torch
        dnn_m.eval()
        with torch.no_grad():
            y_pred_sc = dnn_m(torch.from_numpy(X_te).float()).numpy()
        y_pred_dnn = y_sc.inverse_transform(y_pred_sc)
        for i, t in enumerate(TARGET_COLS):
            curves["DNN"][t].append(float(r2_score(y_te[:, i], y_pred_dnn[:, i])))

        print(f"[lcurve] frac={frac:.0%} N={n}: "
              f"XGB Cl R²={curves['XGBoost']['Cl'][-1]:.3f}  "
              f"DNN Cl R²={curves['DNN']['Cl'][-1]:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, tname in zip(axes, TARGET_COLS):
        for name in mnames:
            ax.plot(ns, curves[name][tname], "-o",
                    color=colors[name], lw=1.8, ms=5, label=name)
        ax.set_xlabel("Training samples $N$")
        ax.set_ylabel(f"$R^2$ (${tname}$)")
        ax.set_title(f"Learning curve — ${tname}$")
        ax.set_xscale("log")
        ax.axhline(1.0, color="k", lw=0.4, linestyle=":")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Data efficiency: $R^2$ on fixed test split vs training-set size\n"
        "(fixed best hyperparameters, no re-tuning)"
    )
    fig.tight_layout()
    fig.savefig(_ensure(ANALYSIS_FIG_DIR / "learning_curves.png"))
    plt.close(fig)
    return {"curves": curves, "ns": ns}


# ─── 3. Mach Extrapolation ────────────────────────────────────────────────────

def run_mach_extrap(df: pd.DataFrame, meta: Dict) -> Dict:
    """Train on M≤0.80, evaluate on M=0.85 (unseen near-critical regime).

    RAE2822 excluded to keep geometry consistent with the main study.
    Saves mach_extrap_metrics.tex and mach_extrap_residuals.png.
    """
    hp       = _meta_hp(meta)
    df_pool  = df[df["airfoil"] != HELDOUT_AIRFOIL].copy()
    train_df = df_pool[df_pool["Mach"] < 0.85]
    test_df  = df_pool[df_pool["Mach"] == 0.85]

    x_sc = StandardScaler().fit(train_df[FEATURE_COLS].values)
    y_sc = StandardScaler().fit(train_df[TARGET_COLS].values)

    X_tr = x_sc.transform(train_df[FEATURE_COLS].values).astype(np.float32)
    y_tr = y_sc.transform(train_df[TARGET_COLS].values).astype(np.float32)
    X_te = x_sc.transform(test_df[FEATURE_COLS].values).astype(np.float32)
    y_te = test_df[TARGET_COLS].values

    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED)

    models: Dict[str, Any] = _fit_classical(X_tr, y_tr, hp)
    models["DNN"] = _fit_dnn(X_tr2, y_tr2, X_va, y_va, meta["dnn"])

    results: Dict[str, Dict] = {}
    for name, model in models.items():
        y_pred        = y_sc.inverse_transform(_predict(model, X_te))
        results[name] = _metrics(y_te, y_pred)
        print(f"[mextrap] {name}: Cl R²={results[name]['Cl']['r2']:.3f}  "
              f"Cm R²={results[name]['Cm']['r2']:.3f}")

    rows = [{"label": n, **results[n]}
            for n in ("Polynomial", "Random Forest", "XGBoost", "DNN")]
    _save_tex(rows, TABLES_DIR / "mach_extrap_metrics.tex",
              header="Model & Target & RMSE & $R^2$")

    # Cm vs alpha at M=0.85 for one airfoil
    airfoil_eg  = "NACA4412"
    test_reset  = test_df.reset_index(drop=True)
    sub         = test_reset[test_reset["airfoil"] == airfoil_eg].sort_values("alpha")
    if len(sub) > 0:
        sub_idx    = sub.index.tolist()
        alpha_vals = sub["alpha"].values
        cm_cfd     = sub["Cm"].values
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(alpha_vals, cm_cfd, "k-o", ms=4, lw=1.5, label="CFD truth")
        for (name, model), c in zip(models.items(), plt.cm.tab10.colors):
            y_pred_full = y_sc.inverse_transform(_predict(model, X_te))
            ax.plot(alpha_vals, y_pred_full[sub_idx, 1], "--",
                    color=c, lw=1.3, label=name)
        ax.set_xlabel(r"$\alpha$ (deg)")
        ax.set_ylabel(r"$C_m$")
        ax.set_title(
            f"{airfoil_eg} at $M_\\infty=0.85$ (unseen Mach): $C_m(\\alpha)$\n"
            "Models trained on $M\\leq0.80$ only"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(_ensure(ANALYSIS_FIG_DIR / "mach_extrap_residuals.png"))
        plt.close(fig)

    return results


# ─── 4. Feature Ablation ─────────────────────────────────────────────────────

def run_feature_ablation(
    splits: Dict,
    x_sc_full: StandardScaler,
    y_sc: StandardScaler,
    meta: Dict,
) -> Dict:
    """Compare all-18-feature vs 5-feature (domain-selected) models on test split.

    SELECTED_FEATURES mirrors the source paper's feature subset, allowing
    direct comparison: does the full 16-coordinate geometry parameterization
    add predictive value over alpha, Mach, and three key coordinates?

    Saves ablation_metrics.tex and feature_ablation.png.
    """
    hp       = _meta_hp(meta)
    train_df = splits["train"]
    test_df  = splits["test"]
    val_df   = splits["val"]

    x_sc_sel = StandardScaler().fit(train_df[SELECTED_FEATURES].values)

    X_tr_full = x_sc_full.transform(train_df[FEATURE_COLS].values).astype(np.float32)
    X_te_full = x_sc_full.transform(test_df[FEATURE_COLS].values).astype(np.float32)
    X_va_full = x_sc_full.transform(val_df[FEATURE_COLS].values).astype(np.float32)
    X_tr_sel  = x_sc_sel.transform(train_df[SELECTED_FEATURES].values).astype(np.float32)
    X_te_sel  = x_sc_sel.transform(test_df[SELECTED_FEATURES].values).astype(np.float32)
    X_va_sel  = x_sc_sel.transform(val_df[SELECTED_FEATURES].values).astype(np.float32)

    y_tr = y_sc.transform(train_df[TARGET_COLS].values).astype(np.float32)
    y_va = y_sc.transform(val_df[TARGET_COLS].values).astype(np.float32)
    y_te = test_df[TARGET_COLS].values

    results: Dict[str, Dict] = {}
    for label, X_tr, X_te, X_va in [
        ("18 features", X_tr_full, X_te_full, X_va_full),
        ("5 features",  X_tr_sel,  X_te_sel,  X_va_sel),
    ]:
        for name, model in _fit_classical(X_tr, y_tr, hp).items():
            key = f"{name} ({label})"
            y_pred     = y_sc.inverse_transform(_predict(model, X_te))
            results[key] = _metrics(y_te, y_pred)

        dnn_m = _fit_dnn(X_tr, y_tr, X_va, y_va, meta["dnn"])
        key   = f"DNN ({label})"
        y_pred      = y_sc.inverse_transform(_predict(dnn_m, X_te))
        results[key] = _metrics(y_te, y_pred)

        for n in ("Polynomial", "Random Forest", "XGBoost", "DNN"):
            k = f"{n} ({label})"
            print(f"[ablate] {k}: Cl R²={results[k]['Cl']['r2']:.4f}")

    rows = [
        {"label": f"{n} / {lbl}", **results[f"{n} ({lbl})"]}
        for n in ("Polynomial", "Random Forest", "XGBoost", "DNN")
        for lbl in ("18 features", "5 features")
    ]
    _save_tex(rows, TABLES_DIR / "ablation_metrics.tex",
              header="Model / Features & Target & RMSE & $R^2$")

    # Compact delta table — one row per model, ΔR² per target
    mnames = ["Polynomial", "Random Forest", "XGBoost", "DNN"]
    delta_lines = [
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Model & $\Delta R^2(C_l)$ & $\Delta R^2(C_m)$ \\ \midrule",
    ]
    for n in mnames:
        dr2_cl = results[f"{n} (5 features)"]["Cl"]["r2"] - results[f"{n} (18 features)"]["Cl"]["r2"]
        dr2_cm = results[f"{n} (5 features)"]["Cm"]["r2"] - results[f"{n} (18 features)"]["Cm"]["r2"]
        delta_lines.append(f"{n} & {dr2_cl:+.4f} & {dr2_cm:+.4f} \\\\")
    delta_lines += [r"\bottomrule", r"\end{tabular}"]
    delta_path = TABLES_DIR / "ablation_delta.tex"
    _ensure(delta_path)
    delta_path.write_text("\n".join(delta_lines))

    # Bar chart
    mnames = ["Polynomial", "Random Forest", "XGBoost", "DNN"]
    x = np.arange(len(mnames))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, tname in zip(axes, TARGET_COLS):
        r2_full = [results[f"{n} (18 features)"][tname]["r2"] for n in mnames]
        r2_sel  = [results[f"{n} (5 features)"][tname]["r2"]  for n in mnames]
        ax.bar(x - w/2, r2_full, w, label="18 features", color="#4c78a8", alpha=0.85)
        ax.bar(x + w/2, r2_sel,  w, label="5 features",  color="#f58518", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(mnames, fontsize=8)
        ax.set_ylabel(f"$R^2$ (${tname}$)")
        ax.set_title(f"Feature ablation — ${tname}$")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Feature ablation: full 18-feature vs 5 domain-selected features\n"
        r"(5 features: $y_{U1}, y_{U2}, y_{L8}, \alpha, M_\infty$)",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(_ensure(ANALYSIS_FIG_DIR / "feature_ablation.png"))
    plt.close(fig)
    return results
