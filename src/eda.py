"""Exploratory data analysis figure helpers.

Each function takes a DataFrame (and optionally a save path) and writes
a single figure to ``results/figures/eda/``. No model dependencies — pure
data visualization. Called by ``scripts/run_all.py --stage eda``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import viz  # noqa: F401  — applies shared matplotlib rcParams
from src.config import FIGURES_DIR


EDA_DIR = FIGURES_DIR / "eda"


def _ensure_outdir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ─── Airfoil shape grid ─────────────────────────────────────────────────
def plot_airfoil_shapes(
    df: pd.DataFrame,
    save_path: Path = EDA_DIR / "airfoil_shapes.png",
    airfoil_col: str = "airfoil",
    order: Iterable[str] | None = None,
) -> Path:
    """Render one sampled profile per airfoil as an 8-station closed curve.

    Upper surface at x = linspace(0, 1, 8), lower surface at the same x
    reversed; filled to make the shape visually obvious. Grid is 2x4
    (8 airfoils). Used to visually validate the airfoil-name lookup
    in ``src.data._SIG_TO_NAME``.
    """
    if order is None:
        order = sorted(df[airfoil_col].unique())
    assert len(order) == 8, f"Expected 8 airfoils, got {len(order)}"

    x = np.linspace(0.0, 1.0, 8)
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5), sharex=True, sharey=True)

    for ax, name in zip(axes.flat, order):
        sample = df[df[airfoil_col] == name].iloc[0]
        y_up = np.array([sample[f"y_U{i}"] for i in range(1, 9)])
        y_lo = np.array([sample[f"y_L{i}"] for i in range(1, 9)])

        ax.fill_between(x, y_up, y_lo, color="#4c78a8", alpha=0.25, lw=0)
        ax.plot(x, y_up, color="#1f3e66", lw=1.3)
        ax.plot(x, y_lo, color="#1f3e66", lw=1.3)
        ax.axhline(0.0, color="k", lw=0.3, alpha=0.4)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(-0.10, 0.12)
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x/c$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$y/c$")

    fig.suptitle(
        "Airfoil profiles sampled at 8 stations per surface",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(_ensure_outdir(save_path))
    plt.close(fig)
    return save_path


# ─── Coefficient-vs-alpha overlays ─────────────────────────────────────
def plot_coefficient_vs_alpha(
    df: pd.DataFrame,
    target: str,
    save_path: Path | None = None,
    airfoil_col: str = "airfoil",
) -> Path:
    """Scatter of ``target`` vs α, one subplot per airfoil, points
    colored by Mach.

    Used to visualize the nonlinearity each airfoil exhibits in the
    transonic regime. RAE2822's characteristic shock-induced rise in
    Cl and drop in Cm should be visible here — motivating the story
    that simple regressors will struggle near Mach 0.8.
    """
    if target not in ("Cl", "Cm"):
        raise ValueError("target must be 'Cl' or 'Cm'")
    if save_path is None:
        save_path = EDA_DIR / f"{target.lower()}_vs_alpha.png"

    order = sorted(df[airfoil_col].unique())
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)

    sc = None
    for ax, name in zip(axes.flat, order):
        sub = df[df[airfoil_col] == name]
        sc = ax.scatter(
            sub["alpha"], sub[target],
            c=sub["Mach"], cmap="viridis",
            s=16, vmin=df["Mach"].min(), vmax=df["Mach"].max(),
            edgecolors="none",
        )
        ax.set_title(name, fontsize=10)
        ax.axhline(0.0, color="k", lw=0.3, alpha=0.4)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\alpha$ (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"${target}$")

    cbar = fig.colorbar(
        sc, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02, label="Mach",
    )
    cbar.set_ticks(sorted(df["Mach"].unique()))
    fig.suptitle(f"${target}$ vs. α per airfoil, colored by $M_\\infty$", y=1.0)
    fig.savefig(_ensure_outdir(save_path))
    plt.close(fig)
    return save_path


# ─── Correlation heatmap ───────────────────────────────────────────────
def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Path = EDA_DIR / "correlation_heatmap.png",
) -> Path:
    """Pearson correlation heatmap over features + targets.

    Dominant expected correlations: α with Cl (strong positive),
    Mach with Cl/Cm (transonic effects), geometry columns with each
    other (correlated within airfoil family). Used in the report to
    motivate feature importance analysis.
    """
    from src.config import FEATURE_COLS, TARGET_COLS
    cols = FEATURE_COLS + TARGET_COLS
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)

    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center", fontsize=5,
                    color="white" if abs(v) > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature and target correlations", fontsize=11)
    fig.tight_layout()
    fig.savefig(_ensure_outdir(save_path))
    plt.close(fig)
    return save_path


# ─── Target distributions ──────────────────────────────────────────────
_SPLIT_ORDER = ("train", "val", "test", "heldout_rae2822")


def plot_target_distributions_by_split(
    splits: dict,
    save_path: Path = EDA_DIR / "target_dist_by_split.png",
) -> Path:
    """Violin plot of $C_l$ and $C_m$ across the four splits.

    Sanity check: train/val/test should have similar distributions
    (random split should not introduce distributional drift).
    heldout_rae2822 can differ — it is a single airfoil. Used in the
    report to argue the random split is representative.
    """
    order = [s for s in _SPLIT_ORDER if s in splits]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, target in zip(axes, ("Cl", "Cm")):
        data_list = [splits[s][target].values for s in order]
        parts = ax.violinplot(
            data_list, showmeans=True, showmedians=False, widths=0.85,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("#4c78a8")
            pc.set_alpha(0.5)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=12)
        ax.set_ylabel(f"${target}$")
        ax.set_title(f"${target}$ distribution by split")

    fig.tight_layout()
    fig.savefig(_ensure_outdir(save_path))
    plt.close(fig)
    return save_path


def plot_target_distributions_by_airfoil(
    df: pd.DataFrame,
    save_path: Path = EDA_DIR / "target_dist_by_airfoil.png",
    airfoil_col: str = "airfoil",
) -> Path:
    """Box plots of $C_l$ and $C_m$ per airfoil.

    Shows the range of coefficients each geometry produces across its
    ~170 flight conditions. Used in the report to argue RAE2822
    occupies a distinct region of the target space (so the held-out
    evaluation is a non-trivial extrapolation).
    """
    order = sorted(df[airfoil_col].unique())
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, target in zip(axes, ("Cl", "Cm")):
        data_list = [df[df[airfoil_col] == a][target].values for a in order]
        bp = ax.boxplot(
            data_list, tick_labels=order, widths=0.55, patch_artist=True,
            boxprops=dict(facecolor="#d0d8e3", edgecolor="#1f3e66"),
            medianprops=dict(color="#c23b22"),
        )
        ax.set_ylabel(f"${target}$")
        ax.set_title(f"${target}$ distribution per airfoil")
    axes[-1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(_ensure_outdir(save_path))
    plt.close(fig)
    return save_path
