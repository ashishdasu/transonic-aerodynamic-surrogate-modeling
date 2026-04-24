# Surrogate Aerodynamic Modeling in the Transonic Regime

**CS 6140 Machine Learning — Final Project, Spring 2026**
**Ashish Dasu · Khoury College of Computer Sciences, Northeastern University**
**`dasu.a@northeastern.edu`**

---

## What this project does

This project trains four ML regression models to predict the **lift coefficient
$C_l$** and **pitching moment coefficient $C_m$** of a 2D airfoil directly from
flight conditions and geometry, bypassing multi-hour CFD. The central finding
is that **interpolation accuracy (R²>0.997) does not guarantee extrapolation to
unseen airfoil families** — a result the original paper's random-split evaluation
concealed.

| Model | Test R² (Cl) | Test R² (Cm) | Latency |
|---|---|---|---|
| Polynomial Ridge | 0.9904 | 0.9881 | 0.13 ms |
| Random Forest | 0.9937 | 0.9953 | 3.5 ms |
| **XGBoost** | **0.9987** | **0.9971** | 0.51 ms |
| DNN (PyTorch) | 0.9970 | 0.9964 | 0.11 ms |

On the held-out RAE2822 geometry (unseen supercritical airfoil), all models
yield **negative R²** — worse than predicting the training mean.

---

## Quick start (grader / TA replication)

**Requirements:** Python ≥ 3.11, `make`. No LaTeX needed — the report PDF is pre-compiled and committed.

```bash
git clone https://github.com/ashishdasu/transonic-aerodynamic-surrogate-modeling.git
cd final-project-transonic-surrogate
make all
```

That single command handles everything: creates a virtual environment, installs dependencies, runs unit tests, trains all four models, and generates every figure and table referenced in the report.

**Expected wall clock on a modern laptop (CPU only):** ~30 minutes total (`make analysis` dominates at ~20 min due to 8-fold LOAO cross-validation).

**What gets produced under `results/`:**
```
results/
├── models/          trained model files + train_meta.json
├── figures/
│   ├── eda/         airfoil profiles, target distributions, correlation heatmap
│   ├── eval/        parity plots, residuals by Mach, feature importance, DNN arch
│   └── analysis/    LOAO bar chart, learning curves, Mach extrap, feature ablation
└── tables/          LaTeX-formatted metric tables (also printed to console)
```

The report is at `report/dasu_cs6140_transonic_surrogate.pdf` — open it directly, no LaTeX required. To recompile from source (optional): `make report` (requires pdflatex + bibtex).

**Running stages individually:**
```bash
make setup      # create .venv and install deps (~2 min, first run only)
make test       # unit tests — should pass in <30 s
make train      # train all four models and serialize to results/models/
make eval       # evaluation figures and metric tables (~1 min)
make analysis   # LOAO, learning curves, Mach extrap, feature ablation (~20 min)
```

---

## Repository layout

```
.
├── Input_Data.csv              TransonicSurrogate dataset (committed, 192 KB)
├── Makefile                    Single entry point — all stages
├── requirements.txt            Pinned dependencies
├── pyproject.toml              Project metadata
├── conftest.py                 pytest root config
│
├── src/
│   ├── config.py               Paths, seeds, feature/target names (single source of truth)
│   ├── data.py                 Load CSV, recover airfoil IDs, splits, scalers
│   ├── eda.py                  Six EDA figures
│   ├── train.py                Training orchestration (CV + model serialization)
│   ├── evaluate.py             Metrics, parity plots, latency benchmark
│   ├── analysis.py             LOAO, learning curves, Mach extrap, feature ablation
│   ├── viz.py                  Shared matplotlib rcParams
│   └── models/
│       ├── polynomial.py       Ridge regression with polynomial expansion
│       ├── random_forest.py    Multi-output RF with 5-fold CV grid
│       ├── xgboost_model.py    Per-target XGBRegressor + wrapper
│       └── dnn.py              AeroMLP: shared trunk + dual Cl/Cm heads
│
├── scripts/
│   └── run_all.py              Makefile-invoked CLI (--stage eda/train/eval/analysis)
│
├── tests/
│   └── test_data.py            Data pipeline unit tests
│
├── notebooks/
│   └── 01_eda.ipynb            Exploratory notebook (figures also in make eda)
│
├── report/
│   ├── final.tex               LaTeX source (article class, 9 pages)
│   ├── final.pdf               Pre-compiled PDF
│   └── refs.bib                Bibliography
│
└── results/                    All regenerated outputs (do not edit manually)
    ├── figures/
    │   ├── eda/                Six EDA figures
    │   ├── eval/               Parity, residuals, Cm-vs-alpha, feature importance
    │   └── analysis/           LOAO bar, learning curves, Mach extrap, ablation
    ├── tables/                 LaTeX snippet tables (\input'd by report)
    └── models/                 Serialized models + train_meta.json
```

---

## Dataset

**TransonicSurrogate** (Elrefaie et al., AIAA SciTech 2024).
`Input_Data.csv` is committed at the repo root — no external download needed.

| Property | Value |
|---|---|
| Rows | 1,362 CFD simulations |
| Airfoils | RAE2822, RAE5212, NACA0012/2412/4412/23012/24112/25112 |
| Mach range | 0.65 – 0.85 (5 discrete values) |
| AoA range | −2° to +14.5° (34 values, 0.5° step) |
| Features | 18: y\_U1–y\_U8, y\_L1–y\_L8, α, M∞ |
| Targets | Cl, Cm (multi-output regression) |
| CFD solver | OpenFOAM rhoCentralFoam (Euler, ~7 min/sim on 4-core laptop) |

**Splits (deterministic, seed 42):**
- All 172 RAE2822 rows carved out as held-out geometry probe (never seen in training)
- Remaining 1,190 rows: 60/20/20 → 714 train / 238 val / 238 test

---

## Models

### 1. Polynomial Regression (baseline)
`src/models/polynomial.py` — degree-3 Ridge regression, 5-fold CV over
`degree ∈ {2,3}` × `alpha ∈ {0.01, 0.1, 1.0, 10.0}`.

### 2. Random Forest
`src/models/random_forest.py` — sklearn `MultiOutputRegressor(RandomForestRegressor)`,
5-fold CV over `n_estimators ∈ {50,100,200}` × `max_depth ∈ {5,10,None}`.

### 3. XGBoost
`src/models/xgboost_model.py` — one `XGBRegressor` per output, 5-fold CV
per target. Wrapped in `XGBoostMultiOutput` for a unified `.predict()` interface.

### 4. Deep Neural Network (PyTorch)
`src/models/dnn.py` — **AeroMLP**: shared fully-connected trunk
(BatchNorm → ReLU → Dropout) with two independent linear heads for Cl and Cm.

Architecture: **[512 → 256 → 128]**, dropout p=0.1 (selected by val loss from
20 configurations). Trained with Adam + ReduceLROnPlateau + early stopping
(patience 20). Three seeds (0, 1, 42) to quantify variance:
val RMSE = 0.0571 / 0.0631 / 0.0594 (mean ± std: 0.060 ± 0.003).

---

## Evaluation

All models are evaluated on two distinct splits:

| Split | N | Description |
|---|---|---|
| **Test** | 238 | Random 20% of 7-airfoil pool — *interpolation* |
| **RAE2822 holdout** | 172 | Entire RAE2822 geometry — *extrapolation* |

Metrics: per-target RMSE and R² in original (unscaled) coefficient units.

**Key finding:** XGBoost achieves R²=0.9987 on the interpolation test
(matching the paper's ANN at 0.996) but drops to R²(Cl)=−0.57 on
the RAE2822 holdout. All four models yield negative R² on the unseen geometry.

---

## Extended Analyses (`make analysis`)

Four supplementary analyses beyond the core comparison:

| Analysis | What it shows |
|---|---|
| **LOAO** | 8-fold geometry holdout — confirms RAE2822 is the most OOD airfoil |
| **Learning curves** | R² vs training fraction — XGBoost saturates at 50% data |
| **Mach extrapolation** | Train on M≤0.80, test on M=0.85 — Cm degrades to R²≈0.72 |
| **Feature ablation** | 18 features vs. 5 (domain-selected) — ~0.01–0.02 R² penalty |

---

## Reproducing a specific output

To regenerate just the evaluation figures without retraining:
```bash
make eval
```

To regenerate only EDA figures:
```bash
make eda
```

To run unit tests:
```bash
make test
```

To recompile the PDF (requires pdflatex):
```bash
make report
```

---

## Dependencies

All pinned in `requirements.txt`. Key packages:

| Package | Version | Role |
|---|---|---|
| torch | 2.x | DNN training |
| xgboost | 2.x | Gradient boosting |
| scikit-learn | 1.x | RF, polynomial, CV |
| matplotlib | 3.x | All figures |
| numpy / pandas | — | Data handling |

Install with `make setup` (creates `.venv`).

> **macOS note:** XGBoost and PyTorch both use OpenMP. A known deadlock
> (`__kmp_join_barrier`) occurs if both OMP runtimes load in the same process.
> `scripts/run_all.py` sets `OMP_NUM_THREADS=1` before any numerical import
> to eliminate this. Do not remove those lines.

---

## Reference

Elrefaie, M., Ayman, T., Elrefaie, M., Sayed, E., Ayyad, M., AbdelRahman, M. M.
*Surrogate Modeling of the Aerodynamic Performance for Airfoils in Transonic Regime.*
AIAA SciTech Forum, 2024.
Dataset: <https://github.com/Mohamedelrefaie/TransonicSurrogate>

---

## Acknowledgements

Thanks to Prof. Ehsan Elhamifar for teaching CS 6140 and enabling a project like this.

---

## License

MIT — see `LICENSE`.
