# Surrogate Aerodynamic Modeling in the Transonic Regime

Machine learning regression surrogates for predicting airfoil lift
($C_l$) and pitching moment ($C_m$) coefficients from flight conditions
and geometry, replacing multi-hour CFD with sub-millisecond inference.

Final project for **CS 6140 Machine Learning**, Northeastern University
(Spring 2026). Author: Ashish Dasu (`dasu.a@northeastern.edu`).

## Reproducing the results

Requires **Python 3.11** and (for the report only) a working LaTeX
install providing `pdflatex` and `bibtex`. Wall-clock from a clean clone
on a modern laptop: **~5 minutes total**, CPU only.

```
make setup     # create .venv, install pinned dependencies
make all       # test -> eda -> train -> eval -> report
```

After `make all`:

- `report/final.pdf` — the submitted document.
- `results/figures/` — every figure referenced in the report.
- `results/tables/` — every table referenced in the report.
- `results/models/` — serialized trained models.

If `pdflatex` is unavailable, skip `make report` and open the
pre-compiled `report/final.pdf` instead.

## What's being predicted

| Item | Value |
|---|---|
| Dataset | TransonicSurrogate (Elrefaie et al., AIAA SciTech 2024) |
| Samples | 1,362 CFD simulations |
| Airfoils | 8 (NACA0012/2412/4412/23012/24112/25112, RAE2822/5212) |
| Mach range | 0.3 – 0.8 |
| AoA range | -4° to +8° |
| Features (18) | 16 shape y-coordinates + α + M∞ |
| Targets (2) | $C_l$, $C_m$ |

`Input_Data.csv` (192 KB) is committed at the repo root so the TAs do
not need to fetch anything.

## Models compared

1. **Polynomial regression (ridge)** — smooth-fit baseline.
2. **Random Forest** — nonlinear ensemble, axis-aligned splits.
3. **XGBoost** — gradient-boosted trees, one model per target.
4. **Deep Neural Network (PyTorch)** — shared trunk + two output heads,
   joint training.

All models are tuned via 5-fold CV on the training split. The DNN is
additionally trained across three seeds so reported metrics include
variance bars.

## Evaluation protocol

- 60/20/20 train/val/test random split (seed 42).
- **RAE2822 is held out entirely from training and validation** and
  evaluated separately as an unseen-geometry generalization probe.
- Per-target $R^2$ and RMSE on test and on the RAE2822 slice.
- Mach-binned residual analysis ($M < 0.6$ / $0.6$–$0.7$ / $> 0.7$).
- Physical plausibility: predicted $C_m$-vs-α overlays on CFD ground
  truth at fixed Mach values.
- Feature importance (RF and XGBoost).
- Inference latency benchmark (1,000 runs, CPU, single sample).

## Repository layout

```
Input_Data.csv          raw TransonicSurrogate dataset (committed)
Makefile                single entry point for all replication
src/
  config.py             paths, seeds, feature/target names (authoritative)
  data.py               loading, airfoil-ID recovery, splits, scaling
  models/               one file per model family
  train.py              training orchestration
  evaluate.py           metrics, tables, figures
  viz.py                shared matplotlib rcParams
scripts/run_all.py      Makefile-invoked CLI stages
tests/test_data.py      data-pipeline sanity tests
notebooks/01_eda.ipynb  exploratory analysis (auto-exported to figures)
report/
  final.tex             IEEEtran conference template, 9-page report
  refs.bib              references
results/                regenerated outputs (figures, tables, models)
```

## Dataset & paper reference

Elrefaie, M., Ayman, T., Elrefaie, M., Sayed, E., Ayyad, M.,
AbdelRahman, M. M. *Surrogate Modeling of the Aerodynamic Performance
for Airfoils in Transonic Regime.* AIAA SciTech Forum, 2024.
<https://github.com/Mohamedelrefaie/TransonicSurrogate>

## License

MIT — see `LICENSE`.
