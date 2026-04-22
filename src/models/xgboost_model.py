"""XGBoost regressor, one model per target (Cl, Cm).

XGBoost does not natively multi-output, so we wrap two XGBRegressors.

Hyperparameters (5-fold CV, per target):
  - learning_rate in {0.01, 0.05, 0.1}
  - max_depth     in {3, 5, 7}
  - n_estimators  in {100, 300, 500}
  - subsample     in {0.8, 1.0}
"""
# TODO Phase 5
