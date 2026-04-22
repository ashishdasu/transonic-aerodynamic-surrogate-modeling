"""Polynomial regression baseline: degree-d basis expansion + ridge.

Hyperparameters (5-fold CV on train):
  - degree in {1, 2, 3}
  - ridge alpha in {1e-2, 1e-1, 1, 10}

Exposes train(...) -> fitted model and a .predict(X) -> (n, 2) method.
"""
# TODO Phase 3
