"""PyTorch MLP with shared trunk and two output heads (Cl, Cm).

Architecture search:
  - hidden layers in {[128,128], [256,128], [256,256], [256,128,64], [512,256,128]}
  - dropout p     in {0.1, 0.3}
  - learning rate in {1e-3, 5e-4}

Training: Adam + MSE + ReduceLROnPlateau + early stopping.
Reports mean ± std over src.config.DNN_SEEDS.
"""
# TODO Phase 6
