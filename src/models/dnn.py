"""PyTorch MLP with shared trunk and two output heads (Cl, Cm).

Architecture search (validation loss decides):
  - hidden layers ∈ {[128,128], [256,128], [256,256], [256,128,64], [512,256,128]}
  - dropout p     ∈ {0.1, 0.3}
  - learning rate ∈ {1e-3, 5e-4}

Training: Adam + MSE + ReduceLROnPlateau + early stopping (patience=20).
Reported metrics are mean ± std over src.config.DNN_SEEDS = (0, 1, 42),
evaluated on the held-out validation split.

Public API
----------
train(X_tr, y_tr, X_val, y_val) -> (model, meta)
    Runs architecture search once per seed combination; returns the best
    single model (lowest val loss) and a meta dict with arch/lr/seed/val_rmse.

train_multi_seed(X_tr, y_tr, X_val, y_val) -> (models, stats)
    Calls train() for each seed in DNN_SEEDS; returns list of models and
    {"mean_val_rmse": float, "std_val_rmse": float}.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.config import DNN_SEEDS

# ─── Architecture search space ──────────────────────────────────────────────
HIDDEN_CONFIGS = (
    (128, 128),
    (256, 128),
    (256, 256),
    (256, 128, 64),
    (512, 256, 128),
)
DROPOUTS       = (0.1, 0.3)
LEARNING_RATES = (1e-3, 5e-4)

_BATCH_SIZE    = 64
_MAX_EPOCHS    = 300
_PATIENCE      = 20


# ─── Model definition ────────────────────────────────────────────────────────
class AeroMLP(nn.Module):
    """Shared-trunk MLP with two scalar heads for Cl and Cm."""

    def __init__(
        self,
        in_features: int,
        hidden: Tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk   = nn.Sequential(*layers)
        self.head_cl = nn.Linear(prev, 1)
        self.head_cm = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h    = self.trunk(x)
        cl   = self.head_cl(h)
        cm   = self.head_cm(h)
        return torch.cat([cl, cm], dim=1)   # (n, 2)


# ─── Single-config training loop ─────────────────────────────────────────────
def _train_one(
    X_tr:  np.ndarray,
    y_tr:  np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden: Tuple[int, ...],
    dropout: float,
    lr: float,
    seed: int,
    device: torch.device,
) -> Tuple[AeroMLP, float, List[float]]:
    """Train one architecture config. Returns (model, best_val_loss, val_loss_history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_t = torch.from_numpy(X_tr).float().to(device)
    y_t = torch.from_numpy(y_tr).float().to(device)
    X_v = torch.from_numpy(X_val).float().to(device)
    y_v = torch.from_numpy(y_val).float().to(device)

    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=_BATCH_SIZE, shuffle=True,
    )

    model     = AeroMLP(X_tr.shape[1], hidden, dropout).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    val_history: List[float] = []

    for _ in range(_MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()
        scheduler.step(val_loss)
        val_history.append(float(np.sqrt(val_loss)))  # store as RMSE

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= _PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_val_loss, val_history


# ─── Architecture search for one seed ─────────────────────────────────────
def train(
    X_tr:  np.ndarray,
    y_tr:  np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed:  int = DNN_SEEDS[0],
    hidden_configs: tuple = HIDDEN_CONFIGS,
    dropouts: tuple = DROPOUTS,
    learning_rates: tuple = LEARNING_RATES,
) -> Tuple[AeroMLP, Dict]:
    """Search architectures for one seed; return best model and meta dict."""
    device    = torch.device("cpu")
    best_loss = float("inf")
    best_model: AeroMLP | None = None
    best_meta: Dict = {}
    best_history: List[float] = []

    for hidden in hidden_configs:
        for dropout in dropouts:
            for lr in learning_rates:
                model, val_loss, history = _train_one(
                    X_tr, y_tr, X_val, y_val,
                    hidden, dropout, lr, seed, device,
                )
                if val_loss < best_loss:
                    best_loss    = val_loss
                    best_model   = model
                    best_history = history
                    best_meta    = {
                        "hidden":       hidden,
                        "dropout":      dropout,
                        "lr":           lr,
                        "seed":         seed,
                        "val_rmse":     float(np.sqrt(best_loss)),
                        "val_history":  history,
                        "n_epochs":     len(history),
                    }

    return best_model, best_meta


# ─── Multi-seed wrapper ────────────────────────────────────────────────────
def train_multi_seed(
    X_tr:  np.ndarray,
    y_tr:  np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seeds: Tuple[int, ...] = DNN_SEEDS,
) -> Tuple[List[AeroMLP], Dict]:
    """Run train() for each seed; return list of models and aggregate stats."""
    models    = []
    per_seed  = []
    val_rmses = []

    for seed in seeds:
        model, meta = train(X_tr, y_tr, X_val, y_val, seed=seed)
        models.append(model)
        per_seed.append(meta)
        val_rmses.append(meta["val_rmse"])

    stats = {
        "mean_val_rmse": float(np.mean(val_rmses)),
        "std_val_rmse":  float(np.std(val_rmses)),
        "per_seed":      per_seed,
    }
    return models, stats
