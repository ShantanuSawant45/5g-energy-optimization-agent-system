"""
federated/local_model.py

A small PyTorch MLP that each BS trains locally on its own CLstat data.
Features: [hour_of_day, day_of_week, current_load]
Target  : next-hour load (regression)

After local training, weights are sent to fl_server.py which runs FedAvg
and returns a global model. The global weights are saved to FL_MODEL_PATH.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

FL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "global_model.pt")


# ── Model architecture ────────────────────────────────────────────────────────
class LoadPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()          # output in [0, 1]
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Feature engineering ───────────────────────────────────────────────────────
def prepare_features(cell_stats: pd.DataFrame, bs: str) -> tuple:
    """
    Returns (X, y) numpy arrays for one BS.
    X shape: (n, 3)  — [hour, day_of_week, load_t]
    y shape: (n,)    — load_t+1  (next-step load)
    """
    df = cell_stats[cell_stats["BS"] == bs].sort_values("Time").copy()
    df = df.dropna(subset=["Time", "load"])

    if len(df) < 2:
        return None, None

    df["hour"] = df["Time"].dt.hour
    df["dow"]  = df["Time"].dt.dayofweek

    X = df[["hour", "dow", "load"]].values.astype(np.float32)[:-1]  # all but last
    y = df["load"].values.astype(np.float32)[1:]                     # shifted by 1

    return X, y


# ── Local training ────────────────────────────────────────────────────────────
def train_local_model(X: np.ndarray,
                      y: np.ndarray,
                      global_weights: dict = None,
                      epochs: int = 20,
                      lr: float = 1e-3) -> dict:
    """
    Train the model locally on one BS's data.
    Optionally initialise from global weights (Federated Learning round).
    Returns the trained model's state_dict.
    """
    model = LoadPredictor()
    if global_weights:
        model.load_state_dict(global_weights)   # start from global model

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()

    return model.state_dict()


# ── Inference using saved global model ───────────────────────────────────────
def predict_with_model(features: np.ndarray) -> float:
    """
    Run inference using the saved global model.
    features: shape (1, 3) = [[hour, day_of_week, current_load]]
    Returns a float in [0, 1].
    """
    model = LoadPredictor()
    model.load_state_dict(torch.load(FL_MODEL_PATH, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features)
        return model(x).item()


# ── Quick local test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate synthetic data for sanity check
    X = np.random.rand(100, 3).astype(np.float32)
    y = np.random.rand(100).astype(np.float32)
    weights = train_local_model(X, y, epochs=10)
    print("Local model trained. Keys:", list(weights.keys()))