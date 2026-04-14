"""
federated/fl_server.py

Federated Learning Server — FedAvg algorithm (no Flower dependency needed).

How it works:
  1. For each BS (client), train a local model on its own CLstat data
  2. Collect all local model weights
  3. Average them → global model weights  (FedAvg)
  4. Repeat for N rounds
  5. Save final global model to global_model.pt

Run this ONCE before running main.py to pre-train the FL model.
Usage:
    python -m federated.fl_server
"""

import os
import copy
import torch
import pandas as pd
import numpy as np

from federated.local_model import (
    LoadPredictor,
    FL_MODEL_PATH,
    prepare_features,
    train_local_model,
)

# ── Config ────────────────────────────────────────────────────────────────────
FL_ROUNDS       = 5     # number of federation rounds
LOCAL_EPOCHS    = 15    # epochs each client trains per round
MIN_SAMPLES     = 10    # skip BS if fewer than this many data points
MAX_CLIENTS     = 50    # cap to avoid very long training (use top N busiest BSes)


def fedavg(weights_list: list) -> dict:
    """
    FedAvg: element-wise average of all clients' model weights.
    """
    avg = copy.deepcopy(weights_list[0])
    for key in avg:
        for w in weights_list[1:]:
            avg[key] += w[key]
        avg[key] = torch.div(avg[key], len(weights_list))
    return avg


def run_federated_training(cell_stats_path: str = None):
    """
    Full FL training loop. Saves global model to FL_MODEL_PATH.
    """
    # ── Load CLstat ──────────────────────────────────────────────────────────
    if cell_stats_path is None:
        cell_stats_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "CLstat.csv"
        )

    print(f"[FL Server] Loading CLstat from: {cell_stats_path}")
    df = pd.read_csv(cell_stats_path)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["load"] = pd.to_numeric(df["load"], errors="coerce")
    df = df.dropna(subset=["Time", "load"])

    all_bs = df["BS"].unique().tolist()

    # Pick the busiest BSes (most data points) to cap training time
    bs_counts = df.groupby("BS").size().sort_values(ascending=False)
    selected_bs = bs_counts.head(MAX_CLIENTS).index.tolist()
    print(f"[FL Server] Total BSes: {len(all_bs)} | Training on: {len(selected_bs)}")

    # ── Initialise global model ──────────────────────────────────────────────
    global_model = LoadPredictor()
    global_weights = global_model.state_dict()

    # ── FL rounds ────────────────────────────────────────────────────────────
    for rnd in range(1, FL_ROUNDS + 1):
        print(f"\n[FL Server] ── Round {rnd}/{FL_ROUNDS} ──")
        local_weights_list = []
        skipped = 0

        for bs in selected_bs:
            X, y = prepare_features(df, bs)
            if X is None or len(X) < MIN_SAMPLES:
                skipped += 1
                continue

            weights = train_local_model(
                X, y,
                global_weights=copy.deepcopy(global_weights),
                epochs=LOCAL_EPOCHS
            )
            local_weights_list.append(weights)

        if not local_weights_list:
            print("[FL Server] No valid clients this round — skipping.")
            continue

        global_weights = fedavg(local_weights_list)
        print(f"[FL Server] Aggregated {len(local_weights_list)} clients "
              f"({skipped} skipped). FedAvg done.")

    # ── Save global model ────────────────────────────────────────────────────
    global_model.load_state_dict(global_weights)
    torch.save(global_weights, FL_MODEL_PATH)
    print(f"\n[FL Server] ✅ Global model saved → {FL_MODEL_PATH}")


if __name__ == "__main__":
    run_federated_training()