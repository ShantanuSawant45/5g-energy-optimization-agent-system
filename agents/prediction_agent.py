"""
agents/prediction_agent.py
Agent 2 — Traffic Prediction Agent

Uses the Federated Learning trained model (if available) to predict
next-hour load for each (BS, CellName). Falls back to a rolling
average if the FL model is not yet trained.

The FL model is a small PyTorch MLP trained per BS on CLstat data,
then aggregated by the FL server.
"""

import os
import numpy as np
import pandas as pd

# Try importing FL model; if torch not installed yet, skip gracefully
try:
    from federated.local_model import predict_with_model, FL_MODEL_PATH
    FL_AVAILABLE = os.path.exists(FL_MODEL_PATH)
except Exception:
    FL_AVAILABLE = False

from graph.state import NetworkState


def _rolling_average_predict(cell_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback: predict next-hour load as the rolling mean of last 6 readings.
    """
    df = cell_stats.sort_values("Time")
    pred = (
        df.groupby(["BS", "CellName"])["load"]
          .apply(lambda x: x.rolling(6, min_periods=1).mean().iloc[-1])
          .reset_index()
          .rename(columns={"load": "predicted_load"})
    )
    return pred


def _fl_predict(cell_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Use the federated-trained global model for prediction.
    Input features: hour_of_day, day_of_week, current_load
    """
    from federated.local_model import predict_with_model

    latest = (
        cell_stats.sort_values("Time")
                  .groupby(["BS", "CellName"], as_index=False)
                  .last()
    )

    results = []
    for _, row in latest.iterrows():
        hour    = row["Time"].hour if pd.notnull(row["Time"]) else 12
        dow     = row["Time"].dayofweek if pd.notnull(row["Time"]) else 0
        load    = row["load"] if pd.notnull(row["load"]) else 0.0
        features = np.array([[hour, dow, load]], dtype=np.float32)
        pred_load = predict_with_model(features)
        results.append({
            "BS"            : row["BS"],
            "CellName"      : row["CellName"],
            "predicted_load": float(pred_load)
        })

    return pd.DataFrame(results)


def prediction_agent(state: NetworkState) -> NetworkState:
    print("\n" + "="*60)
    print("  [AGENT 2] Traffic Prediction Agent — running")
    print("="*60)

    cell_stats: pd.DataFrame = state["cell_stats"]

    if FL_AVAILABLE:
        print("  Mode: Federated Learning model")
        pred_df = _fl_predict(cell_stats)
    else:
        print("  Mode: Rolling average (FL model not trained yet)")
        pred_df = _rolling_average_predict(cell_stats)

    # Clip predictions to [0, 1]
    pred_df["predicted_load"] = pred_df["predicted_load"].clip(0.0, 1.0)

    # Classify predicted state
    pred_df["predicted_state"] = pred_df["predicted_load"].apply(
        lambda x: "HIGH" if x > 0.7 else ("LOW" if x < 0.2 else "MEDIUM")
    )

    print(f"\n  Predictions (top 5):")
    print(pred_df.head().to_string(index=False))
    print(f"\n  Distribution — HIGH: {(pred_df['predicted_state']=='HIGH').sum()} | "
          f"MEDIUM: {(pred_df['predicted_state']=='MEDIUM').sum()} | "
          f"LOW: {(pred_df['predicted_state']=='LOW').sum()}")

    logs = state.get("llm_logs") or []
    logs.append(f"[Predict] Predicted loads for {len(pred_df)} (BS,Cell) pairs. "
                f"FL used: {FL_AVAILABLE}")

    return {
        **state,
        "prediction_report": pred_df,
        "llm_logs": logs,
    }