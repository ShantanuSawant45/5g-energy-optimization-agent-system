"""
graph/state.py
Defines the shared state object that flows through all 4 agents.
Each agent reads from it and writes its results back into it.
"""

from typing import TypedDict, Optional
import pandas as pd


class NetworkState(TypedDict):
    # ── Raw data (loaded once, passed through) ──────────────────────────────
    bs_info       : object   # pd.DataFrame  — BSinfo.csv
    energy_stats  : object   # pd.DataFrame  — ECstat.csv
    cell_stats    : object   # pd.DataFrame  — CLstat.csv

    # ── Agent 1 output ───────────────────────────────────────────────────────
    monitor_report: Optional[object]   # pd.DataFrame: current snapshot + health flags

    # ── Agent 2 output ───────────────────────────────────────────────────────
    prediction_report: Optional[object]  # pd.DataFrame: predicted load per (BS, Cell)

    # ── Agent 3 output ───────────────────────────────────────────────────────
    optimization_plan: Optional[object]  # dict: { "B_0": "ACTIVE", "B_1": "SLEEP", ... }

    # ── Agent 4 output ───────────────────────────────────────────────────────
    control_report: Optional[object]   # dict: final actions taken + energy saved

    # ── LLM reasoning logs (optional, useful for demo) ───────────────────────
    llm_logs: Optional[list]

