"""
agents/monitor_agent.py
Agent 1 — Network Monitor Agent

Reads the latest snapshot from CLstat + BSinfo and produces a
health report for every (BS, Cell) pair. It flags:
  - overloaded cells  (load > 0.90)
  - idle cells        (load < 0.10)
  - active ESModes    (any ESMode flag = 1)
"""

import pandas as pd
from graph.state import NetworkState
from data.loader import xget_latest_snapshot


# ── Thresholds ────────────────────────────────────────────────────────────────
OVERLOAD_THRESHOLD = 0.90   # if load > 90% → stressed
IDLE_THRESHOLD     = 0.10   # if load < 10% → candidate for sleep


def monitor_agent(state: NetworkState) -> NetworkState:
    print("\n" + "="*60)
    print("  [AGENT 1] Network Monitor Agent — running")
    print("="*60)

    cell_stats : pd.DataFrame = state["cell_stats"]
    bs_info    : pd.DataFrame = state["bs_info"]

    # ── Step 1: grab latest snapshot per (BS, CellName) ──────────────────────
    snapshot = xget_latest_snapshot(cell_stats)

    # ── Step 2: merge hardware info ───────────────────────────────────────────
    snapshot = snapshot.merge(
        bs_info[["BS", "CellName", "RUType", "Frequency", "Bandwidth",
                 "Antennas", "TXpower"]],
        on=["BS", "CellName"],
        how="left"
    )

    # ── Step 3: add health flags ──────────────────────────────────────────────
    snapshot["is_overloaded"] = snapshot["load"] > OVERLOAD_THRESHOLD
    snapshot["is_idle"]       = snapshot["load"] < IDLE_THRESHOLD

    esmode_cols = [c for c in snapshot.columns if c.startswith("ESMode")]
    snapshot["any_esmode_active"] = snapshot[esmode_cols].sum(axis=1) > 0

    # ── Step 4: summary print ─────────────────────────────────────────────────
    total      = len(snapshot)
    overloaded = snapshot["is_overloaded"].sum()
    idle       = snapshot["is_idle"].sum()
    normal     = total - overloaded - idle

    print(f"  Total (BS, Cell) pairs : {total}")
    print(f"  Overloaded (load>90%)  : {overloaded}")
    print(f"  Idle       (load<10%)  : {idle}")
    print(f"  Normal                 : {normal}")
    print(f"\n  Sample snapshot (top 5):")
    print(snapshot[["BS","CellName","load","is_overloaded","is_idle"]].head().to_string(index=False))

    logs = state.get("llm_logs") or []
    logs.append(f"[Monitor] Snapshot: {total} cells, {idle} idle, {overloaded} overloaded.")

    return {
        **state,
        "monitor_report": snapshot,
        "llm_logs": logs,
    }