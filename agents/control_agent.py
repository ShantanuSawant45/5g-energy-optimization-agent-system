"""
agents/control_agent.py
Agent 4 — Tower Control Agent

Executes the optimization plan:
  1. For each BS marked SLEEP → simulate user handover to nearest active BS
  2. Send SLEEP command (simulated)
  3. Calculate energy savings
  4. Generate final report
"""

import pandas as pd
from graph.state import NetworkState

# Base power per antenna (Watts) — standard 5G figures
ACTIVE_POWER_PER_ANTENNA_W  = 500.0   # ~2000W for 4-antenna BS
SLEEP_POWER_W               = 150.0   # sleep mode baseline


def _calculate_power(row: pd.Series, state: str) -> float:
    """Estimate power consumption for a BS row."""
    antennas = int(row.get("Antennas", 2)) if pd.notnull(row.get("Antennas")) else 2
    if state == "SLEEP":
        return SLEEP_POWER_W
    return antennas * ACTIVE_POWER_PER_ANTENNA_W


def _find_nearest_active(bs: str,
                          decisions: dict,
                          bs_info: pd.DataFrame,
                          pred_df: pd.DataFrame) -> str | None:
    """
    Find the nearest active BS that can absorb handover users.
    Heuristic: same RUType, ACTIVE, lowest current predicted load.
    """
    ru_type = bs_info[bs_info["BS"] == bs]["RUType"].values
    if len(ru_type) == 0:
        ru_type = None
    else:
        ru_type = ru_type[0]

    candidates = [
        b for b, d in decisions.items()
        if d == "ACTIVE" and b != bs
        and (ru_type is None or
             bs_info[bs_info["BS"] == b]["RUType"].values[0] == ru_type
             if len(bs_info[bs_info["BS"] == b]) > 0 else True)
    ]

    if not candidates:
        # Fallback: any active BS
        candidates = [b for b, d in decisions.items() if d == "ACTIVE" and b != bs]

    if not candidates:
        return None

    # Pick the one with the lowest predicted load (most capacity available)
    cand_loads = pred_df[pred_df["BS"].isin(candidates)].groupby("BS")["predicted_load"].mean()
    if cand_loads.empty:
        return candidates[0]
    return cand_loads.idxmin()


def control_agent(state: NetworkState) -> NetworkState:
    print("\n" + "="*60)
    print("  [AGENT 4] Tower Control Agent — running")
    print("="*60)

    decisions : dict          = state["optimization_plan"]
    bs_info   : pd.DataFrame  = state["bs_info"]
    pred_df   : pd.DataFrame  = state["prediction_report"]
    ec_stats  : pd.DataFrame  = state["energy_stats"]

    # Deduplicate bs_info to one row per BS (take first cell's hardware config)
    bs_hw = bs_info.groupby("BS").first().reset_index()

    actions    = []
    total_before_w = 0.0
    total_after_w  = 0.0

    for bs, decision in decisions.items():
        hw_row = bs_hw[bs_hw["BS"] == bs]
        hw     = hw_row.iloc[0] if len(hw_row) > 0 else pd.Series({"Antennas": 2})

        power_before = _calculate_power(hw, "ACTIVE")  # all towers were ACTIVE before
        power_after  = _calculate_power(hw, decision)

        total_before_w += power_before
        total_after_w  += power_after

        if decision == "SLEEP":
            target = _find_nearest_active(bs, decisions, bs_info, pred_df)
            actions.append({
                "BS"           : bs,
                "Action"       : "SLEEP",
                "HandoverTo"   : target or "N/A",
                "PowerBefore_W": power_before,
                "PowerAfter_W" : power_after,
                "Saved_W"      : power_before - power_after,
            })
        else:
            actions.append({
                "BS"           : bs,
                "Action"       : "ACTIVE",
                "HandoverTo"   : "—",
                "PowerBefore_W": power_before,
                "PowerAfter_W" : power_after,
                "Saved_W"      : 0.0,
            })

    actions_df   = pd.DataFrame(actions)
    total_saved  = total_before_w - total_after_w
    pct_saved    = (total_saved / total_before_w * 100) if total_before_w > 0 else 0

    sleep_count  = (actions_df["Action"] == "SLEEP").sum()
    active_count = (actions_df["Action"] == "ACTIVE").sum()

    print(f"\n  {'BS':<8} {'Action':<8} {'HandoverTo':<10} {'Saved(W)':>10}")
    print(f"  {'-'*40}")
    for _, row in actions_df.head(15).iterrows():
        print(f"  {row['BS']:<8} {row['Action']:<8} {str(row['HandoverTo']):<10} "
              f"{row['Saved_W']:>10.1f}")
    if len(actions_df) > 15:
        print(f"  ... ({len(actions_df)-15} more)")

    print(f"\n  ⚡ ENERGY SUMMARY")
    print(f"  {'─'*40}")
    print(f"  Total BSes     : {len(actions_df)}")
    print(f"  ACTIVE         : {active_count}")
    print(f"  SLEEP          : {sleep_count}")
    print(f"  Before (W)     : {total_before_w:,.1f}")
    print(f"  After  (W)     : {total_after_w:,.1f}")
    print(f"  Saved  (W)     : {total_saved:,.1f}")
    print(f"  Savings        : {pct_saved:.1f}%")

    control_report = {
        "actions_df"     : actions_df,
        "total_before_w" : total_before_w,
        "total_after_w"  : total_after_w,
        "total_saved_w"  : total_saved,
        "pct_saved"      : pct_saved,
        "active_count"   : active_count,
        "sleep_count"    : sleep_count,
    }

    logs = state.get("llm_logs") or []
    logs.append(f"[Control] {sleep_count} towers slept. Energy saved: {pct_saved:.1f}%")

    return {
        **state,
        "control_report": control_report,
        "llm_logs": logs,
    }

