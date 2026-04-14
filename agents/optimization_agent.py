"""
agents/optimization_agent.py
Agent 3 — Energy Optimization Agent  (the brain)

Decides ACTIVE or SLEEP for every BS based on:
  1. Predicted load from Agent 2
  2. Hardware config from BSinfo (TXpower, Antennas)
  3. Coverage rules (never sleep if no neighbour can cover)

Uses Gemini LLM to reason about edge cases.
"""

import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from graph.state import NetworkState

# ── LLM setup (FREE Gemini) ───────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.1,
)

# ── Thresholds ────────────────────────────────────────────────────────────────
SLEEP_LOAD_THRESHOLD   = 0.20   # if predicted load < 20% → candidate for sleep
OVERLOAD_CAP           = 0.85   # active neighbour must be below 85% before accepting users
MIN_ACTIVE_PER_RUTYPE  = 1      # keep at least 1 BS active per RUType cluster


def _build_neighbour_map(bs_info: pd.DataFrame) -> dict:
    """
    Simple neighbour heuristic: BSes with the same RUType and similar Frequency
    (within ±50 MHz) are considered neighbours.
    """
    neighbours = {}
    bs_list = bs_info["BS"].unique()
    for bs in bs_list:
        row = bs_info[bs_info["BS"] == bs].iloc[0]
        nbrs = bs_info[
            (bs_info["BS"] != bs) &
            (bs_info["RUType"] == row["RUType"]) &
            (bs_info["Frequency"].between(row["Frequency"] - 50,
                                          row["Frequency"] + 50))
        ]["BS"].tolist()
        neighbours[bs] = nbrs
    return neighbours


def _rule_based_decision(pred_df: pd.DataFrame,
                          bs_info: pd.DataFrame,
                          neighbours: dict) -> dict:
    """
    Returns dict { BS: "ACTIVE" | "SLEEP" } using hard rules.
    """
    decisions = {}

    # Track current predicted load per BS (mean across its cells)
    bs_load = pred_df.groupby("BS")["predicted_load"].mean()

    # Group BSes by RUType to enforce min-active constraint
    rutype_groups = bs_info.groupby("RUType")["BS"].apply(list).to_dict()

    for bs, load in bs_load.items():
        if load >= SLEEP_LOAD_THRESHOLD:
            decisions[bs] = "ACTIVE"
        else:
            # Check if at least one neighbour is ACTIVE and not overloaded
            nbrs = neighbours.get(bs, [])
            active_nbrs = [
                n for n in nbrs
                if bs_load.get(n, 1.0) >= SLEEP_LOAD_THRESHOLD
                and bs_load.get(n, 1.0) < OVERLOAD_CAP
            ]
            decisions[bs] = "SLEEP" if active_nbrs else "ACTIVE"

    # Enforce: at least 1 active per RUType
    for rutype, bs_group in rutype_groups.items():
        group_decisions = {b: decisions.get(b, "ACTIVE") for b in bs_group}
        if all(v == "SLEEP" for v in group_decisions.values()):
            # Wake the one with highest predicted load
            best = max(bs_group, key=lambda b: bs_load.get(b, 0))
            decisions[best] = "ACTIVE"

    return decisions


def _llm_review_edge_cases(decisions: dict,
                            pred_df: pd.DataFrame,
                            bs_info: pd.DataFrame) -> str:
    """
    Ask Gemini to review the edge cases and flag any concerns.
    Returns a text summary.
    """
    sleep_count  = sum(1 for v in decisions.values() if v == "SLEEP")
    active_count = sum(1 for v in decisions.values() if v == "ACTIVE")

    sample = pred_df[["BS","predicted_load","predicted_state"]].head(10).to_string(index=False)

    prompt = f"""
You are a 5G network energy optimization expert.
Here is a summary of decisions made by a rule-based optimizer:
- Total BSes: {len(decisions)}
- Decided ACTIVE: {active_count}
- Decided SLEEP: {sleep_count}

Sample predicted loads:
{sample}

Rules applied:
- Sleep if predicted load < 20%
- Never sleep if no neighbour can absorb users
- Keep at least 1 BS active per RUType group

In 2-3 sentences, confirm if this strategy looks reasonable,
or flag any obvious risks (coverage gaps, overload risks).
Be concise.
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return f"[LLM unavailable: {e}]"


def optimization_agent(state: NetworkState) -> NetworkState:
    print("\n" + "="*60)
    print("  [AGENT 3] Energy Optimization Agent — running")
    print("="*60)

    pred_df : pd.DataFrame = state["prediction_report"]
    bs_info : pd.DataFrame = state["bs_info"]

    neighbours = _build_neighbour_map(bs_info)
    decisions  = _rule_based_decision(pred_df, bs_info, neighbours)

    sleep_bses  = [b for b, d in decisions.items() if d == "SLEEP"]
    active_bses = [b for b, d in decisions.items() if d == "ACTIVE"]

    print(f"  ACTIVE BSes : {len(active_bses)}")
    print(f"  SLEEP  BSes : {len(sleep_bses)}")
    print(f"\n  First 10 decisions:")
    for bs, dec in list(decisions.items())[:10]:
        load = pred_df[pred_df["BS"]==bs]["predicted_load"].mean()
        print(f"    {bs:6s} → {dec:6s}  (pred load: {load:.2%})")

    # Ask LLM to review
    print("\n  [LLM REVIEW] Asking Gemini to validate decisions...")
    llm_review = _llm_review_edge_cases(decisions, pred_df, bs_info)
    print(f"  Gemini says: {llm_review}")

    logs = state.get("llm_logs") or []
    logs.append(f"[Optimize] {len(active_bses)} ACTIVE, {len(sleep_bses)} SLEEP. LLM: {llm_review}")

    return {
        **state,
        "optimization_plan": decisions,
        "llm_logs": logs,
    }