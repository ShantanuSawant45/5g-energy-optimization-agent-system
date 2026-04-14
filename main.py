"""
main.py — Entry point for the 5G Energy Optimizer

Usage:
    # Step 1 (once): Train the Federated Learning model
    python -m federated.fl_server

    # Step 2: Run the full agent pipeline
    python main.py

Make sure your GEMINI_API_KEY is set:
    Windows:  set GEMINI_API_KEY=your_key_here
    Mac/Linux: export GEMINI_API_KEY=your_key_here
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from data.loader import load_bs_info, load_energy_stats, load_cell_stats
from graph.pipeline import build_pipeline

# ── File paths — adjust if your data folder is elsewhere ─────────────────────
DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
BSINFO_PATH    = os.path.join(DATA_DIR, "BSinfo.csv")
ECSTAT_PATH    = os.path.join(DATA_DIR, "ECstat.csv")
CLSTAT_PATH    = os.path.join(DATA_DIR, "CLstat.csv")


def main():
    print("\n" + "█"*60)
    print("  5G NETWORK ENERGY OPTIMIZATION — AGENT SYSTEM")
    print("  Group 25 | AI in 5G Networks | LangGraph + Gemini + FL")
    print("█"*60)

    # ── Load real datasets ────────────────────────────────────────────────────
    print("\n[MAIN] Loading datasets...")
    bs_info     = load_bs_info(BSINFO_PATH)
    energy_stats= load_energy_stats(ECSTAT_PATH)
    cell_stats  = load_cell_stats(CLSTAT_PATH)

    # ── Build initial state ───────────────────────────────────────────────────
    initial_state = {
        "bs_info"           : bs_info,
        "energy_stats"      : energy_stats,
        "cell_stats"        : cell_stats,
        "monitor_report"    : None,
        "prediction_report" : None,
        "optimization_plan" : None,
        "control_report"    : None,
        "llm_logs"          : [],
    }

    # ── Build and run pipeline ────────────────────────────────────────────────
    print("\n[MAIN] Building LangGraph pipeline...")
    pipeline = build_pipeline()

    print("\n[MAIN] Running pipeline...\n")
    final_state = pipeline.invoke(initial_state)

    # ── Final summary ─────────────────────────────────────────────────────────
    report = final_state["control_report"]
    print("\n" + "█"*60)
    print("  FINAL REPORT")
    print("█"*60)
    print(f"  Active BSes       : {report['active_count']}")
    print(f"  Sleeping BSes     : {report['sleep_count']}")
    print(f"  Power before      : {report['total_before_w']:,.0f} W")
    print(f"  Power after       : {report['total_after_w']:,.0f} W")
    print(f"  Energy saved      : {report['total_saved_w']:,.0f} W  ({report['pct_saved']:.1f}%)")
    print("\n  Agent reasoning log:")
    for log in final_state["llm_logs"]:
        print(f"    → {log}")
    print("█"*60 + "\n")

    # Save actions to CSV
    out_path = os.path.join(os.path.dirname(__file__), "output_decisions.csv")
    report["actions_df"].to_csv(out_path, index=False)
    print(f"[MAIN] Full decision table saved → {out_path}")


if __name__ == "__main__":
    main()