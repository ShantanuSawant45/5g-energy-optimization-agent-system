"""
data/loader.py
Loads and preprocesses all 3 real CSV datasets.

Files expected inside a  'data/' folder:
  - BSinfo.csv   → base station hardware config  (Image 1)
  - ECstat.csv   → energy consumption per BS     (Image 2)
  - CLstat.csv   → load + ESMode flags per cell  (Image 3)
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__))


def load_bs_info(path: str = None) -> pd.DataFrame:
    """
    Load BSinfo.csv — base station hardware config.

    Columns: BS, CellName, RUType, Mode, Frequency, Bandwidth, Antennas, TXpower
    """
    path = path or os.path.join(DATA_DIR, "BSinfo.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Normalise types
    df["Frequency"]  = pd.to_numeric(df["Frequency"],  errors="coerce")
    df["Bandwidth"]  = pd.to_numeric(df["Bandwidth"],  errors="coerce")
    df["Antennas"]   = pd.to_numeric(df["Antennas"],   errors="coerce")
    df["TXpower"]    = pd.to_numeric(df["TXpower"],    errors="coerce")
    print(f"[LOADER] BSinfo   → {len(df)} rows | towers: {df['BS'].nunique()}")
    return df


def load_energy_stats(path: str = None) -> pd.DataFrame:
    """
    Load ECstat.csv — energy consumption over time per BS.

    Columns: Time, BS, Energy
    """
    path = path or os.path.join(DATA_DIR, "ECstat.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Time"]   = pd.to_datetime(df["Time"], errors="coerce")
    df["Energy"] = pd.to_numeric(df["Energy"], errors="coerce")
    df = df.dropna(subset=["Time"])
    df = df.sort_values("Time").reset_index(drop=True)
    print(f"[LOADER] ECstat   → {len(df)} rows | BSes: {df['BS'].nunique()} | "
          f"range: {df['Time'].min()} → {df['Time'].max()}")
    return df


def load_cell_stats(path: str = None) -> pd.DataFrame:
    """
    Load CLstat.csv — per-cell load and energy-saving mode flags.

    Columns: Time, BS, CellName, load, ESMode1..ESMode6
    """
    path = path or os.path.join(DATA_DIR, "CLstat.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["load"] = pd.to_numeric(df["load"], errors="coerce")

    esmode_cols = [c for c in df.columns if c.startswith("ESMode")]
    for col in esmode_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=["Time"])
    df = df.sort_values("Time").reset_index(drop=True)
    print(f"[LOADER] CLstat   → {len(df)} rows | BSes: {df['BS'].nunique()} | "
          f"ESMode cols: {esmode_cols}")
    return df


def xget_latest_snapshot(cell_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the most recent row per (BS, CellName) pair.
    This is what the Monitor Agent reads as the current network state.
    """
    latest = (
        cell_stats
        .sort_values("Time")
        .groupby(["BS", "CellName"], as_index=False)
        .last()
    )
    return latest


def get_tower_energy_baseline(energy_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Returns mean energy per BS — used to calculate savings.
    """
    return energy_stats.groupby("BS")["Energy"].mean().reset_index()\
                       .rename(columns={"Energy": "AvgEnergy"})


if __name__ == "__main__":
    bs   = load_bs_info()
    ec   = load_energy_stats()
    cl   = load_cell_stats()
    snap = get_latest_snapshot(cl)
    print("\n[SNAPSHOT — first 5 rows]")
    print(snap.head())