"""
01_data_loading.py
------------------
Step 1: Load psl_data.csv and perform basic inspection.

Expected file: data/raw/psl_data.csv
Expected columns:
    id, match_id, date, season, venue, inning, batting_team, bowling_team,
    over, ball, batter, bowler, non_striker, batsman_runs, extra_runs,
    total_runs, extras_type, is_wicket, player_dismissed, dismissal_kind,
    fielder, winner, win_by, match_type, player_of_match, umpire_1, umpire_2
"""

import pandas as pd
import os
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_FILE = os.path.join(RAW_DIR,  "psl_data.csv")
os.makedirs(PROC_DIR, exist_ok=True)

EXPECTED_COLS = [
    "id", "match_id", "date", "season", "venue", "inning",
    "batting_team", "bowling_team", "over", "ball", "batter", "bowler",
    "non_striker", "batsman_runs", "extra_runs", "total_runs", "extras_type",
    "is_wicket", "player_dismissed", "dismissal_kind", "fielder",
    "winner", "win_by", "match_type", "player_of_match", "umpire_1", "umpire_2",
]


def load_data() -> pd.DataFrame:
    if not os.path.exists(RAW_FILE):
        print("=" * 55)
        print("ERROR: psl_data.csv not found!")
        print(f"Please place your dataset at:\n  {RAW_FILE}")
        print("=" * 55)
        sys.exit(1)
    df = pd.read_csv(RAW_FILE)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    return df


def validate_columns(df: pd.DataFrame):
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        print(f"\nWARNING: These expected columns are missing: {missing}")
        print(f"Found columns: {list(df.columns)}")
    else:
        print("All expected columns found.")


def inspect(df: pd.DataFrame):
    print("=" * 55)
    print("PSL MATCH PREDICTOR — Data Loading")
    print("=" * 55)
    print(f"\nShape            : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Unique matches   : {df['match_id'].nunique()}")
    print(f"Seasons covered  : {sorted(df['season'].unique())}")
    print(f"Unique venues    : {df['venue'].nunique()}")
    print(f"Unique batters   : {df['batter'].nunique()}")
    print(f"Unique bowlers   : {df['bowler'].nunique()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nSample (5 rows):")
    print(df.head(5).to_string())


def main():
    df = load_data()
    validate_columns(df)
    inspect(df)

    out = os.path.join(PROC_DIR, "01_loaded.csv")
    df.to_csv(out, index=False)
    print(f"\n✔  Saved to {out}")


if __name__ == "__main__":
    main()
