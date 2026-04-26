"""
02_data_cleaning.py
-------------------
Step 2: Clean the ball-by-ball dataset.

Operations:
  - Strip whitespace from strings
  - Fix data types (season→int, is_wicket→int, over→int, ball→int)
  - Fill missing non-critical fields (extras_type, player_dismissed, etc.)
  - Drop rows with missing critical fields (match_id, inning, batting_team, winner)
  - Remove duplicate deliveries
  - Report cleaning summary
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
IN_FILE  = os.path.join(PROC_DIR, "01_loaded.csv")
OUT_FILE = os.path.join(PROC_DIR, "02_cleaned.csv")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Input shape: {df.shape}")
    original_rows = len(df)

    # 1. Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
    print("Whitespace stripped from string columns.")

    # 2. Replace 'nan' strings (from str conversion) back to NaN
    df.replace("nan", pd.NA, inplace=True)

    # 3. Fix numeric types
    df["season"]     = pd.to_numeric(df["season"],     errors="coerce").astype("Int64")
    df["inning"]     = pd.to_numeric(df["inning"],     errors="coerce").astype("Int64")
    df["over"]       = pd.to_numeric(df["over"],       errors="coerce").astype("Int64")
    df["ball"]       = pd.to_numeric(df["ball"],       errors="coerce").astype("Int64")
    df["is_wicket"]  = pd.to_numeric(df["is_wicket"],  errors="coerce").fillna(0).astype(int)
    for col in ["batsman_runs", "extra_runs", "total_runs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # 4. Fill non-critical missing fields
    df["extras_type"]      = df["extras_type"].fillna("none")
    df["player_dismissed"] = df["player_dismissed"].fillna("none")
    df["dismissal_kind"]   = df["dismissal_kind"].fillna("none")
    df["fielder"]          = df["fielder"].fillna("none")

    # 5. Drop rows missing critical fields
    critical_cols = ["match_id", "inning", "batting_team", "bowling_team", "winner"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    print(f"Rows dropped (missing critical fields): {before - len(df)}")

    # 6. Remove duplicate deliveries (same match, inning, over, ball)
    before = len(df)
    df = df.drop_duplicates(subset=["match_id", "inning", "over", "ball"])
    print(f"Duplicate deliveries removed: {before - len(df)}")

    # 7. Keep only innings 1 and 2 (remove super overs if any)
    df = df[df["inning"].isin([1, 2])].copy()

    df = df.reset_index(drop=True)
    print(f"\nCleaned shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Rows removed total: {original_rows - len(df)}")
    return df


def main():
    print("=" * 55)
    print("PSL MATCH PREDICTOR — Data Cleaning")
    print("=" * 55)

    df = pd.read_csv(IN_FILE)
    df_clean = clean(df)

    print(f"\nMissing after cleaning:\n{df_clean.isnull().sum()[df_clean.isnull().sum() > 0]}")

    df_clean.to_csv(OUT_FILE, index=False)
    print(f"\n✔  Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
