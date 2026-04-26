"""
03_feature_engineering.py
--------------------------
Step 3: Aggregate ball-by-ball data → match-level features for ML.

Phase A — Match aggregation (per inning stats):
  inning1_runs, inning2_runs, inning1_wickets, inning2_wickets,
  inning1_extras, inning2_extras,
  inning1_pp_runs  (powerplay overs 1-6),
  inning2_pp_runs,
  inning1_death_runs (death overs 17-20),
  inning2_death_runs,
  run_diff         (inning1_runs - inning2_runs)

Phase B — Historical features (computed row-by-row to avoid data leakage):
  team1_win_rate        overall historical win rate
  team2_win_rate
  h2h_team1_win_rate    head-to-head win rate between the two teams
  team1_recent_form     win rate in last 5 matches
  team2_recent_form
  venue_team1_win_rate  win rate at this specific venue for team1
  team1_avg_inning1     average inning-1 score for team1 in past matches

Target:
  target = 1 if team1 (inning 1 batting team) wins, else 0
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
IN_FILE  = os.path.join(PROC_DIR, "02_cleaned.csv")
OUT_FILE = os.path.join(PROC_DIR, "03_featured.csv")


# ── Phase A: Match-level aggregation ──────────────────────────────────────────

def aggregate_innings(df: pd.DataFrame, inning: int) -> pd.DataFrame:
    sub = df[df["inning"] == inning].copy()
    grp = sub.groupby("match_id")

    agg = pd.DataFrame({
        f"inning{inning}_runs":       grp["total_runs"].sum(),
        f"inning{inning}_wickets":    grp["is_wicket"].sum(),
        f"inning{inning}_extras":     grp["extra_runs"].sum(),
        f"inning{inning}_pp_runs":    grp.apply(lambda x: x[x["over"].between(1, 6)]["total_runs"].sum()),
        f"inning{inning}_death_runs": grp.apply(lambda x: x[x["over"].between(17, 20)]["total_runs"].sum()),
        f"inning{inning}_team":       grp["batting_team"].first(),
    })
    return agg.reset_index()


def build_match_df(df: pd.DataFrame) -> pd.DataFrame:
    """One row per match with inning stats and match metadata."""
    meta_cols = ["match_id", "date", "season", "venue", "winner",
                 "win_by", "match_type", "player_of_match", "umpire_1", "umpire_2"]

    meta = (
        df[meta_cols]
        .drop_duplicates(subset=["match_id"])
        .set_index("match_id")
    )

    inn1 = aggregate_innings(df, 1).set_index("match_id")
    inn2 = aggregate_innings(df, 2).set_index("match_id")

    match_df = meta.join(inn1, how="left").join(inn2, how="left")
    match_df = match_df.reset_index()

    # Rename inning teams for clarity
    match_df.rename(columns={
        "inning1_team": "team1",
        "inning2_team": "team2",
    }, inplace=True)

    # Target: 1 if team1 (batting first) wins
    match_df["target"] = (match_df["winner"] == match_df["team1"]).astype(int)

    # Run differential (positive = team1 dominated, negative = team2 dominated)
    match_df["run_diff"] = match_df["inning1_runs"] - match_df["inning2_runs"]

    # Sort by date for historical feature computation
    match_df["date"] = pd.to_datetime(match_df["date"], errors="coerce")
    match_df = match_df.sort_values("date").reset_index(drop=True)

    return match_df


# ── Phase B: Historical features (leak-free) ──────────────────────────────────

def _win_rate(past: pd.DataFrame, team: str) -> float:
    played = past[(past["team1"] == team) | (past["team2"] == team)]
    if len(played) == 0:
        return 0.5
    return (played["winner"] == team).mean()


def _h2h(past: pd.DataFrame, team1: str, team2: str) -> float:
    matchups = past[
        ((past["team1"] == team1) & (past["team2"] == team2)) |
        ((past["team1"] == team2) & (past["team2"] == team1))
    ]
    if len(matchups) == 0:
        return 0.5
    return (matchups["winner"] == team1).mean()


def _recent_form(past: pd.DataFrame, team: str, n: int = 5) -> float:
    played = past[(past["team1"] == team) | (past["team2"] == team)].tail(n)
    if len(played) == 0:
        return 0.5
    return (played["winner"] == team).mean()


def _venue_win_rate(past: pd.DataFrame, team: str, venue: str) -> float:
    at_venue = past[
        ((past["team1"] == team) | (past["team2"] == team)) &
        (past["venue"] == venue)
    ]
    if len(at_venue) == 0:
        return 0.5
    return (at_venue["winner"] == team).mean()


def _avg_score(past: pd.DataFrame, team: str) -> float:
    batting_first = past[past["team1"] == team]["inning1_runs"]
    if len(batting_first) == 0:
        return past["inning1_runs"].mean() if len(past) > 0 else 150.0
    return batting_first.mean()


def add_historical_features(match_df: pd.DataFrame) -> pd.DataFrame:
    print("Computing historical features (leak-free)...")
    feats = {
        "team1_win_rate":       [],
        "team2_win_rate":       [],
        "h2h_team1_win_rate":   [],
        "team1_recent_form":    [],
        "team2_recent_form":    [],
        "venue_team1_win_rate": [],
        "team1_avg_inning1":    [],
    }

    for i, row in match_df.iterrows():
        past = match_df.iloc[:i]
        t1, t2, venue = row["team1"], row["team2"], row["venue"]

        feats["team1_win_rate"].append(_win_rate(past, t1))
        feats["team2_win_rate"].append(_win_rate(past, t2))
        feats["h2h_team1_win_rate"].append(_h2h(past, t1, t2))
        feats["team1_recent_form"].append(_recent_form(past, t1))
        feats["team2_recent_form"].append(_recent_form(past, t2))
        feats["venue_team1_win_rate"].append(_venue_win_rate(past, t1, venue))
        feats["team1_avg_inning1"].append(_avg_score(past, t1))

    for col, values in feats.items():
        match_df[col] = values

    return match_df


def main():
    print("=" * 55)
    print("PSL MATCH PREDICTOR — Feature Engineering")
    print("=" * 55)

    df = pd.read_csv(IN_FILE)
    print(f"Ball-by-ball rows: {len(df):,}")

    print("\nPhase A: Aggregating ball-by-ball → match level...")
    match_df = build_match_df(df)
    print(f"Match-level rows: {len(match_df)}")
    print(f"Columns after aggregation: {list(match_df.columns)}")

    print("\nPhase B: Adding historical features...")
    match_df = add_historical_features(match_df)

    print(f"\nFinal shape: {match_df.shape}")
    print("\nFeature summary:")
    feature_cols = [
        "inning1_runs", "inning2_runs", "inning1_wickets", "inning2_wickets",
        "inning1_pp_runs", "inning2_pp_runs", "inning1_death_runs", "inning2_death_runs",
        "run_diff", "team1_win_rate", "team2_win_rate", "h2h_team1_win_rate",
        "team1_recent_form", "team2_recent_form", "venue_team1_win_rate", "team1_avg_inning1",
    ]
    print(match_df[feature_cols].describe().round(2))

    match_df.to_csv(OUT_FILE, index=False)
    print(f"\n✔  Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
