"""
04_eda_graphs.py
----------------
Step 4: Exploratory Data Analysis — 8 graphs from real ball-by-ball data.

Graphs saved to outputs/graphs/:
  1.  win_distribution.png       Team win counts
  2.  season_wins.png            Wins per team per season
  3.  top_batsmen.png            Top 10 run scorers
  4.  top_bowlers.png            Top 10 wicket takers
  5.  dismissal_types.png        How wickets fall (pie chart)
  6.  venue_win_rate.png         First-innings win rate per venue
  7.  powerplay_vs_win.png       Powerplay runs vs match outcome
  8.  feature_correlation.png    Heatmap of engineered features
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

matplotlib.use("Agg")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "graphs")
BALL_FILE  = os.path.join(PROC_DIR, "02_cleaned.csv")
FEAT_FILE  = os.path.join(PROC_DIR, "03_featured.csv")
os.makedirs(GRAPHS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})
PALETTE = ["#1a6b8a", "#e05a2b", "#2e9b6f", "#c9a227", "#7b3fa0", "#c94040", "#2e5fa0", "#6b8a1a"]


def save(fig, name):
    path = os.path.join(GRAPHS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ── Graph 1: Win Distribution ─────────────────────────────────────────────────
def plot_win_distribution(mdf):
    wins = mdf["winner"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(wins.index, wins.values, color=PALETTE)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("Total Wins by Team")
    ax.set_ylabel("Matches Won")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save(fig, "win_distribution.png")


# ── Graph 2: Season-wise Wins ─────────────────────────────────────────────────
def plot_season_wins(mdf):
    top_teams = mdf["winner"].value_counts().head(4).index.tolist()
    pivot = (
        mdf[mdf["winner"].isin(top_teams)]
        .groupby(["season", "winner"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot.plot(kind="bar", ax=ax, color=PALETTE[:4], width=0.7)
    ax.set_title("Season-wise Wins — Top 4 Teams")
    ax.set_ylabel("Wins")
    ax.set_xlabel("Season")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "season_wins.png")


# ── Graph 3: Top Batsmen ──────────────────────────────────────────────────────
def plot_top_batsmen(bdf):
    top = bdf.groupby("batter")["batsman_runs"].sum().nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top.index[::-1], top.values[::-1], color=PALETTE[0])
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("Top 10 Run Scorers (All PSL)")
    ax.set_xlabel("Total Runs")
    fig.tight_layout()
    save(fig, "top_batsmen.png")


# ── Graph 4: Top Bowlers ──────────────────────────────────────────────────────
def plot_top_bowlers(bdf):
    wickets = bdf[bdf["is_wicket"] == 1]
    # Exclude run-outs (fielder takes credit, not bowler)
    wickets = wickets[wickets["dismissal_kind"] != "run out"]
    top = wickets.groupby("bowler")["is_wicket"].sum().nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top.index[::-1], top.values[::-1], color=PALETTE[1])
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("Top 10 Wicket Takers (All PSL)")
    ax.set_xlabel("Wickets")
    fig.tight_layout()
    save(fig, "top_bowlers.png")


# ── Graph 5: Dismissal Types ──────────────────────────────────────────────────
def plot_dismissal_types(bdf):
    wickets = bdf[bdf["is_wicket"] == 1]
    dismissals = wickets["dismissal_kind"].value_counts()
    dismissals = dismissals[dismissals.index != "none"]
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        dismissals.values,
        labels=dismissals.index,
        autopct="%1.1f%%",
        colors=PALETTE[:len(dismissals)],
        startangle=140,
        pctdistance=0.8,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Dismissal Types")
    fig.tight_layout()
    save(fig, "dismissal_types.png")


# ── Graph 6: Venue — First Innings Win Rate ───────────────────────────────────
def plot_venue_win_rate(mdf):
    venue_data = mdf.groupby("venue").apply(
        lambda g: pd.Series({
            "matches": len(g),
            "team1_win_rate": g["target"].mean(),
        })
    ).reset_index()
    # Only venues with enough matches
    venue_data = venue_data[venue_data["matches"] >= 5].sort_values("team1_win_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(venue_data) * 0.5)))
    colors = ["#e05a2b" if v > 0.5 else "#1a6b8a" for v in venue_data["team1_win_rate"]]
    bars = ax.barh(venue_data["venue"], venue_data["team1_win_rate"], color=colors)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("First-Innings Win Rate by Venue")
    ax.set_xlabel("Win Rate (team batting first)")
    ax.set_xlim(0, 1)
    red_patch  = mpatches.Patch(color="#e05a2b", label="> 50% (batting first wins)")
    blue_patch = mpatches.Patch(color="#1a6b8a", label="< 50% (chasing team wins)")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8)
    fig.tight_layout()
    save(fig, "venue_win_rate.png")


# ── Graph 7: Powerplay Runs vs Win ────────────────────────────────────────────
def plot_powerplay_vs_win(mdf):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label in zip(
        axes,
        ["inning1_pp_runs", "inning2_pp_runs"],
        ["Inning 1 Powerplay Runs", "Inning 2 Powerplay Runs"],
    ):
        winners   = mdf[mdf["target"] == 1][col].dropna()
        losers    = mdf[mdf["target"] == 0][col].dropna()
        ax.hist(winners, bins=15, alpha=0.7, color="#2e9b6f", label="Team1 Won")
        ax.hist(losers,  bins=15, alpha=0.7, color="#e05a2b", label="Team1 Lost")
        ax.set_title(label)
        ax.set_xlabel("Runs in Overs 1-6")
        ax.set_ylabel("Matches")
        ax.legend(fontsize=8)
    fig.suptitle("Powerplay Runs vs Match Outcome", fontsize=13)
    fig.tight_layout()
    save(fig, "powerplay_vs_win.png")


# ── Graph 8: Feature Correlation Heatmap ─────────────────────────────────────
def plot_correlation(mdf):
    feat_cols = [
        "inning1_runs", "inning2_runs", "inning1_wickets", "inning2_wickets",
        "inning1_pp_runs", "inning2_pp_runs", "run_diff",
        "team1_win_rate", "team2_win_rate", "h2h_team1_win_rate",
        "team1_recent_form", "team2_recent_form", "venue_team1_win_rate",
        "team1_avg_inning1", "target",
    ]
    available = [c for c in feat_cols if c in mdf.columns]
    corr = mdf[available].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=ax, linewidths=0.4, annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    save(fig, "feature_correlation.png")


def main():
    print("=" * 55)
    print("PSL MATCH PREDICTOR — EDA & Graphs")
    print("=" * 55)

    bdf = pd.read_csv(BALL_FILE)    # ball-by-ball
    mdf = pd.read_csv(FEAT_FILE)    # match-level with features
    print(f"Ball-by-ball rows : {len(bdf):,}")
    print(f"Match-level rows  : {len(mdf)}")
    print("\nGenerating graphs...")

    plot_win_distribution(mdf)
    plot_season_wins(mdf)
    plot_top_batsmen(bdf)
    plot_top_bowlers(bdf)
    plot_dismissal_types(bdf)
    plot_venue_win_rate(mdf)
    plot_powerplay_vs_win(mdf)
    plot_correlation(mdf)

    print(f"\n✔  All 8 graphs saved to {GRAPHS_DIR}")


if __name__ == "__main__":
    main()
