"""
app.py — PSL Match Predictor Dashboard (Streamlit)
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="PSL Match Predictor", page_icon="🏏", layout="wide")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "graphs")
FEAT_FILE  = os.path.join(BASE_DIR, "data", "processed", "03_featured.csv")
BALL_FILE  = os.path.join(BASE_DIR, "data", "processed", "02_cleaned.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "best_model.pkl")

TEAM_COLORS = {
    "Islamabad United":  "#003366",
    "Karachi Kings":     "#c00000",
    "Lahore Qalandars":  "#006633",
    "Peshawar Zalmi":    "#f5a623",
    "Quetta Gladiators": "#1b1b1b",
    "Multan Sultans":    "#7b2d8b",
}


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_match_data():
    if not os.path.exists(FEAT_FILE):
        return None
    return pd.read_csv(FEAT_FILE)


@st.cache_data
def load_ball_data():
    if not os.path.exists(BALL_FILE):
        return None
    return pd.read_csv(BALL_FILE)


def get_teams_and_venues(mdf):
    teams  = sorted(set(mdf["team1"].dropna().tolist() + mdf["team2"].dropna().tolist()))
    venues = sorted(mdf["venue"].dropna().unique().tolist())
    return teams, venues


def compute_features(mdf, team1, team2, venue):
    """Compute historical features for prediction."""
    def win_rate(team):
        played = mdf[(mdf["team1"] == team) | (mdf["team2"] == team)]
        return (played["winner"] == team).mean() if len(played) > 0 else 0.5

    def h2h(t1, t2):
        m = mdf[((mdf["team1"] == t1) & (mdf["team2"] == t2)) |
                ((mdf["team1"] == t2) & (mdf["team2"] == t1))]
        return (m["winner"] == t1).mean() if len(m) > 0 else 0.5

    def recent_form(team, n=5):
        played = mdf[(mdf["team1"] == team) | (mdf["team2"] == team)].tail(n)
        return (played["winner"] == team).mean() if len(played) > 0 else 0.5

    def venue_wr(team, v):
        at = mdf[((mdf["team1"] == team) | (mdf["team2"] == team)) & (mdf["venue"] == v)]
        return (at["winner"] == team).mean() if len(at) > 0 else 0.5

    def avg_inning1(team):
        scores = mdf[mdf["team1"] == team]["inning1_runs"]
        return scores.mean() if len(scores) > 0 else mdf["inning1_runs"].mean()

    return pd.DataFrame([{
        "team1_win_rate":       win_rate(team1),
        "team2_win_rate":       win_rate(team2),
        "h2h_team1_win_rate":   h2h(team1, team2),
        "team1_recent_form":    recent_form(team1),
        "team2_recent_form":    recent_form(team2),
        "venue_team1_win_rate": venue_wr(team1, venue),
        "team1_avg_inning1":    avg_inning1(team1),
    }])


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏏 PSL Match Predictor")
st.caption("Machine Learning model trained on real PSL ball-by-ball data")

mdf = load_match_data()
bdf = load_ball_data()

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 EDA Graphs", "📈 Player Stats", "🤖 Model"])

# ── Tab 1: Predict ─────────────────────────────────────────────────────────────
with tab1:
    if mdf is None:
        st.error("Run the pipeline first (scripts 01–05) to generate processed data.")
    else:
        teams, venues = get_teams_and_venues(mdf)
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Team 1 (batting first)", teams)
        with col2:
            team2 = st.selectbox("Team 2 (batting second)", [t for t in teams if t != team1])

        venue = st.selectbox("Venue", venues)

        if st.button("🔮 Predict Winner", use_container_width=True, type="primary"):
            model_obj = load_model()
            if model_obj is None:
                st.error("Model not found. Run 05_model_training.py first.")
            else:
                features = compute_features(mdf, team1, team2, venue)
                pipeline = model_obj["model"]
                proba    = pipeline.predict_proba(features)[0]
                p1, p2   = proba[1], proba[0]
                winner   = team1 if p1 >= p2 else team2
                conf     = max(p1, p2) * 100

                st.markdown("---")
                c1, c2, c3 = st.columns([2, 1, 2])
                with c1:
                    clr = TEAM_COLORS.get(team1, "#1a6b8a")
                    st.markdown(
                        f"<div style='text-align:center;padding:20px;border-radius:10px;"
                        f"border:2px solid {clr};background:{clr}18'>"
                        f"<h2 style='color:{clr}'>{team1}</h2><h3>{p1*100:.1f}%</h3></div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown("<div style='text-align:center;padding:40px'><h1>VS</h1></div>",
                                unsafe_allow_html=True)
                with c3:
                    clr = TEAM_COLORS.get(team2, "#e05a2b")
                    st.markdown(
                        f"<div style='text-align:center;padding:20px;border-radius:10px;"
                        f"border:2px solid {clr};background:{clr}18'>"
                        f"<h2 style='color:{clr}'>{team2}</h2><h3>{p2*100:.1f}%</h3></div>",
                        unsafe_allow_html=True,
                    )
                wclr = TEAM_COLORS.get(winner, "#2e9b6f")
                st.markdown(
                    f"<div style='text-align:center;margin-top:16px;padding:16px;"
                    f"border-radius:10px;border:2px solid {wclr};background:{wclr}15'>"
                    f"<h2>🏆 {winner}</h2><p>Confidence: {conf:.1f}%</p></div>",
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                st.subheader("Historical Context")
                m_played = mdf[
                    ((mdf["team1"] == team1) & (mdf["team2"] == team2)) |
                    ((mdf["team1"] == team2) & (mdf["team2"] == team1))
                ]
                if len(m_played) > 0:
                    t1w = (m_played["winner"] == team1).sum()
                    t2w = (m_played["winner"] == team2).sum()
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Total H2H Matches", len(m_played))
                    col_b.metric(f"{team1} wins", t1w)
                    col_c.metric(f"{team2} wins", t2w)
                else:
                    st.info("No head-to-head history found between these teams.")


# ── Tab 2: EDA Graphs ──────────────────────────────────────────────────────────
with tab2:
    graphs = {
        "Team Win Distribution":         "win_distribution.png",
        "Season-wise Wins":              "season_wins.png",
        "Powerplay Runs vs Outcome":     "powerplay_vs_win.png",
        "Venue — First-Innings Win Rate":"venue_win_rate.png",
        "Feature Correlation Heatmap":   "feature_correlation.png",
    }
    for title, fname in graphs.items():
        path = os.path.join(GRAPHS_DIR, fname)
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(path, use_container_width=True)
            st.markdown("---")
        else:
            st.info(f"Run 04_eda_graphs.py to generate: {fname}")


# ── Tab 3: Player Stats ────────────────────────────────────────────────────────
with tab3:
    for title, fname in [("Top 10 Run Scorers", "top_batsmen.png"),
                         ("Top 10 Wicket Takers", "top_bowlers.png"),
                         ("Dismissal Types", "dismissal_types.png")]:
        path = os.path.join(GRAPHS_DIR, fname)
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(path, use_container_width=False)
            st.markdown("---")

    if bdf is not None:
        st.subheader("Search Player Stats")
        all_batters = sorted(bdf["batter"].dropna().unique().tolist())
        player = st.selectbox("Select batter", all_batters)
        pdata  = bdf[bdf["batter"] == player]
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Runs",   int(pdata["batsman_runs"].sum()))
        col_b.metric("Balls Faced",  len(pdata))
        fours = (pdata["batsman_runs"] == 4).sum()
        sixes = (pdata["batsman_runs"] == 6).sum()
        col_c.metric("4s", int(fours))
        col_d.metric("6s", int(sixes))
        sr = (pdata["batsman_runs"].sum() / len(pdata) * 100) if len(pdata) > 0 else 0
        st.metric("Strike Rate", f"{sr:.1f}")


# ── Tab 4: Model ──────────────────────────────────────────────────────────────
with tab4:
    comp_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    if os.path.exists(comp_path):
        st.subheader("Model Comparison")
        comp = pd.read_csv(comp_path)
        st.dataframe(
            comp.style.highlight_max(subset=["CV Acc", "CV F1", "CV ROC-AUC"], color="#d4edda"),
            use_container_width=True,
        )

    for fname, label in [
        ("model_comparison_chart.png", "Model Comparison Chart"),
        ("confusion_matrix.png",       "Confusion Matrix"),
        ("roc_curve.png",              "ROC Curve"),
        ("feature_importance.png",     "Feature Importance"),
    ]:
        path = os.path.join(GRAPHS_DIR, fname)
        if os.path.exists(path):
            st.markdown(f"**{label}**")
            st.image(path, use_container_width=False)

    model_obj = load_model()
    if model_obj:
        st.success(f"✅ Best model: **{model_obj['name']}**")
        st.code(f"Features: {model_obj['features']}")
