# 🏏 PSL Match Predictor

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge)](https://psl-match-predictor.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/shahzad3349/PSL-Match-Predictor)

> End-to-end Machine Learning project predicting **Pakistan Super League** match outcomes using real ball-by-ball data.

🔴 **Live App:** [psl-match-predictor.streamlit.app](https://psl-match-predictor.streamlit.app)

---

## Dataset Setup

**Before running anything**, place your dataset file here:

```
data/raw/psl_data.csv
```

**Required columns:**

| Column | Description |
|---|---|
| `id` | Unique delivery ID |
| `match_id` | Match identifier |
| `date` | Match date |
| `season` | PSL season year |
| `venue` | Stadium name |
| `inning` | 1 or 2 |
| `batting_team` | Team currently batting |
| `bowling_team` | Team currently bowling |
| `over` | Over number |
| `ball` | Ball number in over |
| `batter` | Batter name |
| `bowler` | Bowler name |
| `non_striker` | Non-striker batter |
| `batsman_runs` | Runs scored off bat |
| `extra_runs` | Extra runs |
| `total_runs` | Total runs on delivery |
| `extras_type` | Type of extra (wide, no-ball, etc.) |
| `is_wicket` | 1 if wicket fell, else 0 |
| `player_dismissed` | Dismissed batter name |
| `dismissal_kind` | How dismissed (caught, bowled, etc.) |
| `fielder` | Fielder involved |
| `winner` | Match winner |
| `win_by` | Win margin |
| `match_type` | league / qualifier / final |
| `player_of_match` | Player of the match |
| `umpire_1` | First umpire |
| `umpire_2` | Second umpire |

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline (in order)

```bash
python src/01_data_loading.py       # Load & validate dataset
python src/02_data_cleaning.py      # Clean & fix types
python src/03_feature_engineering.py # Aggregate to match level + build features
python src/04_eda_graphs.py         # Generate 8 EDA graphs
python src/05_model_training.py     # Train 3 models, save best
python src/06_model_evaluation.py   # Full evaluation report
```

### 3. Launch Streamlit Dashboard

```bash
streamlit run app.py
```

---

## Feature Engineering

**Phase A — Ball-by-ball → Match aggregation:**
- `inning1_runs`, `inning2_runs` — total runs per inning
- `inning1_wickets`, `inning2_wickets`
- `inning1_pp_runs`, `inning2_pp_runs` — powerplay (overs 1–6)
- `inning1_death_runs`, `inning2_death_runs` — death overs (17–20)
- `run_diff` — scoring dominance indicator

**Phase B — Historical features (leak-free, pre-match):**
- `team1_win_rate`, `team2_win_rate`
- `h2h_team1_win_rate` — head-to-head history
- `team1_recent_form`, `team2_recent_form` — last 5 matches
- `venue_team1_win_rate` — venue-specific win rate
- `team1_avg_inning1` — average first-innings score

---

## Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Ensemble, handles non-linearity |
| XGBoost | Gradient boosting, usually best |

Best model is auto-selected and saved.

---

## EDA Insights Generated

1. Team win distributions
2. Season-wise performance trends
3. Top 10 run scorers
4. Top 10 wicket takers
5. Dismissal type breakdown
6. First-innings win rate by venue
7. Powerplay runs vs match outcome
8. Feature correlation heatmap

---

## Project Structure

```
PSL_Predictor/
├── data/
│   ├── raw/
│   │   └── psl_data.csv          ← YOUR DATASET HERE
│   └── processed/
│       ├── 01_loaded.csv
│       ├── 02_cleaned.csv
│       └── 03_featured.csv
├── src/
│   ├── 01_data_loading.py
│   ├── 02_data_cleaning.py
│   ├── 03_feature_engineering.py
│   ├── 04_eda_graphs.py
│   ├── 05_model_training.py
│   └── 06_model_evaluation.py
├── outputs/
│   ├── graphs/                    ← all .png files
│   └── models/
│       ├── best_model.pkl
│       └── model_comparison.csv
├── app.py                         ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Dashboard Tabs

| Tab | Content |
|---|---|
| 🔮 Predict | Select teams + venue → get win probability |
| 📊 EDA Graphs | All visualizations from step 04 |
| 📈 Player Stats | Top batsmen, bowlers, search any player |
| 🤖 Model | Model comparison, confusion matrix, ROC curve, feature importance |

---

## Future Improvements

- Add player-level features (top batsman/bowler per team)
- Live match state prediction (given current score/wickets)
- Hyperparameter tuning with Optuna
- Deploy on Streamlit Cloud

---

## Author

**[Your Name]**
BS Computer Science — UET Lahore
[GitHub](#) | [LinkedIn](#)
