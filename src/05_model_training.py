"""
05_model_training.py
--------------------
Step 5: Train and compare 3 models on match-level features.

Models:
  1. Logistic Regression  (interpretable baseline)
  2. Random Forest        (ensemble, handles non-linearity)
  3. XGBoost              (gradient boosting — usually best)

Features used (pre-match + historical — no data leakage):
  team1_win_rate, team2_win_rate, h2h_team1_win_rate,
  team1_recent_form, team2_recent_form, venue_team1_win_rate,
  team1_avg_inning1

Post-match features (inning stats used for EDA only, NOT for pre-match prediction):
  These are excluded intentionally to make the model realistic —
  you cannot know inning1_runs before the match starts.

Best model saved to: outputs/models/best_model.pkl
Comparison saved to: outputs/models/model_comparison.csv
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from xgboost                 import XGBClassifier

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
IN_FILE    = os.path.join(PROC_DIR, "03_featured.csv")
os.makedirs(MODELS_DIR, exist_ok=True)

# Pre-match features only (realistic for actual prediction)
FEATURE_COLS = [
    "team1_win_rate",
    "team2_win_rate",
    "h2h_team1_win_rate",
    "team1_recent_form",
    "team2_recent_form",
    "venue_team1_win_rate",
    "team1_avg_inning1",
]
TARGET_COL = "target"

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=500, C=1.0, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=6,
                min_samples_leaf=3, random_state=42, n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, verbosity=0,
            )),
        ]),
    }


def evaluate(models: dict, X, y) -> pd.DataFrame:
    results = []
    for name, pipeline in models.items():
        print(f"  Evaluating: {name} ...")
        acc = cross_val_score(pipeline, X, y, cv=CV, scoring="accuracy")
        f1  = cross_val_score(pipeline, X, y, cv=CV, scoring="f1")
        roc = cross_val_score(pipeline, X, y, cv=CV, scoring="roc_auc")
        results.append({
            "Model":      name,
            "CV Acc":     round(acc.mean(), 4),
            "CV F1":      round(f1.mean(),  4),
            "CV ROC-AUC": round(roc.mean(), 4),
            "Acc Std":    round(acc.std(),  4),
        })
        print(f"    Acc={acc.mean():.4f}  F1={f1.mean():.4f}  AUC={roc.mean():.4f}")
    return pd.DataFrame(results).sort_values("CV Acc", ascending=False)


def save_best(models: dict, comparison: pd.DataFrame, X, y) -> str:
    best_name = comparison.iloc[0]["Model"]
    pipeline  = models[best_name]
    pipeline.fit(X, y)
    artifact = {"model": pipeline, "features": FEATURE_COLS, "name": best_name}
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(artifact, path)
    print(f"\n✔  '{best_name}' saved to {path}")
    return best_name


def main():
    print("=" * 55)
    print("PSL MATCH PREDICTOR — Model Training")
    print("=" * 55)

    df = pd.read_csv(IN_FILE)
    X  = df[FEATURE_COLS].fillna(0.5)
    y  = df[TARGET_COL]
    print(f"Samples  : {len(X)}")
    print(f"Features : {FEATURE_COLS}")
    print(f"Class balance: {y.value_counts().to_dict()}\n")

    models     = build_models()
    comparison = evaluate(models, X, y)

    comp_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    comparison.to_csv(comp_path, index=False)

    print(f"\nModel Comparison:\n{comparison.to_string(index=False)}")
    best = save_best(models, comparison, X, y)
    print(f"✔  Comparison saved to {comp_path}")
    print(f"✔  Best model: {best}")


if __name__ == "__main__":
    main()
