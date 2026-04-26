"""
06_model_evaluation.py
----------------------
Step 6: Full evaluation of the best saved model.

Outputs:
  - Classification report printed to console
  - confusion_matrix.png
  - roc_curve.png
  - feature_importance.png
  - model_comparison bar chart  →  model_comparison_chart.png
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay,
)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "graphs")
IN_FILE    = os.path.join(PROC_DIR, "03_featured.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "best_model.pkl")

FEATURE_COLS = [
    "team1_win_rate", "team2_win_rate", "h2h_team1_win_rate",
    "team1_recent_form", "team2_recent_form",
    "venue_team1_win_rate", "team1_avg_inning1",
]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor":   "white",
    "axes.spines.top":  False,   "axes.spines.right": False,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,      "axes.labelsize":    11,
})


def save(fig, name):
    path = os.path.join(GRAPHS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def plot_confusion_matrix(y_test, y_pred):
    cm   = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Team2 Wins", "Team1 Wins"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    save(fig, "confusion_matrix.png")


def plot_roc(model, X_test, y_test):
    y_prob      = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#1a6b8a", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "roc_curve.png")
    return roc_auc


def plot_feature_importance(pipeline, model_name):
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        return
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return
    feat_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(feat_df["Feature"], feat_df["Importance"], color="#2e9b6f")
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    save(fig, "feature_importance.png")


def plot_model_comparison():
    comp_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    if not os.path.exists(comp_path):
        return
    comp = pd.read_csv(comp_path)
    metrics = ["CV Acc", "CV F1", "CV ROC-AUC"]
    x = np.arange(len(comp))
    width = 0.25
    colors = ["#1a6b8a", "#2e9b6f", "#c9a227"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, comp[metric], width, label=metric, color=color)
    ax.set_xticks(x + width)
    ax.set_xticklabels(comp["Model"], rotation=10)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "model_comparison_chart.png")


def main():
    print("=" * 55)
    print("PSL MATCH PREDICTOR — Model Evaluation")
    print("=" * 55)

    df = pd.read_csv(IN_FILE)
    X  = df[FEATURE_COLS].fillna(0.5)
    y  = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model_obj  = joblib.load(MODEL_FILE)
    model_name = model_obj["name"]
    pipeline   = model_obj["model"]
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\nBest Model   : {model_name}")
    print(f"Test samples : {len(y_test)}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Team2 Wins", "Team1 Wins"]))

    roc_auc = plot_roc(pipeline, X_test, y_test)
    print(f"ROC-AUC      : {roc_auc:.4f}")
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(pipeline, model_name)
    plot_model_comparison()

    comp_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    if os.path.exists(comp_path):
        print(f"\nAll Models:\n{pd.read_csv(comp_path).to_string(index=False)}")

    print(f"\n✔  Evaluation complete. All outputs in {GRAPHS_DIR}")


if __name__ == "__main__":
    main()
