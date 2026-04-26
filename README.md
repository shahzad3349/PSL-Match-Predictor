# 🏏 PSL Machine Learning Project

## Overview
A complete machine learning project based on Pakistan Super League (PSL) ball-by-ball data.
Goal: Predict the match winner after the first innings..

## Dataset
- **Source:** `data/psl_data.csv`
- **Records:** 73,784 ball-by-ball entries
- **Seasons:** PSL 2016–2025
- **Teams:** 6 (Islamabad United, Karachi Kings, Lahore Qalandars, Multan Sultans, Peshawar Zalmi, Quetta Gladiators)

## Project Structure

psl_ml_project/
├── data/
│   ├── psl_data.csv              # Raw ball-by-ball data
│   └── psl_match_features.csv    # Processed match-level features (generated after Notebook 3)
├── notebooks/
│   ├── 01_data_loading.ipynb     # Data loading, overview, missing values
│   ├── 02_eda.ipynb              # Exploratory Data Analysis and visualizations
│   ├── 03_preprocessing.ipynb    # Feature engineering and encoding
│   ├── 04_model_training.ipynb   # Train and evaluate 5 models
│   └── 05_prediction.ipynb       # Ready-to-use prediction pipeline
├── outputs/
│   ├── models/                   # Saved ML models (.pkl files)
│   └── plots/                    # Saved visualization images
├── src/                          # Helper scripts (for future use)
└── README.md
## Notebook Execution Order
01_data_loading.ipynb — Run this first
02_eda.ipynb
03_preprocessing.ipynb ← This generates psl_match_features.csv
04_model_training.ipynb
05_prediction.ipynb

## Models Used
| Model | Type |
|-------|------|
| Logistic Regression | Linear |
| Decision Tree | Tree-based |
| **Random Forest** | Ensemble ✅ Best |
| Gradient Boosting | Ensemble |
| SVM | Kernel-based |

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```
