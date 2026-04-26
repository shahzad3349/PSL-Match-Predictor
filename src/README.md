# 🏏 PSL Machine Learning Project

## Overview
Pakistan Super League (PSL) ball-by-ball data par complete ML project.
**Goal:** 1st innings ke baad predict karo ki match kaun jeetega.

## Dataset
- **Source:** `data/psl_data.csv`
- **Records:** 73,784 ball-by-ball entries
- **Seasons:** PSL 2016–2025
- **Teams:** 6 (Islamabad United, Karachi Kings, Lahore Qalandars, Multan Sultans, Peshawar Zalmi, Quetta Gladiators)

## Project Structure
```
psl_ml_project/
├── data/
│   ├── psl_data.csv              # Raw ball-by-ball data
│   └── psl_match_features.csv    # Processed match-level features (Notebook 3 ke baad banta hai)
├── notebooks/
│   ├── 01_data_loading.ipynb     # Data load, overview, missing values
│   ├── 02_eda.ipynb              # Exploratory Data Analysis + visualizations
│   ├── 03_preprocessing.ipynb   # Feature engineering + encoding
│   ├── 04_model_training.ipynb  # 5 models train + evaluate
│   └── 05_prediction.ipynb      # Ready-to-use prediction pipeline
├── outputs/
│   ├── models/                   # Saved ML models (.pkl files)
│   └── plots/                    # Saved visualization images
├── src/                          # Helper scripts (future use)
└── README.md
```

## Notebooks Run Karne Ka Order
1. `01_data_loading.ipynb` — pehle chalao
2. `02_eda.ipynb`
3. `03_preprocessing.ipynb` ← ye `psl_match_features.csv` banata hai
4. `04_model_training.ipynb`
5. `05_prediction.ipynb`

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
