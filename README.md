# TransferIQ — Football Player Valuation

> Predicts football player market values using CatBoost regression, with an interactive Streamlit dashboard for scouting and exploration.

---

## Overview

TransferIQ analyzes player attributes — age, position, performance statistics, and league quality — to estimate transfer market valuations. The project covers the full ML pipeline: data ingestion, exploratory analysis, feature selection, model training, and a deployed web interface.

## Project Structure

```
transferIQ/
├── app/                    # Streamlit app pages
├── images/                 # README and app assets
├── notebooks/              # EDA and modeling notebooks
├── src/
│   ├── data_loader.py      # Data ingestion and merging
│   ├── train_catboost.py   # Model training script
│   └── ui.py               # Streamlit UI utilities
├── data/
│   ├── raw/                # Source CSV files (gitignored)
│   └── processed/          # Model-ready datasets (gitignored)
├── Home.py                 # Streamlit entry point
└── requirements.txt
```

## Dataset

Source: [Kaggle — Player Scores Dataset](https://www.kaggle.com/datasets/davidcariboo/player-scores/code) by David Cariboo. Includes player profiles, appearance statistics, and club metadata across major European leagues.

## Methodology

| Stage | Details |
|---|---|
| Data cleaning | Multi-file merge, missing value imputation, feature engineering |
| EDA | Distribution analysis, correlation heatmaps, positional breakdowns |
| Feature selection | Correlation filtering and CatBoost feature importance |
| Model | CatBoost regressor with hyperparameter tuning via grid search |
| Evaluation | R², MAE, and RMSE on held-out test set |
| Deployment | Streamlit app for live predictions and scouting view |

## Getting Started

```bash
# Clone the repository
git clone https://github.com/KishoharS/transferIQ.git
cd transferIQ

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train_catboost.py

# Launch the app
streamlit run Home.py
```

## Tech Stack

- **Language**: Python
- **Modelling**: CatBoost, scikit-learn
- **Data**: pandas, NumPy
- **Visualisation**: Plotly, Matplotlib, Seaborn
- **App**: Streamlit

## Limitations

- Model is position-agnostic; a striker and a goalkeeper with similar stats receive similar valuations
- Training data excludes non-European leagues and players with sparse appearance records
- Transfer fees in the dataset reflect historical market conditions and may not reflect current valuations

## Roadmap

- Incorporate recent season data for improved accuracy
- Build position-specific sub-models
- Deploy via Flask REST API for programmatic access
- Extend analysis to club-level and game event data

## Author

**Kishohar S** — B.Tech, AI and Data Science  
[GitHub](https://github.com/KishoharS) · [LinkedIn](https://linkedin.com/in/kishohar)
