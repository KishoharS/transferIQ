# Football Player Valuation Project (TransferIQ)

## Overview
This project focuses on predicting football players' market values using machine learning. It analyzes factors like age, position, performance stats, and league quality to estimate player valuations. The project includes data processing, exploratory analysis, model training, and a Streamlit web app for predictions.

## Objectives
- Predict player market values based on profile and stats
- Identify key factors influencing valuations
- Build an interactive dashboard for scouting

## Success Metrics
- R² Score: Measure of explained variance
- MAE/RMSE: Prediction accuracy in euros

## Dataset
Source: [Kaggle Player Scores Dataset](https://www.kaggle.com/datasets/davidcariboo/player-scores/code)

## Project Structure
```
football_valuation_project/
├── data/
│   ├── raw/           # Raw CSV files (players.csv, appearances.csv, etc.)
│   └── processed/     # Cleaned and processed data (model_ready_selected.csv, datascience_ready.csv)
├── src/               # Source code
│   ├── data_loader.py # Data loading functions
│   ├── train_xgboost.py # Model training script
│   └── ui.py          # UI utilities for Streamlit
├── models/            # Trained models and feature lists (xgboost_model.pkl, etc.)
├── notebooks/         # Jupyter notebooks for EDA and modeling
├── app/               # Streamlit app pages
├── images/            # Images for README or app
├── Home.py            # Main Streamlit app entry point
└── README.md          # This file
```

## Process Overview
1. **Data Cleaning**: Merged datasets, handled missing values, feature engineering
2. **EDA**: Explored distributions, correlations, created visualizations
3. **Feature Selection**: Identified impactful predictors
4. **Model Building**: Trained XGBoost regressor with hyperparameter tuning
5. **Evaluation**: Assessed performance with R², MAE, RMSE
6. **Deployment**: Built Streamlit app for predictions

## Tools and Libraries
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost, CatBoost
- Matplotlib, Seaborn, Plotly
- Streamlit
- Jupyter

## Results Summary
- **R² Score**: 0.52
- **MAE**: €1,462,334
- **RMSE**: €5,117,841

## Getting Started
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run Home.py`
4. Train model: `python src/train_xgboost.py`

## Future Improvements
- Add more features (social media, advanced stats)
- Try neural networks
 Best model: Random Forest Regressor
 Matrics:
  R² Score: 0.6230
  MAE: 0.7453
  RMSE: 0.9750

## Future work!!
 - Inxorporate more recent player data for higher accuracy.
 - Deploy the model using Flast for interactive use.
 - Will dive deeper into other aspects of this dataset like clubs, transfers, game_events. and so on!!

## My learning!!
- This project practically enables my understanding about how machiene learning actually works and what is real data science world look like. I learnt about how to train a model by without giving any target variables.
- This project also improves my exploration skills by enabling me to go beyond a certain limit and not just doing basic EDA with simple CSV file!!
- Looking forward to go beyond in this data world!!

## Future enhancements
- I built this model to help new people who've come into the football world and really know nothing about how players' market value will be determined.
- I think about building a web app which will be helpful for a end users and some of the official football clubs.
