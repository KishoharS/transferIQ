import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_model(data_path: str, model_save_path: str):
    """Trains the CatBoost valuation model and saves it to disk."""
    print(f"Loading processed data from {data_path}...")
    
    # 1. LOAD DATA
    # Using the dataset you exported at the end of Notebook 5
    df = pd.read_csv(data_path)
    
    # 2. FEATURE SELECTION
    # These are the exact top features you identified via permutation importance
    features = [
        'age', 'minutes_played', 'goals_per90', 'assists_per90', 
        'position', 'current_club_domestic_competition_id', 
        'last_season', 'country_of_citizenship'
    ]
    
    # Using the log-transformed target you created to handle skewness
    target = 'target_log' 
    
    # Clean up any lingering missing values just in case
    df = df.dropna(subset=[target] + features)
    
    X = df[features].copy()
    y = df[target]
    
    # Tell CatBoost which features are categorical so it handles them natively
    cat_features = ['position', 'current_club_domestic_competition_id', 'country_of_citizenship']
    
    # Fill categorical NAs with 'Unknown' and numeric NAs with 0
    X.loc[:, cat_features] = X.loc[:, cat_features].fillna('Unknown')
    numeric_features = [col for col in features if col not in cat_features]
    X.loc[:, numeric_features] = X.loc[:, numeric_features].fillna(0)
    
    # 3. SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. TRAIN MODEL
    print("Training CatBoost model...")
    model = CatBoostRegressor(
        iterations=500, 
        learning_rate=0.1, 
        depth=6, 
        cat_features=cat_features,
        verbose=100,
        random_seed=42
    )
    
    # Fit with early stopping to prevent overfitting
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    
    # 5. EVALUATE
    print("Evaluating model...")
    predictions_log = model.predict(X_test)
    
    # Convert log predictions back to actual euros using expm1 for accurate MAE/RMSE
    predictions_eur = np.expm1(predictions_log)
    y_test_eur = np.expm1(y_test)
    
    r2 = r2_score(y_test_eur, predictions_eur)
    mae = mean_absolute_error(y_test_eur, predictions_eur)
    rmse = np.sqrt(mean_squared_error(y_test_eur, predictions_eur))
    
    print(f"\n--- Final Metrics ---")
    print(f"R2 Score: {r2:.2f}")
    print(f"MAE: €{mae:,.0f}")
    print(f"RMSE: €{rmse:,.0f}")
    print(f"---------------------\n")
    
    # 6. SAVE MODEL
    print(f"Saving model to {model_save_path}...")
    joblib.dump(model, model_save_path)
    print("Pipeline complete!")

if __name__ == "__main__":
    # Ensure these paths match your folder structure
    # You saved this CSV in Notebook 5
    train_model(
        data_path="data_clean/model_ready_selected.csv", 
        model_save_path="models/catboost_model.pkl"
    )