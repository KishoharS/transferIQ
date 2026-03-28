import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_loader import load_appearances, load_players, load_valuations

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def prepare_data():
    print("Loading data...")
    players = load_players()
    appearances = load_appearances()
    valuations = load_valuations()

    print("Data loaded. Starting feature engineering...")

    # --- 1. Target Variable (Market Value) ---
    # Get the latest valuation for each player
    latest_valuations = (
        valuations.sort_values("date").groupby("player_id").last().reset_index()
    )

    # Drop market_value_in_eur from players if it exists to avoid collision
    if "market_value_in_eur" in players.columns:
        players = players.drop(columns=["market_value_in_eur"])

    df = pd.merge(
        players,
        latest_valuations[["player_id", "market_value_in_eur"]],
        on="player_id",
        how="inner",
    )

    # Remove players with no market value
    df = df.dropna(subset=["market_value_in_eur"])

    # --- 2. Feature: Contract Remaining ---
    # Convert contract_expiration_date to datetime
    df["contract_expiration_date"] = pd.to_datetime(
        df["contract_expiration_date"], errors="coerce"
    )

    # Calculate months remaining from "now" (or a fixed date if we want reproducibility, currently using today)
    now = pd.Timestamp.now()
    df["contract_remaining_months"] = (
        df["contract_expiration_date"] - now
    ) / pd.Timedelta(days=30)

    # Fill missing contracts with 0 (assuming expired or unknown) or median
    df["contract_remaining_months"] = df["contract_remaining_months"].fillna(0)
    # Clip negative values to 0
    df["contract_remaining_months"] = df["contract_remaining_months"].clip(lower=0)

    # --- 3. Feature: Age ---
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["age"] = (now - df["date_of_birth"]).dt.days / 365.25

    # --- 4. Feature: Position (One-Hot Encoding) ---
    # Simplify position if needed, but 'position' column seems fine (Attack, Defender, Midfield, Goalkeeper)
    # We will one-hot encode this later.

    # --- 5. Feature: League Tier / Quality ---
    # We can use 'current_club_domestic_competition_id'.
    # For simplicity, we will One-Hot Encode the top leagues and group others as 'Other'.
    top_leagues = [
        "GB1",
        "ES1",
        "IT1",
        "L1",
        "FR1",
    ]  # Premier League, La Liga, Serie A, Bundesliga, Ligue 1
    df["league_tier"] = df["current_club_domestic_competition_id"].apply(
        lambda x: x if x in top_leagues else "Other"
    )

    # --- 6. Feature: Performance Stats (Goals, Assists, Minutes) ---
    # Aggregate appearances by player
    # We want per-game or per-90 stats to be comparable
    stats = (
        appearances.groupby("player_id")
        .agg(
            {
                "goals": "sum",
                "assists": "sum",
                "minutes_played": "sum",
                "yellow_cards": "sum",
                "red_cards": "sum",
            }
        )
        .reset_index()
    )

    df = pd.merge(df, stats, on="player_id", how="left")

    # Fill NaN stats with 0 (players who haven't played)
    stats_cols = ["goals", "assists", "minutes_played", "yellow_cards", "red_cards"]
    df[stats_cols] = df[stats_cols].fillna(0)

    # Calculate per 90 stats
    # Avoid division by zero
    df["minutes_played"] = df["minutes_played"].replace(0, 1)

    df["goals_per_90"] = (df["goals"] / df["minutes_played"]) * 90
    df["assists_per_90"] = (df["assists"] / df["minutes_played"]) * 90
    df["cards_per_90"] = (
        (df["yellow_cards"] + df["red_cards"]) / df["minutes_played"]
    ) * 90

    # Filter out players with very few minutes to avoid noise (optional but recommended)
    df = df[df["minutes_played"] > 90]  # At least 1 full game

    # --- Feature Selection ---
    features_to_use = [
        "age",
        "contract_remaining_months",
        "goals_per_90",
        "assists_per_90",
        "cards_per_90",
        "position",
        "league_tier",
    ]

    model_data = df[features_to_use].copy()
    target = df["market_value_in_eur"]

    # Handle Categorical Variables
    model_data = pd.get_dummies(
        model_data, columns=["position", "league_tier"], drop_first=True
    )

    print(f"Final dataset shape: {model_data.shape}")
    return model_data, target


def train_model():
    X, y = prepare_data()

    # Log transform target because market values are highly skewed
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    print("Training XGBoost Model...")
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluation
    predictions_log = model.predict(X_test)
    predictions = np.expm1(predictions_log)
    y_test_original = np.expm1(y_test)

    r2 = r2_score(y_test_original, predictions)
    mae = mean_absolute_error(y_test_original, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_original, predictions))

    print("-" * 30)
    print("Model Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: €{mae:,.2f}")
    print(f"RMSE: €{rmse:,.2f}")
    print("-" * 30)

    # Save Model and Features
    print("Saving model artifacts...")
    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "xgboost_features.pkl"))
    print("Done!")


if __name__ == "__main__":
    train_model()
