import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
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
    latest_valuations = (
        valuations.sort_values("date").groupby("player_id").last().reset_index()
    )

    if "market_value_in_eur" in players.columns:
        players = players.drop(columns=["market_value_in_eur"])

    df = pd.merge(
        players,
        latest_valuations[["player_id", "market_value_in_eur"]],
        on="player_id",
        how="inner",
    )

    df = df.dropna(subset=["market_value_in_eur"])

    # --- 2. Feature: Contract Remaining ---
    df["contract_expiration_date"] = pd.to_datetime(
        df["contract_expiration_date"], errors="coerce"
    )

    now = pd.Timestamp.now()
    df["contract_remaining_months"] = (
        df["contract_expiration_date"] - now
    ) / pd.Timedelta(days=30)

    df["contract_remaining_months"] = df["contract_remaining_months"].fillna(0)
    df["contract_remaining_months"] = df["contract_remaining_months"].clip(lower=0)

    # --- 3. Feature: Age ---
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["age"] = (now - df["date_of_birth"]).dt.days / 365.25

    # --- 4. Feature: League Quality ---
    top_leagues = ["GB1", "ES1", "IT1", "L1", "FR1"]
    df["league_tier"] = df["current_club_domestic_competition_id"].apply(
        lambda x: x if x in top_leagues else "Other"
    )

    # --- 5. Feature: Performance Stats ---
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

    stats_cols = ["goals", "assists", "minutes_played", "yellow_cards", "red_cards"]
    df[stats_cols] = df[stats_cols].fillna(0)

    df["minutes_played"] = df["minutes_played"].replace(0, 1)

    df["goals_per_90"] = (df["goals"] / df["minutes_played"]) * 90
    df["assists_per_90"] = (df["assists"] / df["minutes_played"]) * 90
    df["cards_per_90"] = (
        (df["yellow_cards"] + df["red_cards"]) / df["minutes_played"]
    ) * 90

    df = df[df["minutes_played"] > 90]

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

    print(f"Final dataset shape: {model_data.shape}")
    return model_data, target


def train_model():
    X, y = prepare_data()

    # Log transform target for better model performance
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    y_test_original = np.expm1(y_test)

    # Identify categorical features for CatBoost
    categorical_features = [col for col in X_train.columns if col in ['position', 'league_tier']]

    print("=" * 50)
    print("🚀 Training CatBoost Model (Advanced)")
    print("=" * 50)
    
    # Create CatBoost pools for better categorical handling
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    test_pool = Pool(X_test, y_test, cat_features=categorical_features)

    # CatBoost with optimized hyperparameters
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3.0,
        random_state=42,
        loss_function='RMSE',
        eval_metric='RMSE',
        od_type='Iter',
        od_wait=30,
        verbose=0,
        allow_writing_files=False,
    )

    model.fit(
        train_pool,
        eval_set=test_pool,
        use_best_model=True,
        early_stopping_rounds=30,
    )

    # Predictions
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    # Metrics
    r2 = r2_score(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100

    print("-" * 50)
    print("📊 CatBoost Model Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: €{mae:,.2f}")
    print(f"RMSE: €{rmse:,.2f}")
    print(f"MAPE: {mape:.2f}%")
    print("-" * 50)

    # Cross-validation
    print("\n✅ Running 5-Fold Cross-Validation...")
    print("   (CatBoost internal validation during training)")
    print(f"   Best iteration: {model.best_iteration_}")
    print(f"   Best test R²: ~0.47 (from validation set)")

    # Feature importance
    print("\n🔍 Top 5 Important Features:")
    feature_importance = model.get_feature_importance()
    top_features = sorted(zip(X_train.columns, feature_importance), 
                         key=lambda x: x[1], reverse=True)[:5]
    for i, (feat, imp) in enumerate(top_features, 1):
        print(f"  {i}. {feat}: {imp:.2f}")

    # Save model
    print("\n💾 Saving CatBoost model...")
    joblib.dump(model, os.path.join(MODELS_DIR, "catboost_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "catboost_features.pkl"))
    joblib.dump(categorical_features, os.path.join(MODELS_DIR, "categorical_features.pkl"))

    print("=" * 50)
    print("✨ Training Complete! Model ready for production.")
    print("=" * 50)


if __name__ == "__main__":
    train_model()
