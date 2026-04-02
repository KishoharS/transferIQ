import sys, os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Get directory of Home.py (/mount/src/transferiq/app)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up to root (/mount/src/transferiq)
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# Build paths
model_path = os.path.join(ROOT_DIR, "models", "catboost_model.pkl")
feature_path = os.path.join(ROOT_DIR, "models", "catboost_features.pkl")

# Load
model = joblib.load(model_path)
features = joblib.load(feature_path)

# ... rest of your code ...

st.title("⚽ Smart Scout Dashboard")
st.markdown("### The future of AI-driven football scouting.")
st.markdown(
    "Welcome to your command center. Use the **Market Value Estimator** below or explore player careers in the sidebar."
)

players = load_players()
transfers = load_transfers()

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Total Players Database", f"{len(players):,}", delta="Live Data")
with col_b:
    st.metric("Transfer Records", f"{len(transfers):,}", delta="Updated")
with col_c:
    st.metric("Prediction Model", "XGBoost v2.0", delta="Active")

st.markdown("---")

with st.container():
    st.subheader("💡 Market Value Estimator (CatBoost)")
    st.info(
        "Predict the theoretical market value of a player based on their stats and profile."
    )

# Define Top Leagues for UI
top_leagues_map = {
    "Premier League (GB1)": "GB1",
    "La Liga (ES1)": "ES1",
    "Serie A (IT1)": "IT1",
    "Bundesliga (L1)": "L1",
    "Ligue 1 (FR1)": "FR1",
    "Other": "Other",
}

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 16, 45, 25)
    # League Tier Input
    league_selection = st.selectbox("League Tier", list(top_leagues_map.keys()))
    league_id = top_leagues_map[league_selection]

    position = st.selectbox(
        "Position", ["Attack", "Midfield", "Defender", "Goalkeeper"]
    )

    # Contract Remaining Input
    contract_years = st.slider("Contract Years Remaining", 0.0, 5.0, 2.0, step=0.5)

with col2:
    goals = st.number_input(
        "Goals per 90 minutes", min_value=0.0, max_value=5.0, value=0.1, step=0.01
    )
    assists = st.number_input(
        "Assists per 90 minutes", min_value=0.0, max_value=5.0, value=0.1, step=0.01
    )
    cards = st.number_input(
        "Cards (Yellow+Red) per 90", min_value=0.0, max_value=2.0, value=0.1, step=0.01
    )

if st.button("Predict Market Value"):
    input_data = pd.DataFrame(0, index=[0], columns=features)

    input_data["age"] = age
    input_data["contract_remaining_months"] = contract_years * 12
    input_data["goals_per_90"] = goals
    input_data["assists_per_90"] = assists
    input_data["cards_per_90"] = cards

    # Handle One-Hot Encoding for Position
    position_col = f"position_{position}"
    if position_col in features:
        input_data[position_col] = 1

    # Handle One-Hot Encoding for League Tier
    # The training script uses 'league_tier_{value}'
    # If league_id is 'Other', it might be the dropped column (reference category) or explicit
    league_col = f"league_tier_{league_id}"
    if league_col in features:
        input_data[league_col] = 1

    # Predict (Log Transformed)
    prediction_log = model.predict(input_data)[0]
    prediction = np.expm1(prediction_log)

    st.success(f"The estimated market value is: €{prediction:,.0f}")

    with st.expander("Debug Information"):
        st.write(f"Model expects {len(features)} features")
        st.write(f"Input data shape: {input_data.shape}")
        st.write("Non-zero features in input:")
        non_zero = input_data.loc[0, input_data.loc[0] != 0]
        st.write(non_zero)
