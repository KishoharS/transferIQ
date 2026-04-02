import streamlit as st
import pandas as pd
import os

# 1. Determine the path to the 'src' folder where this file lives
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the project root (transferiq)
BASE_DIR = os.path.dirname(SRC_DIR)

# 3. Define the path to your data folder
# Note: Ensure your CSVs are directly inside the 'data' folder on GitHub
DATA_DIR = os.path.join(BASE_DIR, 'data')

@st.cache_data
def load_players():
    """Loads the players.csv dataset."""
    path = os.path.join(DATA_DIR, 'players.csv')
    return pd.read_csv(path)

@st.cache_data
def load_clubs():
    """Loads the clubs.csv dataset."""
    path = os.path.join(DATA_DIR, 'clubs.csv')
    return pd.read_csv(path)

@st.cache_data
def load_transfers():
    """Loads the transfers.csv dataset."""
    path = os.path.join(DATA_DIR, 'transfers.csv')
    return pd.read_csv(path)

@st.cache_data
def load_valuations():
    """Loads the player_valuations.csv dataset."""
    path = os.path.join(DATA_DIR, 'player_valuations.csv')
    return pd.read_csv(path)

@st.cache_data
def load_events(limit=None):
    """Loads game_events.csv."""
    path = os.path.join(DATA_DIR, 'game_events.csv')
    if limit:
        return pd.read_csv(path, nrows=limit)
    return pd.read_csv(path)

@st.cache_data
def load_appearances(limit=None):
    """Loads appearances.csv."""
    path = os.path.join(DATA_DIR, 'appearances.csv')
    if limit:
        return pd.read_csv(path, nrows=limit)
    return pd.read_csv(path)