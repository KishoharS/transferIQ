import streamlit as st
import pandas as pd
import os

# 1. Get the absolute path of the 'src' directory
# This will be /mount/src/transferiq/src on Streamlit Cloud
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the project root (transferiq)
BASE_DIR = os.path.dirname(SRC_DIR)

# 3. Path to your data folder
# This ensures we look in /mount/src/transferiq/data
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