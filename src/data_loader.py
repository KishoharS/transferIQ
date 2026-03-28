import streamlit as st
import pandas as pd
import os

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_CLEAN_DIR = os.path.join(BASE_DIR, 'data', 'processed')

@st.cache_data
def load_players():
    """Loads the players.csv dataset."""
    path = os.path.join(DATA_RAW_DIR, 'players.csv')
    return pd.read_csv(path)

@st.cache_data
def load_clubs():
    """Loads the clubs.csv dataset."""
    path = os.path.join(DATA_RAW_DIR, 'clubs.csv')
    return pd.read_csv(path)

@st.cache_data
def load_transfers():
    """Loads the transfers.csv dataset."""
    path = os.path.join(DATA_RAW_DIR, 'transfers.csv')
    return pd.read_csv(path)

@st.cache_data
def load_valuations():
    """Loads the player_valuations.csv dataset."""
    path = os.path.join(DATA_RAW_DIR, 'player_valuations.csv')
    return pd.read_csv(path)

@st.cache_data
def load_events(limit=None):
    """
    Loads game_events.csv. 
    Warning: This file can be large. Use 'limit' for testing.
    """
    path = os.path.join(DATA_RAW_DIR, 'game_events.csv')
    if limit:
        return pd.read_csv(path, nrows=limit)
    return pd.read_csv(path)

@st.cache_data
def load_appearances(limit=None):
    """Loads appearances.csv."""
    path = os.path.join(DATA_RAW_DIR, 'appearances.csv')
    if limit:
        return pd.read_csv(path, nrows=limit)
    return pd.read_csv(path)
