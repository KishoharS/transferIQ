import streamlit as st
import pandas as pd
import os

# data_loader.py is inside src/, so go up one level to project root
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')

@st.cache_data
def load_players():
    return pd.read_csv(os.path.join(RAW_DIR, 'players.csv'))

@st.cache_data
def load_clubs():
    return pd.read_csv(os.path.join(RAW_DIR, 'clubs.csv'))

@st.cache_data
def load_transfers():
    return pd.read_csv(os.path.join(RAW_DIR, 'transfers.csv'))

@st.cache_data
def load_valuations():
    return pd.read_csv(os.path.join(RAW_DIR, 'player_valuations.csv'))

@st.cache_data
def load_events(limit=None):
    path = os.path.join(RAW_DIR, 'game_events.csv')
    return pd.read_csv(path, nrows=limit) if limit else pd.read_csv(path)

@st.cache_data
def load_appearances(limit=None):
    path = os.path.join(RAW_DIR, 'appearances.csv')
    return pd.read_csv(path, nrows=limit) if limit else pd.read_csv(path)