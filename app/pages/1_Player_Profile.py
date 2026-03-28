import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_appearances, load_events, load_players
from src.ui import apply_custom_style, render_sidebar

st.set_page_config(page_title="Player Performance", layout="wide", page_icon="🏃")
apply_custom_style()
render_sidebar()

st.title("🏃 Player Performance Profile")
st.markdown("### Deep dive into match stats and season trajectory.")

with st.spinner("Loading Data..."):
    players = load_players()
    appearances = load_appearances(limit=100000)
    pass

players = load_players()
app_df = load_appearances()
events_df = load_events()

player_names = players["name"].unique()
selected_player_name = st.selectbox(
    "Select Player",
    player_names,
    index=list(player_names).index("Cristiano Ronaldo")
    if "Cristiano Ronaldo" in player_names
    else 0,
)

player_id = players[players["name"] == selected_player_name].iloc[0]["player_id"]
p_apps = app_df[app_df["player_id"] == player_id]
p_events = events_df[events_df["player_id"] == player_id]

col1, col2, col3, col4 = st.columns(4)
total_goals = p_apps["goals"].sum()
total_assists = p_apps["assists"].sum()
total_mins = p_apps["minutes_played"].sum()
total_cards = p_apps["yellow_cards"].sum() + p_apps["red_cards"].sum()

with col1:
    st.metric("Total Goals", int(total_goals))
with col2:
    st.metric("Total Assists", int(total_assists))
with col3:
    st.metric("Minutes Played", f"{int(total_mins):,}")
with col4:
    st.metric("Total Cards", int(total_cards))

st.divider()

c1, c2 = st.columns(2)

with c1:
    st.subheader("Season Trajectory")
    if not p_apps.empty:
        p_apps["year"] = pd.to_datetime(p_apps["date"]).dt.year
        yearly_stats = p_apps.groupby("year")[["goals", "assists"]].sum().reset_index()

        fig_season = px.bar(
            yearly_stats,
            x="year",
            y=["goals", "assists"],
            barmode="group",
            title="Goals & Assists by Year",
        )
        st.plotly_chart(fig_season, use_container_width=True)
    else:
        st.info("No appearance data found.")

with c2:
    st.subheader("Discipline & Activity")
    if not p_events.empty:
        event_counts = p_events["type"].value_counts().reset_index()
        event_counts.columns = ["Event Type", "Count"]
        fig_pie = px.pie(
            event_counts,
            values="Count",
            names="Event Type",
            hole=0.4,
            title="Event Distribution",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No event data found.")

st.subheader("Player Radar (Per 90 Min)")

if total_mins > 0:
    goals_p90 = (total_goals / total_mins) * 90
    assists_p90 = (total_assists / total_mins) * 90
    cards_p90 = (total_cards / total_mins) * 90

    categories = ["Goals/90", "Assists/90", "Cards/90"]

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=[goals_p90, assists_p90, cards_p90],
            theta=categories,
            fill="toself",
            name=selected_player_name,
        )
    )

    fig_radar.add_trace(
        go.Scatterpolar(
            r=[0.5, 0.3, 0.1],
            theta=categories,
            fill="toself",
            name="Elite Benchmark",
            line=dict(dash="dot"),
        )
    )

    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)
