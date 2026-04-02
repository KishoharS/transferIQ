import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

import plotly.graph_objects as go
import streamlit as st
from data_loader import load_players, load_transfers, load_valuations
from ui import apply_custom_style, render_sidebar

st.set_page_config(page_title="Transfer Market Analysis", layout="wide", page_icon="💰")
apply_custom_style()
render_sidebar()

st.title("💰 Transfer Market Efficiency")
st.markdown("### Compare estimated value vs. actual fees paid.")

with st.spinner("Loading Data..."):
    players = load_players()
    transfers = load_transfers()
    valuations = load_valuations()

player_names = players["name"].unique()
selected_player_name = st.selectbox(
    "Select Player",
    player_names,
    index=list(player_names).index("Lionel Messi")
    if "Lionel Messi" in player_names
    else 0,
)

player_row = players[players["name"] == selected_player_name].iloc[0]
player_id = player_row["player_id"]

player_vals = valuations[valuations["player_id"] == player_id].sort_values("date")
player_transfers = transfers[transfers["player_id"] == player_id].sort_values(
    "transfer_date"
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=player_vals["date"],
        y=player_vals["market_value_in_eur"],
        mode="lines",
        name="Market Value",
        line=dict(color="#00CC96", width=3),
    )
)

paid_transfers = player_transfers[player_transfers["transfer_fee"] > 0]
free_transfers = player_transfers[player_transfers["transfer_fee"] == 0]

fig.add_trace(
    go.Scatter(
        x=paid_transfers["transfer_date"],
        y=paid_transfers["transfer_fee"],
        mode="markers",
        name="Transfer Fee (Paid)",
        marker=dict(color="#EF553B", size=12, symbol="diamond"),
        text=paid_transfers["from_club_name"] + " ➔ " + paid_transfers["to_club_name"],
        hovertemplate="Fee: €%{y:,.0f}<br>%{text}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=free_transfers["transfer_date"],
        y=[0] * len(free_transfers),
        mode="markers",
        name="Free Transfer",
        marker=dict(color="#636EFA", size=8, symbol="circle-open"),
        text=free_transfers["from_club_name"] + " ➔ " + free_transfers["to_club_name"],
        hovertemplate="Free Transfer<br>%{text}<extra></extra>",
    )
)

fig.update_layout(
    title=f"Valuation History: {selected_player_name}",
    xaxis_title="Year",
    yaxis_title="Value (€)",
    hovermode="x unified",
    height=600,
)

st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)

current_val = (
    player_vals.iloc[-1]["market_value_in_eur"] if not player_vals.empty else 0
)
max_val = player_vals["market_value_in_eur"].max() if not player_vals.empty else 0
total_fees = player_transfers["transfer_fee"].sum()

with col1:
    st.metric("Current Market Value", f"€{current_val:,.0f}")
with col2:
    st.metric("Peak Market Value", f"€{max_val:,.0f}")
with col3:
    st.metric(
        "Total Fees Paid (Career)",
        f"€{total_fees:,.0f}",
        delta=f"{(current_val - total_fees):,.0f} ROI" if total_fees > 0 else None,
        delta_color="normal",
    )

st.subheader("Transfer History")
st.dataframe(
    player_transfers[
        [
            "transfer_date",
            "from_club_name",
            "to_club_name",
            "transfer_fee",
            "market_value_in_eur",
        ]
    ]
    .rename(
        columns={
            "transfer_date": "Date",
            "from_club_name": "From",
            "to_club_name": "To",
            "transfer_fee": "Fee (€)",
            "market_value_in_eur": "Value at Time (€)",
        }
    )
    .style.format({"Fee (€)": "€{:,.0f}", "Value at Time (€)": "€{:,.0f}"})
)
