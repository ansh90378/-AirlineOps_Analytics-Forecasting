# app.py
"""
Run: streamlit run app.py
Requires: clean_flights_for_bi.csv, aggregated_metrics.csv, risk_scores.csv
"""
import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Airline OCC Demo", layout="wide")

@st.cache_data(ttl=3600)
def load_data():
    flights = pd.read_csv(os.path.join('..\data\processed_data', 'clean_flights_for_bi.csv'), low_memory=False)
    agg = pd.read_csv(os.path.join('..\data\processed_data', 'aggregated_metrics.csv'), low_memory=False)
    scores = pd.read_csv(os.path.join('..\data\processed_data', 'risk_scores.csv'), low_memory=False)
    return flights, agg, scores

flights, agg, scores = load_data()

st.title("Airline OCC — Analytics & Next-day Risk (Demo)")
# Top KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Flights", int(flights.shape[0]))
col2.metric("Avg Arrival Delay (min)", round(flights["ARR_DELAY_MIN"].mean(),2))
col3.metric("Overall Delay %", f"{round(flights['IS_DELAY'].mean()*100,2)}%")

st.markdown("---")
# Risk section
st.header("Next-day Risk — High priority flights")
top = scores.head(50)
st.dataframe(top[["OP_CARRIER","ROUTE","prob_delay","CRS_DEP_HOUR","ROUTE_MEAN_ARR_DELAY"]])

st.markdown("### Filters")
carrier = st.selectbox("Carrier", options=["All"] + sorted(flights["OP_CARRIER"].unique().tolist()))
if carrier != "All":
    f = flights[flights["OP_CARRIER"]==carrier]
else:
    f = flights
st.markdown("### Delay heatmap (hour vs daypart)")
hm = f.groupby(["CRS_DEP_HOUR"])["ARR_DELAY_MIN"].mean().reset_index()
st.line_chart(hm.set_index("CRS_DEP_HOUR")["ARR_DELAY_MIN"])

st.markdown("---")
st.header("Drilldown: Route Performance")
route = st.selectbox("Route", options=["All"] + sorted(flights["ROUTE"].value_counts().head(200).index.tolist()))
if route != "All":
    sub = flights[flights["ROUTE"]==route]
    st.write("Route stats:", sub[["ARR_DELAY_MIN","IS_DELAY","IS_CANCELLED"]].describe())
    st.line_chart(sub.groupby("CRS_DEP_HOUR")["ARR_DELAY_MIN"].mean())
    # show any predicted risks for this route
    rs = scores[scores["ROUTE"]==route]
    if not rs.empty:
        st.subheader("Predicted next-day risk for this route")
        st.table(rs[["OP_CARRIER","ROUTE","prob_delay"]])

st.markdown("---")
st.header("Model & Data Notes")
st.write("""
- Model: XGBoost binary classifier (Delay>15min).  
- Encoding: frequency encoding for categorical features to be memory safe.  
- For production: add weather/real-time feeds and schedule automation.
""")
