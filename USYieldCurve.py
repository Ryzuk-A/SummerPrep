import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from fredapi import Fred

# Streamlit page setup
st.set_page_config(page_title="US Treasury Yield Curve", layout="centered")
st.title("US Treasury Yield Curve Dashboard")
st.markdown("Live US Treasury yields from FRED (Federal Reserve Economic Data).")

# Sidebar for FRED API Key input
api_key = st.sidebar.text_input("Enter your FRED API key", type="password")
if not api_key:
    st.warning("Please enter your FRED API key in the sidebar.")
    st.stop()
# API key: 2b7014bef5eb1be6ad45c00dbb77330b

# Connect to FRED
fred = Fred(api_key=api_key)

# Define available FRED tickers
fred_tickers = {
    '1M': 'DGS1MO',
    '3M': 'DGS3MO',
    '6M': 'DGS6MO',
    '1Y': 'DGS1',
    '2Y': 'DGS2',
    '3Y': 'DGS3',
    '5Y': 'DGS5',
    '7Y': 'DGS7',
    '10Y': 'DGS10',
    '20Y': 'DGS20',
    '30Y': 'DGS30',
}

# Function to retrieve the latest available yield data
@st.cache_data(ttl=3600)  # Refresh hourly
def fetch_yields():
    yields = {}
    for maturity, series in fred_tickers.items():
        data = fred.get_series(series).dropna()
        yields[maturity] = data.iloc[-1]
    return pd.Series(yields)

# Fetch and sort the data
try:
    yield_curve = fetch_yields()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Desired maturity order without 2M
maturity_order = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
yield_curve = yield_curve.reindex(maturity_order)

# Plot the yield curve
st.subheader("Current Yield Curve")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(yield_curve.index, yield_curve.values, marker='o', linestyle='-')
ax.set_title(f'US Treasury Yield Curve as of {datetime.today().date()}')
ax.set_xlabel("Maturity")
ax.set_ylabel("Yield (%)")
ax.grid(True)
st.pyplot(fig)

# Optionally show raw data
if st.checkbox("Show raw yield data"):
    st.dataframe(yield_curve.rename("Yield (%)"))



