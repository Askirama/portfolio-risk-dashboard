import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
st.title("📊 Stock Portfolio Risk & Performance Dashboard")

# --- USER INPUT ---
st.sidebar.header("Build Your Portfolio")

tickers_input = st.sidebar.text_input(
    "Enter stock tickers (comma separated)",
    value="AAPL, MSFT, TSLA"
)

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# --- FETCH DATA ---
if st.sidebar.button("Analyse Portfolio"):
    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    st.subheader(f"Analysing: {', '.join(tickers)}")

    raw_data = yf.download(tickers, start=start_date, end=end_date)["Close"]

    if raw_data.empty:
        st.error("No data found. Check your tickers and try again.")
    else:
        st.success("Data loaded successfully!")

        # Show raw closing prices
        st.subheader("📈 Closing Prices Over Time")
        fig = px.line(raw_data, title="Stock Closing Prices")
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

        # Show raw data table
        st.subheader("📋 Raw Data")
        st.dataframe(raw_data.tail(10))