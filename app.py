import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
        # Drop columns that are completely empty
        raw_data = raw_data.dropna(axis=1, how='all')

        if raw_data.empty or len(raw_data) < 2:
            st.error("Data loaded but was empty — likely a connection issue. Try again or switch to a hotspot.")
        else:
            st.success("Data loaded successfully!")

            # --- DAILY RETURNS ---
            daily_returns = raw_data.pct_change().dropna()

            # --- CUMULATIVE RETURNS ---
            cumulative_returns = (1 + daily_returns).cumprod()

            # --- SECTION 1: CUMULATIVE RETURNS CHART ---
            st.subheader("📈 Cumulative Returns Over Time")
            fig1 = px.line(cumulative_returns,
                           title="Cumulative Returns — How $1 invested grew over time")
            fig1.update_layout(xaxis_title="Date", yaxis_title="Growth of $1")
            st.plotly_chart(fig1, use_container_width=True)

            # --- SECTION 2: DAILY RETURNS CHART ---
            st.subheader("📉 Daily Returns")
            fig2 = px.line(daily_returns, title="Daily Returns per Stock")
            fig2.update_layout(xaxis_title="Date", yaxis_title="Daily Return (%)")
            st.plotly_chart(fig2, use_container_width=True)

            # --- SECTION 3: SUMMARY STATS ---
            st.subheader("📋 Performance Summary")
            summary = pd.DataFrame({
                "Total Return (%)": ((cumulative_returns.iloc[-1] - 1) * 100).round(2),
                "Avg Daily Return (%)": (daily_returns.mean() * 100).round(4),
                "Volatility (%)": (daily_returns.std() * 100).round(4),
                "Best Day (%)": (daily_returns.max() * 100).round(2),
                "Worst Day (%)": (daily_returns.min() * 100).round(2),
            })
            st.dataframe(summary)