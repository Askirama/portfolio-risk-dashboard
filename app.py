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
            # --- SECTION 4: RISK METRICS ---
            st.subheader("⚠️ Risk Metrics")

            # Sharpe Ratio (assuming risk free rate of 4.5% annually)
            risk_free_rate = 0.045 / 252
            sharpe_ratio = ((daily_returns.mean() - risk_free_rate) / daily_returns.std() * np.sqrt(252)).round(2)

            # Max Drawdown
            def max_drawdown(returns):
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.cummax()
                drawdown = (cumulative - rolling_max) / rolling_max
                return drawdown.min()

            max_dd = (daily_returns.apply(max_drawdown) * 100).round(2)

            # Beta (against S&P 500)
            sp500 = yf.download("^GSPC", start=start_date, end=end_date)["Close"]
            sp500_returns = sp500.pct_change().dropna()

            betas = {}
            for ticker in daily_returns.columns:
                # Align dates
                aligned = pd.concat([daily_returns[ticker], sp500_returns], axis=1).dropna()
                aligned.columns = ["stock", "market"]
                covariance = aligned.cov().iloc[0, 1]
                market_variance = aligned["market"].var()
                betas[ticker] = round(covariance / market_variance, 2)

            beta_series = pd.Series(betas)

            # Display risk metrics table
            risk_metrics = pd.DataFrame({
                "Sharpe Ratio": sharpe_ratio,
                "Max Drawdown (%)": max_dd,
                "Beta (vs S&P 500)": beta_series
            })

            st.dataframe(risk_metrics)

            # Explain the metrics simply
            st.markdown("""
            **Understanding the metrics:**
            - **Sharpe Ratio** — Above 1.0 is good, above 2.0 is excellent. Higher = better return for the risk taken
            - **Max Drawdown** — The worst % loss from peak to bottom. Closer to 0 is better
            - **Beta** — Above 1.0 means the stock is more volatile than the market. Below 1.0 means more stable
            """)

            # --- SECTION 5: MAX DRAWDOWN CHART ---
            st.subheader("📉 Drawdown Over Time")

            drawdown_df = pd.DataFrame()
            for ticker in daily_returns.columns:
                cumulative = (1 + daily_returns[ticker]).cumprod()
                rolling_max = cumulative.cummax()
                drawdown_df[ticker] = (cumulative - rolling_max) / rolling_max * 100

            fig3 = px.line(drawdown_df, title="Drawdown Over Time (%)")
            fig3.update_layout(xaxis_title="Date", yaxis_title="Drawdown (%)")
            st.plotly_chart(fig3, use_container_width=True)
              # --- SECTION 6: CORRELATION HEATMAP ---
            st.subheader("🔥 Correlation Heatmap")

            correlation_matrix = daily_returns.corr().round(2)

            fig4 = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns.tolist(),
                y=correlation_matrix.columns.tolist(),
                colorscale="RdYlGn",
                zmin=-1, zmax=1,
                text=correlation_matrix.values,
                texttemplate="%{text}",
                showscale=True
            ))

            fig4.update_layout(
                title="Stock Correlation Matrix — how stocks move together",
                xaxis_title="Stock",
                yaxis_title="Stock"
            )

            st.plotly_chart(fig4, use_container_width=True)

            # Interpretation
            st.markdown("""
            **Reading the heatmap:**
            - 🟢 **Green (close to 1.0)** — stocks move together, low diversification
            - 🟡 **Yellow (close to 0.0)** — little relationship between stocks
            - 🔴 **Red (close to -1.0)** — stocks move in opposite directions, great diversification
            """)