# 📊 Stock Portfolio Risk & Performance Dashboard

A professional-grade web application for analysing stock portfolio performance and risk — built with Python and Streamlit.

## 🔗 Live App
[Click here to open the dashboard](#) ← we'll update this link after deployment

## 📌 What It Does
Input your stocks, number of shares, buy prices and date range to instantly get:

- **Portfolio Overview** — current value and weight of each holding
- **Profit & Loss** — how much you've made or lost per stock in $ and %
- **Cumulative & Daily Returns** — interactive charts showing performance over time
- **Performance Summary** — volatility, best/worst days, total return
- **Concentration Risk Alerts** — automatic warnings if portfolio is too heavily weighted
- **Risk Metrics** — Sharpe Ratio, Max Drawdown, Beta vs S&P 500
- **Correlation Heatmap** — shows how stocks move relative to each other
- **PDF Export** — download a professional portfolio report

## 🛠️ Tech Stack
- Python
- Streamlit
- yfinance
- Pandas & NumPy
- Plotly
- ReportLab

## 🚀 Run Locally
```bash
git clone https://github.com/Askirama/portfolio-risk-dashboard.git
cd portfolio-risk-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Requirements
See `requirements.txt`