import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
st.title("📊 Stock Portfolio Risk & Performance Dashboard")

# --- SIDEBAR INPUT ---
st.sidebar.header("Build Your Portfolio")

st.sidebar.markdown("Enter each stock on a new line as: **TICKER, SHARES**")
portfolio_input = st.sidebar.text_area(
    "Your Portfolio",
    value="AAPL, 50\nMSFT, 30\nTSLA, 20",
    height=150
)

st.sidebar.markdown("Enter your buy price for each stock: **TICKER, BUY PRICE**")
buy_price_input = st.sidebar.text_area(
    "Buy Prices",
    value="AAPL, 150\nMSFT, 280\nTSLA, 200",
    height=150
)

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

if st.sidebar.button("Analyse Portfolio", key="analyse_btn"):

    # --- PARSE PORTFOLIO INPUT ---
    portfolio = {}
    errors = []
    for line in portfolio_input.strip().split("\n"):
        try:
            parts = line.strip().split(",")
            ticker = parts[0].strip().upper()
            shares = float(parts[1].strip().replace(',', ''))
            portfolio[ticker] = shares
        except:
            errors.append(line)

    # --- PARSE BUY PRICES ---
    buy_prices = {}
    for line in buy_price_input.strip().split("\n"):
        try:
            parts = line.strip().split(",")
            ticker = parts[0].strip().upper()
            price = float(parts[1].strip().replace(',', ''))
            buy_prices[ticker] = price
        except:
            pass

    if errors:
        st.error(f"Couldn't parse these lines: {errors} — make sure format is TICKER, SHARES")
    elif not portfolio:
        st.error("No valid stocks entered.")
    else:
        tickers = list(portfolio.keys())
        shares_series = pd.Series(portfolio)
        buy_price_series = pd.Series(buy_prices)

        st.subheader(f"Analysing: {', '.join(tickers)}")

        # --- FETCH DATA ---
        raw_data = yf.download(tickers, start=start_date, end=end_date)["Close"]

        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)

        raw_data = raw_data.dropna(axis=1, how='all')

        if raw_data.empty or len(raw_data) < 2:
            st.error("Data loaded but was empty — likely a connection issue. Try again or switch to hotspot.")
        else:
            st.success("Data loaded successfully!")

            # --- CURRENT PRICES & PORTFOLIO VALUE ---
            current_prices = raw_data.iloc[-1]
            holding_values = shares_series * current_prices
            total_value = holding_values.sum()
            weights = (holding_values / total_value * 100).round(2)

            # ── SECTION 1: PORTFOLIO OVERVIEW ──────────────────────────
            st.subheader("💼 Portfolio Overview")
            overview = pd.DataFrame({
                "Shares Owned": shares_series,
                "Current Price ($)": current_prices.round(2),
                "Holding Value ($)": holding_values.round(2),
                "Portfolio Weight (%)": weights
            })
            st.dataframe(overview)
            st.metric(label="💰 Total Portfolio Value", value=f"${total_value:,.2f}")

            # ── SECTION 2: PROFIT & LOSS ────────────────────────────────
            st.subheader("📊 Profit & Loss Summary")

            cost_basis = buy_price_series * shares_series
            current_value = current_prices * shares_series
            pnl_dollars = (current_value - cost_basis).round(2)
            pnl_percent = ((pnl_dollars / cost_basis) * 100).round(2)

            pnl_df = pd.DataFrame({
                "Buy Price ($)": buy_price_series.round(2),
                "Current Price ($)": current_prices.round(2),
                "Cost Basis ($)": cost_basis.round(2),
                "Current Value ($)": current_value.round(2),
                "P&L ($)": pnl_dollars,
                "P&L (%)": pnl_percent
            })

            def color_pnl(val):
                if isinstance(val, (int, float)):
                    return "color: green" if val > 0 else "color: red" if val < 0 else ""
                return ""

            st.dataframe(pnl_df.style.map(color_pnl, subset=["P&L ($)", "P&L (%)"]))

            total_pnl = pnl_dollars.sum()
            total_pnl_pct = ((total_pnl / cost_basis.sum()) * 100).round(2)

            col1, col2 = st.columns(2)
            col1.metric(label="💰 Total P&L ($)", value=f"${total_pnl:,.2f}", delta=f"{total_pnl_pct}%")
            col2.metric(label="📈 Total Return (%)", value=f"{total_pnl_pct}%", delta=f"${total_pnl:,.2f}")

            # ── SECTION 3: PERFORMANCE ──────────────────────────────────
            daily_returns = raw_data.pct_change().dropna()
            cumulative_returns = (1 + daily_returns).cumprod()

            st.subheader("📈 Cumulative Returns Over Time")
            fig1 = px.line(cumulative_returns, title="Cumulative Returns — How $1 invested grew over time")
            fig1.update_layout(xaxis_title="Date", yaxis_title="Growth of $1")
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("📉 Daily Returns")
            fig2 = px.line(daily_returns, title="Daily Returns per Stock")
            fig2.update_layout(xaxis_title="Date", yaxis_title="Daily Return (%)")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("📋 Performance Summary")
            summary = pd.DataFrame({
                "Total Return (%)": ((cumulative_returns.iloc[-1] - 1) * 100).round(2),
                "Avg Daily Return (%)": (daily_returns.mean() * 100).round(4),
                "Volatility (%)": (daily_returns.std() * 100).round(4),
                "Best Day (%)": (daily_returns.max() * 100).round(2),
                "Worst Day (%)": (daily_returns.min() * 100).round(2),
            })
            st.dataframe(summary)

            # ── SECTION 4: CONCENTRATION RISK ──────────────────────────
            st.subheader("🎯 Concentration Risk Check")
            weight_df = pd.DataFrame({
                'Stock': weights.index,
                'Weight (%)': weights.values
            }).sort_values('Weight (%)', ascending=False)
            st.dataframe(weight_df, use_container_width=True)

            alerts = []
            for stock, weight in weights.items():
                if weight > 50:
                    alerts.append(f"🔴 **CRITICAL:** {stock} makes up {weight:.1f}% — extreme single-stock risk")
                elif weight > 30:
                    alerts.append(f"🟡 **WARNING:** {stock} makes up {weight:.1f}% — high concentration")

            top_3_weight = weights.nlargest(3).sum()
            if top_3_weight > 80:
                alerts.append(f"⚠️ Top 3 stocks represent {top_3_weight:.1f}% of portfolio — low diversification")
            if len(weights) == 1:
                alerts.append("🔴 **CRITICAL:** Single-stock portfolio — no diversification")
            elif len(weights) == 2:
                alerts.append("🟡 **WARNING:** Only 2 stocks — consider adding more")

            if alerts:
                st.error("\n\n".join(alerts))
                st.caption("💡 A well-diversified portfolio typically has no single stock > 20%.")
            else:
                st.success("✅ No concentration issues — your portfolio appears well-balanced")

            # ── SECTION 5: RISK METRICS ─────────────────────────────────
            st.subheader("⚠️ Risk Metrics")

            risk_free_rate = 0.045 / 252
            sharpe_ratio = ((daily_returns.mean() - risk_free_rate) / daily_returns.std() * np.sqrt(252)).round(2)

            def max_drawdown(returns):
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.cummax()
                drawdown = (cumulative - rolling_max) / rolling_max
                return drawdown.min()

            max_dd = (daily_returns.apply(max_drawdown) * 100).round(2)

            sp500 = yf.download("^GSPC", start=start_date, end=end_date)["Close"]
            sp500_returns = sp500.pct_change().dropna()
            if isinstance(sp500_returns, pd.DataFrame):
                sp500_returns = sp500_returns.iloc[:, 0]

            betas = {}
            for ticker in daily_returns.columns:
                aligned = pd.concat([daily_returns[ticker], sp500_returns], axis=1).dropna()
                aligned.columns = ["stock", "market"]
                covariance = aligned.cov().iloc[0, 1]
                market_variance = aligned["market"].var()
                betas[ticker] = round(covariance / market_variance, 2)

            beta_series = pd.Series(betas)
            risk_metrics = pd.DataFrame({
                "Sharpe Ratio": sharpe_ratio,
                "Max Drawdown (%)": max_dd,
                "Beta (vs S&P 500)": beta_series
            })
            st.dataframe(risk_metrics)
            st.markdown("""
            **Understanding the metrics:**
            - **Sharpe Ratio** — Above 1.0 is good, above 2.0 is excellent
            - **Max Drawdown** — Worst % loss from peak to bottom. Closer to 0 is better
            - **Beta** — Above 1.0 means more volatile than the market
            """)

            # ── SECTION 6: DRAWDOWN CHART ───────────────────────────────
            st.subheader("📉 Drawdown Over Time")
            drawdown_df = pd.DataFrame()
            for ticker in daily_returns.columns:
                cumulative = (1 + daily_returns[ticker]).cumprod()
                rolling_max = cumulative.cummax()
                drawdown_df[ticker] = (cumulative - rolling_max) / rolling_max * 100

            fig3 = px.line(drawdown_df, title="Drawdown Over Time (%)")
            fig3.update_layout(xaxis_title="Date", yaxis_title="Drawdown (%)")
            st.plotly_chart(fig3, use_container_width=True)

            # ── SECTION 7: CORRELATION HEATMAP ─────────────────────────
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
            fig4.update_layout(title="Stock Correlation Matrix")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("""
            **Reading the heatmap:**
            - 🟢 **Green (close to 1.0)** — stocks move together, low diversification
            - 🟡 **Yellow (close to 0.0)** — little relationship
            - 🔴 **Red (close to -1.0)** — opposite movement, great diversification
            """)
            # ── SECTION 8: PDF EXPORT ───────────────────────────────────
            st.subheader("📄 Export Portfolio Report")

            def generate_pdf(overview, pnl_df, risk_metrics, total_value, total_pnl, total_pnl_pct):
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4,
                                        leftMargin=2*cm, rightMargin=2*cm,
                                        topMargin=2*cm, bottomMargin=2*cm)
                styles = getSampleStyleSheet()
                story = []

                story.append(Paragraph("Portfolio Risk & Performance Report", styles['Title']))
                story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", styles['Normal']))
                story.append(Spacer(1, 0.5*cm))

                story.append(Paragraph("Portfolio Summary", styles['Heading2']))
                story.append(Paragraph(f"Total Portfolio Value: ${total_value:,.2f}", styles['Normal']))
                story.append(Paragraph(f"Total P&L: ${total_pnl:,.2f} ({total_pnl_pct}%)", styles['Normal']))
                story.append(Spacer(1, 0.3*cm))

                story.append(Paragraph("Portfolio Overview", styles['Heading2']))
                overview_reset = overview.reset_index()
                overview_data = [overview_reset.columns.tolist()] + overview_reset.values.tolist()
                overview_table = Table(overview_data, repeatRows=1)
                overview_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A3A5C')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,-1), 8),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F5F5F5')]),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                story.append(overview_table)
                story.append(Spacer(1, 0.3*cm))

                story.append(Paragraph("Profit & Loss", styles['Heading2']))
                pnl_reset = pnl_df.reset_index()
                pnl_data = [pnl_reset.columns.tolist()] + pnl_reset.values.tolist()
                pnl_table = Table(pnl_data, repeatRows=1)
                pnl_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A3A5C')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,-1), 8),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F5F5F5')]),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                story.append(pnl_table)
                story.append(Spacer(1, 0.3*cm))

                story.append(Paragraph("Risk Metrics", styles['Heading2']))
                risk_reset = risk_metrics.reset_index()
                risk_data = [risk_reset.columns.tolist()] + risk_reset.values.tolist()
                risk_table = Table(risk_data, repeatRows=1)
                risk_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A3A5C')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,-1), 8),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F5F5F5')]),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                story.append(risk_table)

                doc.build(story)
                buffer.seek(0)
                return buffer

            pdf_buffer = generate_pdf(overview, pnl_df, risk_metrics, total_value, total_pnl, total_pnl_pct)

            st.download_button(
                label="📥 Download Portfolio Report (PDF)",
                data=pdf_buffer,
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )