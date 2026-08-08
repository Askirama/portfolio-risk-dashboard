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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")

# --- GLOBAL STYLE: narrower sidebar, KPI card polish ---
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    width: 300px !important;
    min-width: 300px !important;
}
div[data-testid="stMetric"] {
    background-color: rgba(128, 128, 128, 0.08);
    border-radius: 10px;
    padding: 12px 16px;
    border: 1px solid rgba(128, 128, 128, 0.15);
}
</style>
""", unsafe_allow_html=True)

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


# ── FORMATTING HELPERS ───────────────────────────────────────────
def fmt_money(x):
    """$ with thousands separator and 2 decimals."""
    try:
        return f"${x:,.2f}"
    except (ValueError, TypeError):
        return "—"


def fmt_pct(x):
    """Percent with 2 decimals."""
    try:
        return f"{x:,.2f}%"
    except (ValueError, TypeError):
        return "—"


def colored_text(text, color):
    return f'<span style="color:{color}; font-weight:600;">{text}</span>'


# ── RISK METRIC HELPERS (reused for per-stock and portfolio-level) ──
RISK_FREE_RATE = 0.045 / 252


def compute_sharpe(returns):
    std = returns.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return round(((returns.mean() - RISK_FREE_RATE) / std) * np.sqrt(252), 2)


def compute_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return round(drawdown.min() * 100, 2)


def compute_beta(returns, market_returns):
    aligned = pd.concat([returns, market_returns], axis=1).dropna()
    aligned.columns = ["asset", "market"]
    if len(aligned) < 2:
        return np.nan
    covariance = aligned.cov().iloc[0, 1]
    market_variance = aligned["market"].var()
    if market_variance == 0:
        return np.nan
    return round(covariance / market_variance, 2)


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
        except Exception:
            errors.append(line)

    # --- PARSE BUY PRICES ---
    buy_prices = {}
    for line in buy_price_input.strip().split("\n"):
        try:
            parts = line.strip().split(",")
            ticker = parts[0].strip().upper()
            price = float(parts[1].strip().replace(',', ''))
            buy_prices[ticker] = price
        except Exception:
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

        # --- FETCH DATA (with loading indicator) ---
        with st.spinner("Analysing portfolio…"):
            raw_data = yf.download(tickers, start=start_date, end=end_date)["Close"]

            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = raw_data.columns.get_level_values(0)

            raw_data = raw_data.dropna(axis=1, how='all')

            sp500 = yf.download("^GSPC", start=start_date, end=end_date)["Close"]
            if isinstance(sp500, pd.DataFrame):
                sp500 = sp500.iloc[:, 0]

        if raw_data.empty or len(raw_data) < 2:
            st.error("Data loaded but was empty — likely a connection issue. Try again or switch to hotspot.")
        else:
            st.success("✅ Portfolio analysed successfully")

            # --- CURRENT PRICES & PORTFOLIO VALUE ---
            current_prices = raw_data.iloc[-1]
            holding_values = shares_series * current_prices
            total_value = holding_values.sum()
            weights = (holding_values / total_value * 100).round(2)

            # --- RETURNS ---
            daily_returns = raw_data.pct_change().dropna()
            cumulative_returns = (1 + daily_returns).cumprod()

            sp500_returns = sp500.pct_change().dropna()

            # --- PORTFOLIO-LEVEL RETURN SERIES (value-weighted) ---
            portfolio_value_series = (raw_data * shares_series).sum(axis=1)
            portfolio_returns = portfolio_value_series.pct_change().dropna()

            # --- P&L ---
            cost_basis = buy_price_series * shares_series
            current_value = current_prices * shares_series
            pnl_dollars = (current_value - cost_basis).round(2)
            pnl_percent = ((pnl_dollars / cost_basis) * 100).round(2)

            total_pnl = pnl_dollars.sum()
            total_pnl_pct = round((total_pnl / cost_basis.sum()) * 100, 2)

            # --- PORTFOLIO-LEVEL RISK METRICS ---
            portfolio_sharpe = compute_sharpe(portfolio_returns)
            portfolio_max_dd = compute_max_drawdown(portfolio_returns)
            portfolio_beta = compute_beta(portfolio_returns, sp500_returns)

            # --- SP500 BENCHMARK RETURN OVER PERIOD ---
            if len(sp500) >= 2:
                sp500_total_return = round(((sp500.iloc[-1] / sp500.iloc[0]) - 1) * 100, 2)
            else:
                sp500_total_return = np.nan

            # ── KPI CARDS ────────────────────────────────────────────
            st.subheader("📌 Key Metrics")

            # Portfolio Value gets its own full-width headline row so the
            # figure never gets clipped inside a narrow column.
            st.markdown(
                f'''
                <div style="text-align:center; background-color: rgba(128,128,128,0.08);
                            border: 1px solid rgba(128,128,128,0.15); border-radius: 10px;
                            padding: 18px 16px; margin-bottom: 14px;">
                    <div style="font-size: 1rem; opacity: 0.75;">💰 Portfolio Value</div>
                    <div style="font-size: 2.6rem; font-weight: 700; line-height: 1.2;">{fmt_money(total_value)}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

            kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(5)
            kpi2.metric("📈 Total Return", fmt_pct(total_pnl_pct), delta=fmt_money(total_pnl))
            kpi3.metric("📊 Sharpe Ratio", f"{portfolio_sharpe:.2f}" if pd.notna(portfolio_sharpe) else "—")
            kpi4.metric("📉 Max Drawdown", fmt_pct(portfolio_max_dd) if pd.notna(portfolio_max_dd) else "—")
            kpi5.metric("⚠️ Portfolio Beta", f"{portfolio_beta:.2f}" if pd.notna(portfolio_beta) else "—")
            kpi6.metric("📦 Holdings", f"{len(tickers)}")

            # ── EXECUTIVE SUMMARY ───────────────────────────────────
            st.subheader("📝 Executive Summary")
            top_weight_stock = weights.idxmax()
            top_weight_value = weights.max()

            vs_benchmark = ""
            if pd.notna(sp500_total_return):
                diff = round(total_pnl_pct - sp500_total_return, 2)
                if diff > 0:
                    vs_benchmark = f"outperforming the S&P 500 by {colored_text(fmt_pct(diff), '#2ecc71')}"
                elif diff < 0:
                    vs_benchmark = f"underperforming the S&P 500 by {colored_text(fmt_pct(abs(diff)), '#e74c3c')}"
                else:
                    vs_benchmark = "in line with the S&P 500"

            concentration_note = ""
            if top_weight_value > 30:
                concentration_note = (
                    f" {top_weight_stock} represents {colored_text(fmt_pct(top_weight_value), '#f39c12')} "
                    f"of the portfolio, suggesting concentration risk. Consider increasing diversification."
                )

            sharpe_note = "moderate"
            if pd.notna(portfolio_sharpe):
                if portfolio_sharpe > 1.5:
                    sharpe_note = "strong"
                elif portfolio_sharpe < 0.5:
                    sharpe_note = "weak"

            return_color = "#2ecc71" if total_pnl_pct >= 0 else "#e74c3c"
            if pd.notna(portfolio_sharpe):
                summary_html = (
                    f"Your portfolio has returned {colored_text(fmt_pct(total_pnl_pct), return_color)} "
                    f"over the selected period"
                    + (f", {vs_benchmark}" if vs_benchmark else "")
                    + f". Risk is {sharpe_note}, with a Sharpe Ratio of {portfolio_sharpe:.2f}."
                )
            else:
                summary_html = (
                    f"Your portfolio has returned {colored_text(fmt_pct(total_pnl_pct), return_color)} "
                    f"over the selected period"
                    + (f", {vs_benchmark}" if vs_benchmark else "")
                    + "."
                )
            summary_html += concentration_note

            st.markdown(
                f'<div style="background-color: rgba(52,152,219,0.08); border-left: 4px solid #3498db; '
                f'padding: 14px 18px; border-radius: 6px; font-size: 1.02rem;">{summary_html}</div>',
                unsafe_allow_html=True
            )
            st.markdown("")

            # ── PORTFOLIO INSIGHTS (AI-style) ───────────────────────
            st.subheader("💡 Portfolio Insights")

            performance_summary = pd.DataFrame({
                "Total Return (%)": ((cumulative_returns.iloc[-1] - 1) * 100).round(2),
                "Avg Daily Return (%)": (daily_returns.mean() * 100).round(2),
                "Volatility (%)": (daily_returns.std() * 100).round(2),
                "Best Day (%)": (daily_returns.max() * 100).round(2),
                "Worst Day (%)": (daily_returns.min() * 100).round(2),
            })

            insights = []

            best_performer = performance_summary["Total Return (%)"].idxmax()
            best_return = performance_summary["Total Return (%)"].max()
            insights.append(("success", f"📈 **{best_performer}** generated the highest return in the portfolio ({fmt_pct(best_return)})."))

            if top_weight_value > 50:
                insights.append(("error", f"🔴 **{top_weight_stock}** accounts for {fmt_pct(top_weight_value)} of the portfolio — extreme concentration risk."))
            elif top_weight_value > 30:
                insights.append(("warning", f"⚠️ **{top_weight_stock}** accounts for {fmt_pct(top_weight_value)} of the portfolio, indicating concentration risk."))

            most_volatile = performance_summary["Volatility (%)"].idxmax()
            insights.append(("warning", f"📉 **{most_volatile}** contributes the highest volatility in the portfolio."))

            top_n = min(3, len(weights))
            top_3_weight = weights.nlargest(top_n).sum()
            if len(weights) == 1:
                diversification = "poor"
                insights.append(("error", "🔴 Single-stock portfolio — no diversification."))
            elif top_3_weight > 80:
                diversification = "low"
                insights.append(("warning", f"⚠️ Top {top_n} holdings represent {fmt_pct(top_3_weight)} of the portfolio — low diversification."))
            elif top_3_weight > 60:
                diversification = "moderate"
                insights.append(("info", f"✅ Portfolio diversification is moderate (top holdings: {fmt_pct(top_3_weight)})."))
            else:
                diversification = "good"
                insights.append(("success", f"✅ Portfolio diversification looks good (top holdings: {fmt_pct(top_3_weight)})."))

            if top_weight_value > 30 or diversification in ("low", "poor"):
                insights.append(("info", "💡 Consider reducing exposure to heavily weighted assets to improve diversification."))

            render_map = {"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}
            for kind, text in insights:
                render_map[kind](text)

            # ── SECTION: PORTFOLIO OVERVIEW ─────────────────────────
            st.subheader("💼 Portfolio Overview")
            overview = pd.DataFrame({
                "Shares Owned": shares_series,
                "Current Price ($)": current_prices.round(2),
                "Holding Value ($)": holding_values.round(2),
                "Portfolio Weight (%)": weights
            })
            overview_display = overview.copy()
            overview_display["Current Price ($)"] = overview_display["Current Price ($)"].map(fmt_money)
            overview_display["Holding Value ($)"] = overview_display["Holding Value ($)"].map(fmt_money)
            overview_display["Portfolio Weight (%)"] = overview_display["Portfolio Weight (%)"].map(fmt_pct)
            st.dataframe(overview_display, use_container_width=True)

            # ── SECTION: PROFIT & LOSS ──────────────────────────────
            st.subheader("📊 Profit & Loss Summary")

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
                    return "color: #2ecc71" if val > 0 else "color: #e74c3c" if val < 0 else ""
                return ""

            pnl_display = pnl_df.copy()
            for col in ["Buy Price ($)", "Current Price ($)", "Cost Basis ($)", "Current Value ($)", "P&L ($)"]:
                pnl_display[col] = pnl_display[col].map(fmt_money)
            pnl_display["P&L (%)"] = pnl_df["P&L (%)"].map(fmt_pct)

            styled_pnl = pnl_display.style.apply(
                lambda col: pnl_df["P&L ($)"].map(color_pnl) if col.name in ["P&L ($)", "P&L (%)"] else [""] * len(col),
                axis=0
            )
            st.dataframe(styled_pnl, use_container_width=True)

            col1, col2 = st.columns(2)
            col1.metric(label="💰 Total P&L ($)", value=fmt_money(total_pnl), delta=fmt_pct(total_pnl_pct))
            col2.metric(label="📈 Total Return (%)", value=fmt_pct(total_pnl_pct), delta=fmt_money(total_pnl))

            # ── SECTION: PERFORMANCE ────────────────────────────────
            st.subheader("📈 Cumulative Returns Over Time")
            fig1 = px.line(cumulative_returns, title="Cumulative Returns — How $1 invested grew over time")
            fig1.update_layout(xaxis_title="Date", yaxis_title="Growth of $1")
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("📉 Daily Returns")
            fig2 = px.line(daily_returns, title="Daily Returns per Stock")
            fig2.update_layout(xaxis_title="Date", yaxis_title="Daily Return (%)")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("📋 Performance Summary")
            perf_display = performance_summary.copy()
            for col in perf_display.columns:
                perf_display[col] = perf_display[col].map(fmt_pct)
            st.dataframe(perf_display, use_container_width=True)

            # ── SECTION: CONCENTRATION RISK ─────────────────────────
            st.subheader("🎯 Concentration Risk Check")
            weight_df = pd.DataFrame({
                'Stock': weights.index,
                'Weight (%)': weights.values
            }).sort_values('Weight (%)', ascending=False)

            def highlight_weight(val):
                if val > 50:
                    return "background-color: rgba(231,76,60,0.25); color: #e74c3c; font-weight:600;"
                elif val > 30:
                    return "background-color: rgba(243,156,18,0.2); color: #f39c12; font-weight:600;"
                return ""

            weight_display = weight_df.copy()
            weight_display["Weight (%)"] = weight_display["Weight (%)"].map(fmt_pct)
            styled_weights = weight_display.style.apply(
                lambda col: weight_df["Weight (%)"].map(highlight_weight) if col.name == "Weight (%)" else [""] * len(col),
                axis=0
            )
            st.dataframe(styled_weights, use_container_width=True)

            alerts = []
            for stock, weight in weights.items():
                if weight > 50:
                    alerts.append(f"🔴 **CRITICAL:** {stock} makes up {fmt_pct(weight)} — extreme single-stock risk")
                elif weight > 30:
                    alerts.append(f"🟡 **WARNING:** {stock} makes up {fmt_pct(weight)} — high concentration")

            if top_3_weight > 80:
                alerts.append(f"⚠️ Top holdings represent {fmt_pct(top_3_weight)} of portfolio — low diversification")
            if len(weights) == 1:
                alerts.append("🔴 **CRITICAL:** Single-stock portfolio — no diversification")
            elif len(weights) == 2:
                alerts.append("🟡 **WARNING:** Only 2 stocks — consider adding more")

            if alerts:
                st.error("\n\n".join(alerts))
                st.caption("💡 A well-diversified portfolio typically has no single stock > 20%.")
            else:
                st.success("✅ No concentration issues — your portfolio appears well-balanced")

            # ── SECTION: RISK METRICS ───────────────────────────────
            st.subheader("⚠️ Risk Metrics")

            sharpe_per_stock = daily_returns.apply(compute_sharpe)
            max_dd_per_stock = daily_returns.apply(compute_max_drawdown)

            betas = {}
            for ticker in daily_returns.columns:
                betas[ticker] = compute_beta(daily_returns[ticker], sp500_returns)
            beta_series = pd.Series(betas)

            risk_metrics = pd.DataFrame({
                "Sharpe Ratio": sharpe_per_stock,
                "Max Drawdown (%)": max_dd_per_stock,
                "Beta (vs S&P 500)": beta_series
            })
            risk_display = risk_metrics.copy()
            risk_display["Max Drawdown (%)"] = risk_display["Max Drawdown (%)"].map(fmt_pct)
            st.dataframe(risk_display, use_container_width=True)
            sharpe_txt = f"{portfolio_sharpe:.2f}" if pd.notna(portfolio_sharpe) else "—"
            beta_txt = f"{portfolio_beta:.2f}" if pd.notna(portfolio_beta) else "—"
            st.caption(f"**Portfolio-level:** Sharpe {sharpe_txt} · Max Drawdown {fmt_pct(portfolio_max_dd)} · Beta {beta_txt}")
            st.markdown("""
            **Understanding the metrics:**
            - **Sharpe Ratio** — Above 1.0 is good, above 2.0 is excellent
            - **Max Drawdown** — Worst % loss from peak to bottom. Closer to 0 is better
            - **Beta** — Above 1.0 means more volatile than the market
            """)

            # ── SECTION: DRAWDOWN CHART ─────────────────────────────
            st.subheader("📉 Drawdown Over Time")
            drawdown_df = pd.DataFrame()
            for ticker in daily_returns.columns:
                cumulative = (1 + daily_returns[ticker]).cumprod()
                rolling_max = cumulative.cummax()
                drawdown_df[ticker] = (cumulative - rolling_max) / rolling_max * 100

            fig3 = px.line(drawdown_df, title="Drawdown Over Time (%)")
            fig3.update_layout(xaxis_title="Date", yaxis_title="Drawdown (%)")
            st.plotly_chart(fig3, use_container_width=True)

            # ── SECTION: CORRELATION HEATMAP ────────────────────────
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

            # ── SECTION: PDF EXPORT ─────────────────────────────────
            st.subheader("📄 Export Portfolio Report")

            # Distinct, print-safe palette + white/light theme for every chart
            # embedded in the PDF. Kept separate from the dashboard's charts
            # (which follow Streamlit's own dark theme) because static export
            # via Kaleido doesn't reliably carry over a dark theme's colorway —
            # left alone, every trace/slice can render as the same color.
            PDF_PALETTE = ["#2E86AB", "#E76F51", "#2A9D8F", "#F4A261", "#8E44AD", "#457B9D", "#E9C46A", "#606C38"]

            def apply_pdf_theme(fig):
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="#222222", size=13),
                    title_font=dict(color="#111111", size=16),
                    legend=dict(bgcolor="white", font=dict(color="#222222")),
                    margin=dict(l=40, r=30, t=60, b=40),
                )
                fig.update_xaxes(color="#222222", gridcolor="#e5e5e5", linecolor="#cccccc")
                fig.update_yaxes(color="#222222", gridcolor="#e5e5e5", linecolor="#cccccc")
                return fig

            def fig_to_image(fig, width_cm=16, height_cm=8):
                """Convert a Plotly figure to a reportlab Image via kaleido."""
                img_bytes = fig.to_image(format="png", width=1100, height=550, scale=2)
                return Image(io.BytesIO(img_bytes), width=width_cm * cm, height=height_cm * cm)

            def build_table(df, index_label="Ticker"):
                reset = df.reset_index()
                reset.rename(columns={"index": index_label}, inplace=True)
                data = [reset.columns.tolist()] + reset.astype(str).values.tolist()
                table = Table(data, repeatRows=1)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A3A5C')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ]))
                return table

            def generate_pdf():
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4,
                                         leftMargin=2 * cm, rightMargin=2 * cm,
                                         topMargin=2 * cm, bottomMargin=2 * cm)
                styles = getSampleStyleSheet()
                story = []

                story.append(Paragraph("Portfolio Risk &amp; Performance Report", styles['Title']))
                story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", styles['Normal']))
                story.append(Spacer(1, 0.5 * cm))

                # Portfolio Summary
                story.append(Paragraph("Portfolio Summary", styles['Heading2']))
                story.append(Paragraph(f"Total Portfolio Value: {fmt_money(total_value)}", styles['Normal']))
                story.append(Paragraph(f"Total P&amp;L: {fmt_money(total_pnl)} ({fmt_pct(total_pnl_pct)})", styles['Normal']))
                story.append(Paragraph(f"Sharpe Ratio: {sharpe_txt}  |  Max Drawdown: {fmt_pct(portfolio_max_dd)}  |  Beta: {beta_txt}", styles['Normal']))
                story.append(Spacer(1, 0.3 * cm))

                # Executive Summary (plain-text version, no HTML color spans)
                story.append(Paragraph("Executive Summary", styles['Heading2']))
                plain_summary = (
                    f"The portfolio returned {fmt_pct(total_pnl_pct)} over the selected period"
                    + (f", vs. an S&amp;P 500 return of {fmt_pct(sp500_total_return)}" if pd.notna(sp500_total_return) else "")
                    + f". Sharpe Ratio stands at {sharpe_txt}, indicating {sharpe_note} risk-adjusted performance. "
                    f"{top_weight_stock} is the largest holding at {fmt_pct(top_weight_value)} of the portfolio."
                )
                story.append(Paragraph(plain_summary, styles['Normal']))
                story.append(Spacer(1, 0.3 * cm))

                # Portfolio Allocation
                story.append(Paragraph("Portfolio Allocation", styles['Heading2']))
                alloc_table_df = overview[["Shares Owned", "Current Price ($)", "Holding Value ($)", "Portfolio Weight (%)"]]
                story.append(build_table(alloc_table_df, index_label="Ticker"))
                story.append(Spacer(1, 0.3 * cm))

                fig_alloc_pdf = px.pie(
                    values=weights.values, names=weights.index,
                    title="Allocation by Weight (%)", hole=0.45,
                    color_discrete_sequence=PDF_PALETTE
                )
                fig_alloc_pdf.update_traces(textinfo="label+percent", textfont_size=13)
                apply_pdf_theme(fig_alloc_pdf)
                story.append(fig_to_image(fig_alloc_pdf, width_cm=12, height_cm=9))
                story.append(Spacer(1, 0.3 * cm))

                # Performance Overview
                story.append(Paragraph("Performance Overview", styles['Heading2']))
                story.append(build_table(performance_summary, index_label="Ticker"))
                story.append(Spacer(1, 0.2 * cm))

                fig1_pdf = px.line(
                    cumulative_returns, title="Cumulative Returns — How $1 invested grew over time",
                    color_discrete_sequence=PDF_PALETTE
                )
                fig1_pdf.update_layout(xaxis_title="Date", yaxis_title="Growth of $1", legend_title_text="")
                fig1_pdf.update_traces(line=dict(width=2.5))
                apply_pdf_theme(fig1_pdf)
                story.append(fig_to_image(fig1_pdf))
                story.append(Spacer(1, 0.3 * cm))

                # Risk Metrics
                story.append(Paragraph("Risk Metrics", styles['Heading2']))
                story.append(build_table(risk_metrics, index_label="Ticker"))
                story.append(Spacer(1, 0.3 * cm))

                # Correlation Heatmap
                story.append(Paragraph("Correlation Heatmap", styles['Heading2']))
                fig4_pdf = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns.tolist(),
                    y=correlation_matrix.columns.tolist(),
                    colorscale="RdYlGn",
                    zmin=-1, zmax=1,
                    text=correlation_matrix.values,
                    texttemplate="%{text}",
                    textfont=dict(color="black", size=13),
                    showscale=True
                ))
                fig4_pdf.update_layout(title="Stock Correlation Matrix")
                apply_pdf_theme(fig4_pdf)
                story.append(fig_to_image(fig4_pdf))
                story.append(Spacer(1, 0.3 * cm))

                # Key Insights
                story.append(Paragraph("Key Insights", styles['Heading2']))
                insight_items = [ListItem(Paragraph(text.replace("**", ""), styles['Normal'])) for _, text in insights]
                story.append(ListFlowable(insight_items, bulletType='bullet'))
                story.append(Spacer(1, 0.3 * cm))

                # Recommendations
                story.append(Paragraph("Portfolio Recommendations", styles['Heading2']))
                recommendations = []
                if top_weight_value > 30:
                    recommendations.append(f"Reduce exposure to {top_weight_stock} to lower concentration risk.")
                if len(weights) < 3:
                    recommendations.append("Add more holdings to improve diversification.")
                if pd.notna(portfolio_sharpe) and portfolio_sharpe < 1.0:
                    recommendations.append("Risk-adjusted return is below 1.0 Sharpe — review position sizing or asset mix.")
                if not recommendations:
                    recommendations.append("Portfolio composition looks reasonable — continue monitoring risk metrics periodically.")
                rec_items = [ListItem(Paragraph(r, styles['Normal'])) for r in recommendations]
                story.append(ListFlowable(rec_items, bulletType='bullet'))

                doc.build(story)
                buffer.seek(0)
                return buffer

            pdf_buffer = generate_pdf()

            st.download_button(
                label="📥 Download Portfolio Report (PDF)",
                data=pdf_buffer,
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
