import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# LAYOUT
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 95%;
    }
    .linkedin-badge {
        background-color: #0A66C2;
        color: white !important;
        padding: 8px 12px;
        border-radius: 4px;
        font-weight: bold;
        text-align: center;
        margin-top: 2rem;
    }
    .linkedin-badge a {
        color: white !important;
        text-decoration: none;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 15px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# TICKER FROM CSV
tickers_df = pd.read_csv(r"nasdaq_tickers.csv")
ticker_list = sorted(tickers_df['Symbol'].astype(str).tolist())
selected_ticker = st.sidebar.selectbox("Select a ticker", ticker_list)

# ADD LINKEDIN BADGE AND DISCLAIMER TO SIDEBAR
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="linkedin-badge"><a href="https://www.linkedin.com/in/matteo-ientile/" target="_blank">'
    '🔗 Connect with me on LinkedIn'
    '</a></div>',
    unsafe_allow_html=True
)
st.sidebar.markdown("""
    <div style="text-align: center; margin-top: 0.5rem; font-size: 0.9rem;">
    Created by Matteo Ientile<br>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.warning("""
**⚠️ IMPORTANT DISCLAIMER**  
This simulation is for **educational purposes only**. It's not financial advice and shouldn't be used for investment decisions. Results are hypothetical and based on simplified models.
""")

# TITLE WITH DISCLAIMER
st.title("Stock Path Simulator - Geometric Brownian Motion")

# INPUT DATA
col1, col2, col3 = st.columns(3)
n_steps = col1.number_input("Days of simulation", min_value=1, value=365)
n_sim = col2.number_input("Number of paths", min_value=1, value=5000)
horizon = col3.radio("Historical window for parameter estimation", [
    "Short-term (6 months)",
    "Mid-term (2 years)",
    "Long-term (5 years)",
    "Since Trump election (11/8/2024)"
])
col3.markdown("Estimation of drift and volatility depends on the selected time horizon.")

ticker = selected_ticker.upper()

#GBM FUNCTION
if ticker:
    dt = 1 / 365
    end_date_dt = datetime.today() - timedelta(days=1)

    # START DATE BASED ON HORIZON TIME
    if horizon == "Short-term (6 months)":
        start_date_dt = end_date_dt - timedelta(days=182)
    elif horizon == "Mid-term (2 years)":
        start_date_dt = end_date_dt - timedelta(days=365 * 2)
    elif horizon == "Long-term (5 years)":
        start_date_dt = end_date_dt - timedelta(days=365 * 5)
    else:  # (Nov 8, 2024)
        start_date_dt = datetime(2024, 11, 8)

    end_date = end_date_dt.strftime('%Y-%m-%d')
    start_date = start_date_dt.strftime('%Y-%m-%d')

    data = yf.download(ticker, start=start_date, end=end_date)
    df = pd.DataFrame(data)
    df_close = df["Close"]
    log_returns = np.log(df_close / df_close.shift(1))
    df["log_returns"] = log_returns

    S0 = df["Close"].iloc[-1]
    daily_sigma = float(log_returns.std())
    sigma = float(daily_sigma * np.sqrt(252))
    mu = float(np.mean(log_returns) * 252 + 0.5 * sigma**2)

    X0 = np.log(S0)
    Xmatrix = np.zeros((n_steps + 1, n_sim))
    Xmatrix[0, :] = X0

    np.random.seed(42)
    for i in range(n_steps):
        Z = np.random.normal(0, 1, size=n_sim)
        Xi = Xmatrix[i, :] + (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        Xmatrix[i + 1, :] = Xi

    Smatrix = np.exp(Xmatrix)
    median_path = np.median(Smatrix, axis=1)
    p5 = np.percentile(Smatrix, 5, axis=1)
    p25 = np.percentile(Smatrix, 25, axis=1)
    p75 = np.percentile(Smatrix, 75, axis=1)
    p95 = np.percentile(Smatrix, 95, axis=1)

    # VAR 95% FOR THE WHOLE LENGTH
    VaR_95 = max(0, float(S0 - p5[n_steps]))

    # PARAMETERS TAB
    st.subheader("📊 Estimated Parameters")
    param_df = pd.DataFrame({
        "Parameter": [
            "Initial Price (S₀)", 
            "Annualized Drift (μ)", 
            "Annualized Volatility (σ)", 
            f"VaR 95% ({n_steps} days)"
        ],
        "Value": [
            round(float(S0), 2), 
            round(mu, 4), 
            round(sigma, 4), 
            round(VaR_95, 2)
        ]
    })
    st.dataframe(param_df, hide_index=True, use_container_width=False)

    fig = go.Figure()

    # 100 (OW WHATEVER, JUST CHANGE 100) PATH PLOTTED
    colors = px.colors.qualitative.Plotly
    for i in range(min(n_sim, 100)):
        fig.add_trace(go.Scatter(
            x=np.arange(n_steps + 1),
            y=Smatrix[:, i],
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=1),
            name=f"Sim {i + 1}",
            opacity=0.3,
            showlegend=False
        ))

    # PERCENTILE LINES
    fig.add_trace(go.Scatter(x=np.arange(n_steps + 1), y=p5, mode='lines',
                             line=dict(color='darkgreen', width=2),
                             name="5th Percentile"))
    fig.add_trace(go.Scatter(x=np.arange(n_steps + 1), y=p95, mode='lines',
                             line=dict(color='darkgreen', width=2),
                             name="95th Percentile"))
    fig.add_trace(go.Scatter(x=np.arange(n_steps + 1), y=p25, mode='lines',
                             line=dict(color='dodgerblue', width=2),
                             name="25th Percentile"))
    fig.add_trace(go.Scatter(x=np.arange(n_steps + 1), y=p75, mode='lines',
                             line=dict(color='dodgerblue', width=2),
                             name="75th Percentile"))
    fig.add_trace(go.Scatter(x=np.arange(n_steps + 1), y=median_path, mode='lines',
                             line=dict(color='crimson', width=4),
                             name="Median Path"))

    fig.update_layout(
        title=f"{ticker} price forecast, Day 0: {end_date} - GBM ({n_sim} simulations)",
        xaxis_title="T (days)",
        yaxis_title="Stock price",
        height=700,
        width=1500,
        legend=dict(y=0.5, traceorder="normal", font=dict(size=12))
    )

    st.plotly_chart(fig, use_container_width=False)

# ADD FINAL DISCLAIMER AT BOTTOM
st.markdown("---")
st.caption("""
**Disclaimer & Purpose Statement**: This simulation demonstrates Geometric Brownian Motion concepts in finance. 
It uses simplified models that don't capture all market realities. The creator makes no claims about accuracy 
and accepts no liability for any use of these simulations. This is purely educational/entertainment software - 
**not investment advice**, **not financial guidance**, and **not a trading tool**. Always consult qualified professionals 
before making financial decisions.
""")



