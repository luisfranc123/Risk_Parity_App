import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

# Portfolio definitions 

default_portfolios = {
    "Aspect Partners Risk Parity": {
        "tickers": ["TLT","IEFA","PDBC","IAGG","IAU","DSMC","RSSB",
                    "ASFYX","APDFX","QMHIX","SPY","GQGIX"],
        "weights": [0.05, 0.11, 0.10, 0.05, 0.10, 0.06, 0.10,
                    0.10, 0.05, 0.10, 0.12, 0.06],
        "color": "#1B8FFB",   
    },
    "SPY + Long Vol": {
        "tickers": ["SPY", "CAOS"],
        "weights": [0.60, 0.40],
        "color": "#80807F",   
    },
    "SPY + Long Bond": {
        "tickers": ["SPY", "AGG"],
        "weights": [0.60, 0.40],
        "color": "#5B9BD5",   
    },
}

benchmark = {
    "name": "State Street Global Allocation ETF",
    "ticker": "GAL",
    "color": "#FFFFFF",       
}

ticker_names = {
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "IEFA": "iShares Core MSCI EAFE ETF",
    "PDBC": "Invesco Optm Yd Dvrs Cdty Stra No K1 ETF",
    "IAGG": "iShares Core International Aggt Bd ETF",
    "IAU": "iShares Gold Trust",
    "DSMC": "Distillate Small/Mid Cash Flow ETF",
    "RSSB": "Return Stacked Global Stocks & Bonds ETF",
    "ASFYX": "Virtus AlphaSimplex Mgd Futs Strat I",
    "APDFX": "Artisan High Income Advisor",
    "QMHIX": "AQR Managed Futures Strategy HV I",
    "SPY": "State Street SPDR S&P 500 ETF",
    "GQGIX": "GQG Partners Emerging Markets EquityInst",
    "CAOS": "Alpha Architect Tail Risk ETF",
    "AGG": "iShares Core US Aggregate Bond ETF",
    "GAL": "State Street Global Allocation ETF",
}


@st.cache_data(show_spinner=False)
def fetch_prices(tickers: list, start: str, end: str,
                 reinvest_dividends: bool = True) -> pd.DataFrame:
    """
    Download prices for a list of tickers.
    reinvest_dividends = True  → auto_adjust = True (adjusted / total-return prices)
    reinvest_dividends = False → auto_adjust = False (unadjusted / price-return only)
    """
    import time
    import requests
    all_tickers = list(set(tickers))
   
    # Spoof a browser session so Yahoo Finance doesn't block cloud IPs
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })

    prices = pd.DataFrame()

    for t in all_tickers:
        for attempt in range(4):
            try:
                ticker_obj = yf.Ticker(t, session=session)
                hist = ticker_obj.history(
                    start=start, end=end,
                    auto_adjust=reinvest_dividends,
                    timeout=20,
                )
                if not hist.empty and "Close" in hist.columns:
                    prices[t] = hist["Close"]
                    break
            except Exception:
                time.sleep(1.5 * (attempt + 1))  # exponential back-off

    if prices.empty:
        return prices

    prices = prices.dropna(how="all").ffill(limit=5)
    return prices
   

def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def get_monthly_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    return daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)


def build_portfolio_returns(
    prices: pd.DataFrame,
    tickers: list,
    weights: list,
    rebalance_freq: str = "quarterly",
    use_risk_parity: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward backtest returning a DataFrame with columns:
      date, portfolio_return, portfolio_value, weights
    """
    from utils.optimize_portfolio import optimize_risk_parity

    # Only keep tickers that downloaded successfully
    available = [t for t in tickers if t in prices.columns]
    if not available:
        return pd.DataFrame()

    port_prices = prices[available].dropna(how="all").ffill(limit=5)
    daily_ret = port_prices.pct_change().dropna()
    monthly_ret = daily_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    # Base weights (normalised to available tickers)
    w_map = dict(zip(tickers, weights))
    base_w = np.array([w_map.get(t, 0.0) for t in available])
    base_w = base_w / base_w.sum()

    results = []
    portfolio_value = 10_000.0
    current_weights = None

    for i, (date, row) in enumerate(monthly_ret.iterrows()):
        # Decide rebalance
        is_rebalance = False
        if rebalance_freq == "quarterly":
            is_rebalance = date.month in [3, 6, 9, 12]
        else:
            is_rebalance = True

        if current_weights is None or is_rebalance:
            if use_risk_parity and i >= 3:
                train = monthly_ret.iloc[:i]
                cov = train.cov().values * 12
                try:
                    current_weights = optimize_risk_parity(cov)
                except Exception:
                    current_weights = base_w.copy()
            else:
                current_weights = base_w.copy()

        # Align weights/returns (some assets may have NaN on this month)
        ret_vals = row.fillna(0).values
        port_ret = float(np.dot(current_weights, ret_vals))
        portfolio_value *= 1 + port_ret

        results.append({
            "date": date,
            "portfolio_return": port_ret,
            "portfolio_value": portfolio_value,
            "weights": dict(zip(available, current_weights)),
        })

    return pd.DataFrame(results)

