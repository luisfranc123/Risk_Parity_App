# Risk Parity Portfolio Analyzer

A Streamlit app that replicates and extends the Portfolio Visualizer PDF report,
running a full walk-forward backtest of three portfolios vs. a benchmark.

## Portfolios
| Portfolio | Description |
|---|---|
| **Aspect Partners Risk Parity** | 13-asset equal-risk-contribution portfolio (customisable) |
| **SPY + Long Vol** | 60% SPY / 40% CAOS |
| **SPY + Long Bond** | 60% SPY / 40% AGG |
| **Benchmark** | State Street Global Allocation ETF (GAL) |

## Features
- 📈 Portfolio growth, annual returns, trailing returns
- ⚖️ 28 risk/return metrics (Sharpe, Sortino, VaR, capture ratios, etc.)
- 🗂 Holdings viewer, risk-parity optimiser, correlation matrix
- 📅 Monthly return heatmaps per portfolio
- 🔬 Per-asset stats, normalised price chart, return decomposition
- 🔄 Monthly or quarterly rebalancing
- 💰 Custom initial investment amount
- 📐 Customisable tickers & weights for Aspect Partners portfolio

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
risk_parity_app/
├── app.py                     # Main Streamlit app
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml            # Dark theme config
└── utils/
    ├── __init__.py
    ├── optimize_portfolio.py  # Risk parity optimiser (SLSQP)
    ├── data_utils.py          # yfinance fetching + walk-forward backtest
    ├── performance_metrics.py # 33 performance metrics
    └── charts.py              # All Plotly visualisations
```

## Data Source
All price data is fetched live from **Yahoo Finance** via `yfinance`.
Mutual fund tickers (ASFYX, APDFX, QMHIX, GQGIX) use forward-fill
for days with no price update, consistent with NAV reporting.

## Notes
- The walk-forward backtest re-optimises weights at each rebalance period
  using historical covariance up to that date (no look-ahead bias).
- Dividends are embedded in Yahoo Finance's adjusted prices (`auto_adjust=True`).
- Benchmark for all metrics: State Street Global Allocation ETF (GAL).
