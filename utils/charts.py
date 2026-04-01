import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Arin Risk Advisors brand palette
arin_navy = "#0D2948"
arin_sky_blue = "#1B8FFB"
arin_white = "#FFFFFF"
arin_gray = "#80807F"
arin_black = "#313131"
arin_grid = "#1a3a5c"
arin_legend_bg = "rgba(7,24,40,0.85)"

# Base layout 
base_layout = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font = dict(family="Segoe UI, Arial, sans-serif", size=12,
                         color=arin_white),
    legend = dict(bgcolor=arin_legend_bg, bordercolor=arin_grid,
                         borderwidth=1, font=dict(color=arin_white)),
    margin = dict(l=10, r=10, t=40, b=10),
)

def _axis(extra=None):
    """Return a standard branded axis dict, optionally merged with extras."""
    base = dict(gridcolor=arin_grid, showgrid=True, zeroline=False,
                color=arin_white, tickfont=dict(color=arin_gray))
    if extra:
        base.update(extra)
    return base


# Portfolio Growth
def plot_portfolio_growth(backtest_map: dict, initial_value: float = 10_000) -> go.Figure:
    fig = go.Figure()
    for label, (bt, color) in backtest_map.items():
        if bt is None or bt.empty:
            continue
        dates  = [bt["date"].iloc[0] - pd.DateOffset(months=1)] + bt["date"].tolist()
        values = [initial_value] + bt["portfolio_value"].tolist()
        fig.add_trace(go.Scatter(
            x=dates, y=values, mode="lines", name=label,
            line=dict(color=color, width=2),
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>" + label + "</extra>",
        ))
    fig.update_layout(
        **base_layout,
        title  = dict(text="Portfolio Growth", font=dict(size=14)),
        height = 420,
        xaxis  = _axis(),
        yaxis  = _axis(dict(tickprefix="$")),
    )
    return fig


# Annual Returns Bar Chart
def plot_annual_returns(metrics_map: dict) -> go.Figure:
    fig = go.Figure()
    for label, (m, color) in metrics_map.items():
        if m is None:
            continue
        yr = m.get("_yearly_returns")
        if yr is None or yr.empty:
            continue
        fig.add_trace(go.Bar(
            x=yr.index.astype(str), y=yr.values * 100,
            name=label, marker_color=color,
            hovertemplate="%{x}<br>%{y:.2f}%<extra>" + label + "</extra>",
        ))
    fig.update_layout(
        **base_layout,
        title  = dict(text="Annual Returns (%)", font=dict(size=14)),
        height = 380,
        barmode = "group",
        xaxis = _axis(dict(showgrid=False)),
        yaxis = _axis(dict(ticksuffix="%", zeroline=True,
                             zerolinecolor=arin_grid)),
    )
    return fig


# Drawdowns
def plot_drawdowns(backtest_map: dict) -> go.Figure:
    fig = go.Figure()
    for label, (bt, color) in backtest_map.items():
        if bt is None or bt.empty:
            continue
        ret = bt["portfolio_return"]
        cum = (1 + ret).cumprod()
        roll_max = cum.expanding().max()
        dd = (cum - roll_max) / roll_max * 100
        # Convert hex to rgba — Plotly only accepts 6-digit hex, not 8-digit
        hex_c = color.lstrip("#")
        r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        fill_color = f"rgba({r},{g},{b},0.10)"
        fig.add_trace(go.Scatter(
            x=bt["date"], y=dd, mode="lines", name=label,
            line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=fill_color,
            hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra>" + label + "</extra>",
        ))
    fig.update_layout(
        **base_layout,
        title = dict(text="Drawdowns (%)", font=dict(size=14)),
        height = 360,
        xaxis = _axis(),
        yaxis = _axis(dict(ticksuffix="%", zeroline=True,
                            zerolinecolor=arin_grid)),
    )
    return fig


# Monthly Returns Heatmap 
def plot_monthly_heatmap(backtest_df: pd.DataFrame, portfolio_label: str) -> go.Figure:
    
    if backtest_df is None or backtest_df.empty:
        return go.Figure()

    bt = backtest_df.copy()
    bt["year"] = bt["date"].dt.year
    bt["month"] = bt["date"].dt.month

    pivot = bt.pivot_table(index="year", columns="month",
                           values="portfolio_return", aggfunc="sum") * 100
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_labels[m - 1] for m in pivot.columns]

    vals = pivot.values[~np.isnan(pivot.values)]
    zmax = max(abs(vals.max()), abs(vals.min()), 0.01) if len(vals) else 1.0
    text = pivot.map(lambda v: f"{v:.1f}%" if not np.isnan(v) else "")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.astype(str).tolist(),
        text=text.values,
        texttemplate="%{text}",
        colorscale=[
            [0.0, "#c0392b"],
            [0.5, arin_navy],
            [1.0, arin_sky_blue],
        ],
        zmid=0, zmin=-zmax, zmax=zmax,
        showscale=True,
        hovertemplate="%{y} %{x}<br>Return: %{z:.2f}%<extra></extra>",
        colorbar=dict(
            ticksuffix="%",
            tickfont=dict(color=arin_white),
            title=dict(text="Return", font=dict(color=arin_white)),
        ),
    ))
    fig.update_layout(
        **base_layout,
        title = dict(text=f"Monthly Returns — {portfolio_label}", font=dict(size=14)),
        height = max(200, 60 * len(pivot) + 80),
        xaxis = _axis(),
        yaxis = _axis(),
    )
    return fig


# Portfolio Allocation Pie
def plot_allocation_pie(tickers: list, weights: list,
                        title: str = "Portfolio Allocation") -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=[f"{t}" for t in tickers],
        values=[w * 100 for w in weights],
        hole=0.4,
        textinfo="label+percent",
        hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
        marker=dict(line=dict(color=arin_navy, width=2)),
    ))
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = dict(family="Segoe UI, Arial, sans-serif", size=12, color=arin_white),
        margin = dict(l=10, r=10, t=40, b=10),
        title = dict(text=title, font=dict(size=13)),
        height = 360,
        showlegend = True,
        legend = dict(bgcolor=arin_legend_bg, bordercolor=arin_grid,
                             borderwidth=1, font=dict(size=10, color=arin_white)),
    )
    return fig


# Risk Contribution Bar 
def plot_risk_contributions(tickers: list, contributions: list,
                             title: str = "Risk Contributions") -> go.Figure:
    palette = px.colors.qualitative.Plotly
    fig = go.Figure(go.Bar(
        x=tickers,
        y=[c * 100 for c in contributions],
        marker_color=palette[:len(tickers)],
        hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
    ))
    eq = 100.0 / len(tickers)
    fig.add_hline(y=eq, line_dash="dash", line_color=arin_gray,
                  annotation_text=f"Equal: {eq:.1f}%",
                  annotation_position="top right",
                  annotation_font_color=arin_white)
    fig.update_layout(
        **base_layout,
        title = dict(text=title, font=dict(size=13)),
        height = 340,
        showlegend = False,
        xaxis = _axis(dict(showgrid=False)),
        yaxis = _axis(dict(ticksuffix="%")),
    )
    return fig


# Correlation Matrix
def plot_correlation_matrix(returns_df: pd.DataFrame,
                             title: str = "Asset Correlation Matrix") -> go.Figure:
    corr = returns_df.corr()
    # Keep only the lower triangle — mask upper triangle with None 
    mask = np.triu(np.ones(corr.shape, dtype=bool))
    z = np.where(mask, corr.values, None)
    text = [[f"{corr.values[r][c]:.2f}" if mask[r][c] else ""
             for c in range(corr.shape[1])]
            for r in range(corr.shape[0])]
    
    fig = go.Figure(go.Heatmap(
        z=z,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        text=text,
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(
            tickfont=dict(color=arin_white),
            title=dict(text="ρ", font=dict(color=arin_white)),
        ),
        hovertemplate="%{y} / %{x}<br>ρ = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **base_layout,
        title = dict(text=title, font=dict(size=13)),
        height = 420,
        xaxis = _axis(),
        yaxis = _axis(),
    )
    return fig


# Rolling Sharpe 
def plot_rolling_sharpe(backtest_map: dict, window: int = 12) -> go.Figure:
    fig = go.Figure()
    for label, (bt, color) in backtest_map.items():
        if bt is None or bt.empty or len(bt) < window + 2:
            continue
        ret   = bt["portfolio_return"]
        roll_mean = ret.rolling(window).mean()
        roll_std = ret.rolling(window).std()
        roll_sharpe = (roll_mean * np.sqrt(12)) / (roll_std * np.sqrt(12))
        fig.add_trace(go.Scatter(
            x=bt["date"], y=roll_sharpe, mode="lines", name=label,
            line=dict(color=color, width=1.8),
            hovertemplate="%{x|%b %Y}<br>Sharpe: %{y:.2f}<extra>" + label + "</extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=arin_gray)
    fig.update_layout(
        **base_layout,
        title = dict(text=f"Rolling {window}-Month Sharpe Ratio", font=dict(size=14)),
        height = 340,
        xaxis = _axis(),
        yaxis = _axis(),
    )
    return fig
