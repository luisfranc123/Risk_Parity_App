"""
Risk Parity Portfolio Analyzer 
=====================================================
Replicates the Portfolio Visualizer report from the PDF with:
- Aspect Partners Risk Parity  (customisable)
- SPY + Long Vol
- SPY + Long Bond
- State Street Global Allocation ETF  (benchmark)
"""

import sys, os
import matplotlib
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

from utils.data_utils import (
    default_portfolios, benchmark, ticker_names,
    fetch_prices, build_portfolio_returns,
)
from utils.performance_metrics import calculate_performance_metrics
from utils.charts import (
    plot_portfolio_growth, plot_annual_returns, plot_drawdowns,
    plot_monthly_heatmap, plot_allocation_pie, plot_risk_contributions,
    plot_correlation_matrix, plot_rolling_sharpe,
)
from utils.optimize_portfolio import optimize_risk_parity, calculate_risk_contributions

st.set_page_config(
    page_title="Risk Parity Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* ── Arin Risk Advisors brand palette ── */
  :root{
    --bg: #0D2948; /* Deep Navy Blue — main background */
    --surf: #0a2038; /* Darker navy — card/surface background */
    --surf2: #071828; /* Deepest navy — sidebar background */
    --acc: #1B8FFB; /* Sky Blue — accents, links, highlights */
    --txt: #FFFFFF; /* White — primary text on navy */
    --mut: #80807F; /* Neutral Gray — secondary / muted text */
    --border: #1a3a5c; /* Navy tint — borders and dividers */
    --black: #313131; /* Brand black — used in tables/badges */
  }
  .stApp{background:var(--bg);color:var(--txt);
         font-family:'Calibri','Georgia',serif}
  h1,h2,h3,h4,h5,h6{font-family:'Segoe UI','Arial',sans-serif}
  section[data-testid="stSidebar"]{background:var(--surf2)!important}
  .kpi{background:var(--surf);border-radius:10px;padding:14px 18px;
       border:1px solid var(--border);text-align:center}
  .kpi .lbl{font-size:11px;color:var(--mut);text-transform:uppercase;
             letter-spacing:.06em;margin-bottom:4px;
             font-family:'Segoe UI','Arial',sans-serif}
  .kpi .val{font-size:22px;font-weight:700;color:var(--acc)}
  .kpi .sub{font-size:11px;color:var(--mut);margin-top:2px}
  .sh{font-size:17px;font-weight:700;color:var(--acc);
      font-family:'Segoe UI','Arial',sans-serif;
      border-bottom:1px solid var(--border);
      padding-bottom:6px;margin:18px 0 12px}
</style>
""", unsafe_allow_html=True)

# Helpers 
def fp(v, d=2):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
    return f"{v*100:.{d}f}%"
def fd(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
    return f"${v:,.2f}"
def fn(v, d=2):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
    return f"{v:.{d}f}"
def kpi(lbl, val, sub=""):
    return f'<div class="kpi"><div class="lbl">{lbl}</div><div class="val">{val}</div><div class="sub">{sub}</div></div>'

labels = {
    "ap": "Aspect Partners Risk Parity",
    "lv": "SPY + Long Vol",
    "lb": "SPY + Long Bond",
    "bench": "State Street Global Allocation ETF",
}
#  Arin Risk Advisors brand colors assigned by portfolio role 
colors = {
    "ap": "#1B8FFB",
    "lv": "#80807F",
    "lb": "#5B9BD5",
    "bench": "#FFFFFF",
}


# SIDEBAR
with st.sidebar:
    st.markdown("## Settings")

    st.markdown("### Date Range")
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start", date(2024,1,1), min_value=date(2018,1,1))
    end_date   = c2.date_input("End",   date.today(),   max_value=date.today())
    if start_date >= end_date:
        st.error("Start must be before End."); st.stop()

    st.markdown("### Initial Investment")
    initial_value = st.number_input("Amount ($)", 1_000, 10_000_000, 10_000, 1_000, "%d")

    st.markdown("### Rebalancing")
    rebalance_freq = st.radio("Frequency", ["monthly","quarterly"], index=1,
                              format_func=str.title)

    st.markdown("### Dividends")
    reinvest_divs = st.toggle(
        "Reinvest dividends",
        value=True,
        help=(
            "ON — uses Yahoo Finance adjusted prices, which embed dividends "
            "automatically (total return). "
            "OFF — uses unadjusted prices so dividend income is excluded "
            "from the backtest (price return only)."
        ),
    )

    st.markdown("### Aspect Partners — Holdings")
    with st.expander("Customise tickers & weights", expanded=False):
        st.caption("Edit tickers or weights. Weights are auto-normalised.")
        dflt = default_portfolios["Aspect Partners Risk Parity"]
        ticker_txt = st.text_area("Tickers (one per line)",
                                  "\n".join(dflt["tickers"]), height=220)
        weight_txt = st.text_area("Weights (one per line, same order)",
                                  "\n".join(str(w) for w in dflt["weights"]), height=220)

    try:
        ct = [t.strip().upper() for t in ticker_txt.split("\n") if t.strip()]
        cw_raw = [float(w.strip()) for w in weight_txt.split("\n") if w.strip()]
        if len(ct) != len(cw_raw):
            st.warning("Ticker/weight count mismatch — using defaults.")
            ct, cw_raw = dflt["tickers"], dflt["weights"]
        s = sum(cw_raw); cw = [w/s for w in cw_raw]
    except Exception:
        ct, cw = dflt["tickers"], dflt["weights"]

    run = st.button("▶ Run Analysis", type="primary", use_container_width=True)
    st.markdown("---"); st.caption("Data: Yahoo Finance · Streamlit")


# HEADER
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:6px">
  <span style="font-size:42px"></span>
  <div>
    <h1 style="margin:0;font-size:28px;color:#1B8FFB;
               font-family:'Segoe UI','Arial',sans-serif;font-weight:700">
      Risk Parity Portfolio Analyzer
    </h1>
    <p style="margin:0;color:#80807F;font-size:13px;
              font-family:'Calibri','Georgia',serif">
      Walk-forward backtest · Equal risk contribution · Full performance report
    </p>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

if "results" not in st.session_state:
    st.session_state.results = None


# RUN
if run:
    st.session_state.results = None
    all_tix = list(set(ct +
        default_portfolios["SPY + Long Vol"]["tickers"] +
        default_portfolios["SPY + Long Bond"]["tickers"] +
        [benchmark["ticker"]]))

    with st.spinner("Downloading price data…"):
        prices = fetch_prices(all_tix, str(start_date), str(end_date),
                              reinvest_dividends=reinvest_divs)
    if prices.empty:
        st.error("No data returned. Check tickers / dates."); st.stop()

    with st.spinner("Running backtests…"):
        def run_bt(tickers, weights, rp=True):
            return build_portfolio_returns(prices, tickers, weights, rebalance_freq, rp)
        bt_ap    = run_bt(ct, cw, True)
        bt_lv    = run_bt(default_portfolios["SPY + Long Vol"]["tickers"],
                          default_portfolios["SPY + Long Vol"]["weights"], False)
        bt_lb    = run_bt(default_portfolios["SPY + Long Bond"]["tickers"],
                          default_portfolios["SPY + Long Bond"]["weights"], False)
        bt_bench = run_bt([benchmark["ticker"]], [1.0], False)

    bench_monthly = (bt_bench.set_index("date")["portfolio_return"]
                     if not bt_bench.empty else None)

    def safe_metrics(bt):
        if bt is None or bt.empty: return None
        try:
            return calculate_performance_metrics(bt, risk_free_rate=0.04,
                benchmark_returns=bench_monthly, initial_value=initial_value)
        except Exception as e:
            st.warning(f"Metrics error: {e}"); return None

    with st.spinner("Computing metrics…"):
        mets = {k: safe_metrics(b) for k, b in
                zip(["ap","lv","lb","bench"],[bt_ap,bt_lv,bt_lb,bt_bench])}

    # Scale portfolio values to initial_value
    scale = initial_value / 10_000.0
    for b in [bt_ap, bt_lv, bt_lb, bt_bench]:
        if b is not None and not b.empty:
            b["portfolio_value"] *= scale

    st.session_state.results = dict(
        prices=prices, bt=dict(ap=bt_ap,lv=bt_lv,lb=bt_lb,bench=bt_bench),
        mets=mets, ct=ct, cw=cw, iv=initial_value,
        reinvest_divs=reinvest_divs,
    )
    st.success("Analysis complete!")

# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.results is None:
    st.info("Configure the settings in the sidebar and click **▶ Run Analysis**.")
    st.stop()

R      = st.session_state.results
bt     = R["bt"]; mets = R["mets"]
prices = R["prices"]; iv = R["iv"]

bmap   = {labels[k]: (bt[k],  colors[k]) for k in ["ap","lv","lb","bench"]}
mmap   = {labels[k]: (mets[k],colors[k]) for k in ["ap","lv","lb","bench"]}


# TABS
t1, t2, t3, t4, t5 = st.tabs([
    "▲ Overview", "± Risk & Metrics", "◈ Holdings",
    "▦ Monthly Returns", "◎ Components"
])


# TAB 1 — OVERVIEW

with t1:
    st.markdown('<div class="sh">Portfolio Summary</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, k in enumerate(["ap","lv","lb","bench"]):
        m = mets[k]
        with cols[i]:
            if m:
                end = m["End_Balance"] * (iv/10_000)
                st.markdown(kpi(labels[k], fd(end), f"CAGR {fp(m['CAGR'])}"),
                            unsafe_allow_html=True)
            else:
                st.markdown(kpi(labels[k],"—","No data"), unsafe_allow_html=True)

    st.plotly_chart(plot_portfolio_growth(bmap, iv), use_container_width=True)

    # Dividend reinvestment indicator
    div_label = "Dividends reinvested (total return)" if R.get("reinvest_divs", True) \
                else "Dividends excluded (price return only)"
    st.caption(div_label)
    st.plotly_chart(plot_annual_returns(mmap),        use_container_width=True)

    # Performance table
    st.markdown('<div class="sh">Portfolio Performance</div>', unsafe_allow_html=True)
    PERF_ROWS = [
        ("Start Balance", "Start_Balance", "d"),
        ("End Balance", "End_Balance", "ds"),
        ("End Balance (inflation adjusted)", "End_Balance_Real", "ds"),
        ("Annualized Return (CAGR)", "CAGR", "p"),
        ("CAGR (inflation adjusted)", "CAGR_Real", "p"),
        ("Standard Deviation", "Std_Dev_Annual", "p"),
        ("Best Year", "Best_Year", "p"),
        ("Worst Year", "Worst_Year", "p"),
        ("Maximum Drawdown", "Maximum_Drawdown", "p"),
        ("Sharpe Ratio", "Sharpe_Ratio", "n"),
        ("Sortino Ratio", "Sortino_Ratio", "n"),
        ("Benchmark Correlation", "Benchmark_Correlation","n"),
    ]
    rows = {}
    for lbl, key, fmt in PERF_ROWS:
        row = {}
        for k in ["ap","lv","lb","bench"]:
            v = mets[k][key] if mets[k] else None
            if v is None or (isinstance(v,float) and np.isnan(v)):
                row[labels[k]] = "—"
            elif fmt=="d": row[labels[k]] = fd(v)
            elif fmt=="ds": row[labels[k]] = fd(v*(iv/10_000))
            elif fmt=="p": row[labels[k]] = fp(v)
            else: row[labels[k]] = fn(v)
        rows[lbl] = row
    st.dataframe(pd.DataFrame(rows).T, use_container_width=True)

    # Trailing returns
    st.markdown('<div class="sh">Trailing Returns</div>', unsafe_allow_html=True)
    tr = {}
    for k in ["ap","lv","lb","bench"]:
        b = bt[k]
        if b is None or b.empty:
            tr[labels[k]] = {"3M":"—","YTD":"—","1Y":"—","Full":"—"}; continue
        latest = b["date"].max()
        def trail(mo):
            s = b[b["date"] >= latest - pd.DateOffset(months=mo)]
            return fp((1+s["portfolio_return"]).prod()-1) if not s.empty else "—"
        ytd_s = b[b["date"] >= pd.Timestamp(latest.year,1,1)]
        tr[labels[k]] = {"3M":trail(3),
                         "YTD": fp((1+ytd_s["portfolio_return"]).prod()-1) if not ytd_s.empty else "—",
                         "1Y":trail(12),
                         "Full": fp((1+b["portfolio_return"]).prod()-1)}
    st.dataframe(pd.DataFrame(tr).T, use_container_width=True)



# TAB 2 — RISK & METRICS
with t2:
    st.plotly_chart(plot_drawdowns(bmap), use_container_width=True)
    st.plotly_chart(plot_rolling_sharpe(bmap, 12), use_container_width=True)

    st.markdown('<div class="sh">Full Risk & Return Metrics</div>', unsafe_allow_html=True)
    RISK_ROWS = [
        ("Arithmetic Mean (monthly)", "Arithmetic_Mean_Monthly", "p"),
        ("Arithmetic Mean (annualized)", "Arithmetic_Mean_Annual", "p"),
        ("Geometric Mean (monthly)", "Geometric_Mean_Monthly", "p"),
        ("Geometric Mean (annualized)",  "Geometric_Mean_Annual", "p"),
        ("Std Dev (monthly)", "Std_Dev_Monthly", "p"),
        ("Std Dev (annualized)", "Std_Dev_Annual", "p"),
        ("Downside Deviation (monthly)", "Downside_Deviation_Monthly","p"),
        ("Maximum Drawdown", "Maximum_Drawdown", "p"),
        ("Benchmark Correlation", "Benchmark_Correlation", "n"),
        ("Beta", "Beta", "n"),
        ("Alpha (annualized)", "Alpha_Annualized", "p"),
        ("R²", "R_Squared", "n4"),
        ("Sharpe Ratio", "Sharpe_Ratio", "n"),
        ("Sortino Ratio", "Sortino_Ratio", "n"),
        ("Treynor Ratio (%)", "Treynor_Ratio_Pct", "n"),
        ("M² Measure", "Modigliani_M2", "p"),
        ("Active Return", "Active_Return", "p"),
        ("Tracking Error", "Tracking_Error", "p"),
        ("Information Ratio", "Information_Ratio", "n"),
        ("Skewness", "Skewness", "n4"),
        ("Excess Kurtosis", "Excess_Kurtosis", "n4"),
        ("Historical VaR (5%)", "Historical_VaR_5pct", "p"),
        ("Analytical VaR (5%)", "Analytical_VaR_5pct", "p"),
        ("Conditional VaR (5%)", "Conditional_VaR_5pct", "p"),
        ("Upside Capture Ratio (%)", "Upside_Capture_Ratio", "n"),
        ("Downside Capture Ratio (%)", "Downside_Capture_Ratio", "n"),
        ("Positive Periods (%)", "Positive_Periods_Pct", "p"),
        ("Gain/Loss Ratio", "Gain_Loss_Ratio", "n"),
    ]
    rrows = {}
    for lbl, key, fmt in RISK_ROWS:
        row = {}
        for k in ["ap","lv","lb","bench"]:
            v = mets[k].get(key) if mets[k] else None
            if v is None or (isinstance(v,float) and np.isnan(v)):
                row[labels[k]] = "—"
            elif fmt=="p":  row[labels[k]] = fp(v)
            elif fmt=="n4": row[labels[k]] = fn(v,4)
            else:           row[labels[k]] = fn(v)
        rrows[lbl] = row
    st.dataframe(pd.DataFrame(rrows).T, use_container_width=True)



# TAB 3 — HOLDINGS
with t3:
    st.markdown('<div class="sh">Aspect Partners — Holdings</div>', unsafe_allow_html=True)
    tickers, weights = R["ct"], R["cw"]
    c1, c2 = st.columns([1,1])
    with c1:
        st.plotly_chart(plot_allocation_pie(tickers, weights, "Client Allocation"),
                        use_container_width=True)
    with c2:
        st.dataframe(pd.DataFrame({
            "Ticker": tickers,
            "Name":   [ticker_names.get(t,t) for t in tickers],
            "Weight": [fp(w) for w in weights],
        }), use_container_width=True, hide_index=True)

    # Risk parity optimised weights
    avail = [t for t in tickers if t in prices.columns]
    if avail:
        dr = prices[avail].pct_change().dropna()
        mr = dr.resample("ME").apply(lambda x:(1+x).prod()-1)
        if len(mr) >= 3:
            cov = mr.cov().values * 12
            try:
                rp_w = optimize_risk_parity(cov)
                rc   = calculate_risk_contributions(rp_w, cov)
                st.markdown('<div class="sh">Risk Parity Optimised Weights</div>',
                            unsafe_allow_html=True)
                ca, cb = st.columns(2)
                with ca:
                    st.plotly_chart(plot_allocation_pie(avail,rp_w,"RP Weights"),
                                    use_container_width=True)
                with cb:
                    st.plotly_chart(plot_risk_contributions(avail,rc,"Risk Contributions"),
                                    use_container_width=True)
                st.dataframe(pd.DataFrame({
                    "Ticker": avail,
                    "Name":   [ticker_names.get(t,t) for t in avail],
                    "Client Wt": [fp(w) for w in weights[:len(avail)]],
                    "RP Wt":    [fp(w) for w in rp_w],
                    "Risk Contrib": [fp(c) for c in rc],
                }), use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Optimisation error: {e}")

    # Correlation matrix
    st.markdown('<div class="sh">Asset Correlation Matrix</div>', unsafe_allow_html=True)
    if len(avail) > 1:
        st.plotly_chart(
            plot_correlation_matrix(prices[avail].pct_change().dropna()),
            use_container_width=True)

    # Comparison portfolio pies
    st.markdown('<div class="sh">Comparison Portfolios</div>', unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        lv = default_portfolios["SPY + Long Vol"]
        st.plotly_chart(plot_allocation_pie(lv["tickers"],lv["weights"],"SPY + Long Vol"),
                        use_container_width=True)
    with cc2:
        lb = default_portfolios["SPY + Long Bond"]
        st.plotly_chart(plot_allocation_pie(lb["tickers"],lb["weights"],"SPY + Long Bond"),
                        use_container_width=True)
    with cc3:
        st.plotly_chart(plot_allocation_pie([benchmark["ticker"]],[1.0],benchmark["name"]),
                        use_container_width=True)



# TAB 4 — MONTHLY RETURNS
with t4:
    st.markdown('<div class="sh">Monthly Return Heatmaps</div>', unsafe_allow_html=True)
    for k in ["ap","lv","lb","bench"]:
        b = bt[k]
        if b is not None and not b.empty:
            st.plotly_chart(plot_monthly_heatmap(b, labels[k]), use_container_width=True)

    st.markdown('<div class="sh">Monthly Returns Table</div>', unsafe_allow_html=True)
    monthly_series = {}
    for k in ["ap","lv","lb","bench"]:
        b = bt[k]
        if b is None or b.empty: continue
        s = b.set_index("date")["portfolio_return"] * 100
        monthly_series[labels[k]] = s
    if monthly_series:
        df_m = pd.DataFrame(monthly_series).round(2)
        df_m.index = df_m.index.strftime("%b %Y")
        def color_returns(val):
            try:
                v = float(str(val).replace("%", ""))
                if v > 0:
                    return "color: #1B8FFB; font-weight: 600"
                elif v < 0:
                    return "color: #c0392b; font-weight: 600"
                else:
                    return ""
            except Exception:
                return ""

st.dataframe(
    df_m.style.format("{:.2f}%").applymap(color_returns),
    use_container_width=True)



# TAB 5 — COMPONENTS
with t5:
    st.markdown('<div class="sh">Individual Asset Performance</div>', unsafe_allow_html=True)
    all_t = list(set(R["ct"] +
        default_portfolios["SPY + Long Vol"]["tickers"] +
        default_portfolios["SPY + Long Bond"]["tickers"]))
    avail_t = [t for t in all_t if t in prices.columns]

    if avail_t:
        dr_all = prices[avail_t].pct_change().dropna()
        mr_all = dr_all.resample("ME").apply(lambda x:(1+x).prod()-1)

        comp_rows = []
        for t in avail_t:
            if t not in mr_all.columns: continue
            s = mr_all[t].dropna()
            if s.empty: continue
            yrs = len(s)/12
            c   = (1+s).prod()**(1/max(yrs,0.01))-1
            sd  = s.std()*np.sqrt(12)
            sh  = (s.mean()*np.sqrt(12))/sd if sd else np.nan
            yr  = s.resample("YE").apply(lambda x:(1+x).prod()-1)
            cum = (1+s).cumprod(); rm = cum.expanding().max()
            mdd = ((cum-rm)/rm).min()
            comp_rows.append({
                "Ticker":t, "Name":ticker_names.get(t,t),
                "CAGR":fp(c), "Std Dev":fp(sd),
                "Best Year":fp(yr.max()) if not yr.empty else "—",
                "Worst Year":fp(yr.min()) if not yr.empty else "—",
                "Max Drawdown":fp(mdd), "Sharpe":fn(sh),
            })
        st.dataframe(pd.DataFrame(comp_rows).set_index("Ticker"),
                     use_container_width=True)

        # Normalised price chart
        st.markdown('<div class="sh">Normalized Price Growth (base = 100)</div>',
                    unsafe_allow_html=True)
        import plotly.graph_objects as go
        import plotly.express as px
        norm = (prices[avail_t]/prices[avail_t].iloc[0]*100).dropna(how="all")
        pal  = px.colors.qualitative.Plotly + px.colors.qualitative.Pastel
        fig  = go.Figure()
        for i,t in enumerate(avail_t):
            if t not in norm.columns: continue
            fig.add_trace(go.Scatter(x=norm.index, y=norm[t], mode="lines",
                name=t, line=dict(width=1.4, color=pal[i%len(pal)]),
                hovertemplate=f"{t}<br>%{{x|%b %Y}}<br>%{{y:.1f}}<extra></extra>"))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font  = dict(family="Segoe UI, Arial, sans-serif", size=12, color="#FFFFFF"),
            height= 440,
            xaxis = dict(gridcolor="#1a3a5c", showgrid=True, zeroline=False,
                         tickfont=dict(color="#80807F")),
            yaxis = dict(gridcolor="#1a3a5c", showgrid=True, zeroline=False,
                         tickfont=dict(color="#80807F")),
            legend= dict(bgcolor="rgba(7,24,40,0.85)", bordercolor="#1a3a5c",
                         borderwidth=1, font=dict(size=10, color="#FFFFFF")),
            margin= dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # Return decomposition
    st.markdown('<div class="sh">Return Decomposition — Aspect Partners</div>',
                unsafe_allow_html=True)
    ap_t = [t for t in R["ct"] if t in prices.columns]
    ap_w = R["cw"][:len(ap_t)]
    if ap_t and bt["ap"] is not None and not bt["ap"].empty:
        dr_ap = prices[ap_t].pct_change().dropna()
        mr_ap = dr_ap.resample("ME").apply(lambda x:(1+x).prod()-1)
        contribs = {
            t: ((1+mr_ap[t]).prod()-1)*w*iv
            for t,w in zip(ap_t,ap_w) if t in mr_ap.columns
        }
        if contribs:
            cdf = pd.DataFrame.from_dict(contribs,orient="index",columns=["$"])
            cdf["Name"] = [ticker_names.get(t,t) for t in cdf.index]
            cdf = cdf.sort_values("$",ascending=False)
            colors = ["#1B8FFB" if v>=0 else "#c0392b" for v in cdf["$"]]
            fig2 = go.Figure(go.Bar(x=cdf.index, y=cdf["$"], marker_color=colors,
                hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>"))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font  = dict(family="Segoe UI, Arial, sans-serif", size=12, color="#FFFFFF"),
                height= 360,
                xaxis = dict(gridcolor="#1a3a5c", showgrid=False, zeroline=False,
                             tickfont=dict(color="#80807F")),
                yaxis = dict(tickprefix="$", gridcolor="#1a3a5c", showgrid=True,
                             zeroline=False, tickfont=dict(color="#80807F")),
                title = dict(text="Dollar Return Attribution per Asset", font=dict(size=13)),
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(cdf[["Name","$"]].rename(columns={"$":"Dollar Contribution ($)"}),
                         use_container_width=True)