import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def calculate_performance_metrics(
    backtest_results: pd.DataFrame,
    inflation_rate: float = 0.025,
    risk_free_rate: float = 0.02,
    benchmark_returns: pd.Series = None,
    initial_value: float = 10_000.0,) -> dict:
    """
    Compute the full set of portfolio performance statistics.
    Returns a dictionary (no printing).
    """
    returns_series = backtest_results["portfolio_return"].copy()
    portfolio_values = backtest_results["portfolio_value"].copy()
    dates = backtest_results["date"].copy()

    monthly_rf = (1 + risk_free_rate)**(1/12) - 1

    # Align with benchmark 
    if benchmark_returns is not None:
        df_bench = benchmark_returns.to_frame("bench").reset_index()
        df_bench.columns = ["Date", "bench"]
        df_port = backtest_results[["date", "portfolio_return"]].rename(
            columns={"date": "Date"})
        aligned = pd.merge(df_bench, df_port, on="Date", how="inner").dropna()
        bench_aligned = aligned["bench"]
        port_aligned = aligned["portfolio_return"]
    else:
        bench_aligned = returns_series.copy()
        port_aligned = returns_series.copy()

    # 1. Balances 
    start_balance = initial_value
    end_balance = portfolio_values.iloc[-1]

    # 2. Period 
    days = (dates.iloc[-1] - dates.iloc[0]).days
    years = max(days/365, 0.01)

    # 3. Inflation-adjusted balance
    end_balance_real = end_balance/((1 + inflation_rate)**years)

    # 4. CAGR 
    total_return = (end_balance/start_balance) - 1
    cagr = (1 + total_return)**(1/years) - 1
    cagr_real = (1 + cagr)/(1 + inflation_rate) - 1

    # 5. Volatility
    annualized_vol = returns_series.std()*np.sqrt(12)
    std_monthly = returns_series.std()
    std_annual = std_monthly*np.sqrt(12)

    # 6. Best / Worst Year 
    yearly_returns = (
        backtest_results.groupby(backtest_results["date"].dt.year)["portfolio_return"]
        .apply(lambda x: (1 + x).prod() - 1)
    )
    best_year = yearly_returns.max() if len(yearly_returns) else np.nan
    worst_year = yearly_returns.min() if len(yearly_returns) else np.nan

    # 7. Drawdown
    cumulative = (1 + returns_series).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # 8. Sharpe 
    excess_returns = returns_series - monthly_rf
    sharpe_ratio = (excess_returns.mean()*np.sqrt(12))/annualized_vol if annualized_vol else np.nan

    # 9. Sortino 
    downside_ret = returns_series[returns_series < monthly_rf]
    downside_dev_ann = downside_ret.std()*np.sqrt(12) if len(downside_ret) > 1 else np.nan
    sortino_ratio = (excess_returns.mean()*np.sqrt(12))/downside_dev_ann if downside_dev_ann else np.nan

    # 10. Benchmark correlation 
    corr = bench_aligned.corr(port_aligned) if benchmark_returns is not None else np.nan

    # 11-14. Arithmetic / Geometric Means 
    mean_monthly = returns_series.mean()
    mean_annual = mean_monthly*12
    geom_mean_monthly = np.exp(np.mean(np.log(1 + returns_series.clip(lower=-0.9999)))) - 1
    geom_mean_annual = (1 + geom_mean_monthly)**12 - 1

    # 17. Downside deviation (monthly, semi-deviation) 
    margin = 0.001
    dd_vals = np.clip([v - margin for v in returns_series], -np.inf, 0)
    downside_dev_monthly = np.sqrt(np.nanmean(np.square(dd_vals)))

    # 18-19. OLS Beta, Alpha, R² 
    if len(bench_aligned) > 5:
        X_ols = sm.add_constant(bench_aligned.values)
        model = sm.OLS(port_aligned.values, X_ols).fit()
        beta = model.params[1]
        r_squared = model.rsquared

        excess_port = port_aligned - monthly_rf
        excess_bench = bench_aligned - monthly_rf
        X_alpha = sm.add_constant(excess_bench.values)
        model_alpha = sm.OLS(excess_port.values, X_alpha).fit()
        annual_alpha = model_alpha.params[0]*12
    else:
        beta, r_squared, annual_alpha = np.nan, np.nan, np.nan

    # 20. Treynor 
    ann_excess = cagr - risk_free_rate
    treynor_ratio = (ann_excess/beta)*100 if beta and beta != 0 else np.nan

    # 21. M² 
    bench_vol_ann = bench_aligned.std()*np.sqrt(12) if benchmark_returns is not None else annualized_vol
    m2 = (sharpe_ratio*bench_vol_ann) + risk_free_rate if not np.isnan(sharpe_ratio) else np.nan

    # 22-24. Active return, Tracking Error, Info Ratio 
    if benchmark_returns is not None and len(bench_aligned) > 3:
        bench_total = (1 + bench_aligned).prod() - 1
        bench_years = len(bench_aligned) / 12
        bench_cagr = (1 + bench_total)**(1 / bench_years) - 1
        active_return = cagr - bench_cagr
        active_series = port_aligned.values - bench_aligned.values
        tracking_error = active_series.std()*np.sqrt(12)
        info_ratio = (active_series.mean()*12 / tracking_error) if tracking_error else np.nan
    else:
        active_return = tracking_error = info_ratio = np.nan

    # 25-26. Skewness / Kurtosis 
    skewness = returns_series.skew()
    excess_kurtosis = returns_series.kurt()

    # 27-29. VaR 
    hist_var_5 = np.percentile(returns_series, 5)
    z_95 = stats.norm.ppf(0.05)
    analytical_var_5 = mean_monthly + z_95 * std_monthly
    cvar_5 = returns_series[returns_series <= hist_var_5].mean()

    # 30-31. Capture Ratios 
    if benchmark_returns is not None and len(bench_aligned) > 5:
        up_mask = bench_aligned > 0
        down_mask = bench_aligned < 0

        def annualized_cum(s, n):
            return (1 + s).prod()**(12 / max(n, 1)) - 1

        up_port_ann = annualized_cum(port_aligned[up_mask], up_mask.sum())
        up_bench_ann = annualized_cum(bench_aligned[up_mask], up_mask.sum())
        upside_capture = (up_port_ann / up_bench_ann * 100) if up_bench_ann else np.nan

        dn_port_ann = annualized_cum(port_aligned[down_mask], down_mask.sum())
        dn_bench_ann = annualized_cum(bench_aligned[down_mask], down_mask.sum())
        downside_capture = (dn_port_ann / dn_bench_ann * 100) if dn_bench_ann else np.nan
    else:
        upside_capture = downside_capture = np.nan

    # 32-33. Win/Loss 
    n_positive = int((returns_series > 0).sum())
    n_total = len(returns_series)
    pct_positive = n_positive/n_total if n_total else np.nan
    avg_gain = returns_series[returns_series > 0].mean()
    avg_loss = returns_series[returns_series < 0].mean()
    gain_loss_ratio = (avg_gain/abs(avg_loss)) if (avg_loss and avg_loss != 0) else np.nan

    return {
        "Start_Balance": start_balance,
        "End_Balance": end_balance,
        "End_Balance_Real": end_balance_real,
        "CAGR": cagr,
        "CAGR_Real": cagr_real,
        "Arithmetic_Mean_Monthly": mean_monthly,
        "Arithmetic_Mean_Annual": mean_annual,
        "Geometric_Mean_Monthly": geom_mean_monthly,
        "Geometric_Mean_Annual": geom_mean_annual,
        "Annualized_Volatility": annualized_vol,
        "Std_Dev_Monthly": std_monthly,
        "Std_Dev_Annual": std_annual,
        "Downside_Deviation_Monthly": downside_dev_monthly,
        "Maximum_Drawdown": max_drawdown,
        "Best_Year": best_year,
        "Worst_Year": worst_year,
        "Skewness": skewness,
        "Excess_Kurtosis": excess_kurtosis,
        "Historical_VaR_5pct": hist_var_5,
        "Analytical_VaR_5pct": analytical_var_5,
        "Conditional_VaR_5pct": cvar_5,
        "Sharpe_Ratio": sharpe_ratio,
        "Sortino_Ratio": sortino_ratio,
        "Treynor_Ratio_Pct": treynor_ratio,
        "Modigliani_M2": m2,
        "Alpha_Annualized": annual_alpha,
        "Beta": beta,
        "R_Squared": r_squared,
        "Benchmark_Correlation": corr,
        "Active_Return": active_return,
        "Tracking_Error": tracking_error,
        "Information_Ratio": info_ratio,
        "Upside_Capture_Ratio": upside_capture,
        "Downside_Capture_Ratio": downside_capture,
        "Positive_Periods": n_positive,
        "Total_Periods": n_total,
        "Positive_Periods_Pct": pct_positive,
        "Gain_Loss_Ratio": gain_loss_ratio,
        # Yearly series for bar chart
        "_yearly_returns": yearly_returns,
        "_drawdowns": drawdowns,
        "_cumulative": cumulative,
    }
