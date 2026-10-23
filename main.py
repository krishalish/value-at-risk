"""
Value at Risk for an equal-weight AAPL / GOOGL / MSFT / TSLA / AMZN portfolio.

    python main.py                    # real prices (Yahoo Finance), full report + 2025 backtest
    python main.py --data synthetic   # offline demo on simulated normal returns
    python main.py --figures          # also redraw the README charts in docs/img/
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from var_system.data_generator import DataGenerator
from var_system.var_calculator import VaRCalculator
from var_system.analysis import RiskAnalyzer
from var_system.advanced_risk import AdvancedRiskModel
from var_system.market_data import load_returns, TICKERS
from var_system.backtest import rolling_var_forecasts, summarise_backtest

warnings.filterwarnings('ignore')
pd.set_option('display.width', 120)


def format_var_results(results: dict) -> pd.DataFrame:
    """Format VaR results into a readable DataFrame."""
    formatted_data = []

    for asset, var_values in results.items():
        row = {'Asset': asset}
        for conf_level, var_value in var_values.items():
            row[f'{conf_level*100:.0f}% VaR'] = f'{var_value*100:.2f}%'
        formatted_data.append(row)

    return pd.DataFrame(formatted_data).set_index('Asset')

def print_section_header(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}\n{title}\n{'='*80}")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', choices=['real', 'synthetic'], default='real')
    parser.add_argument('--start', default='2019-01-01', help='first date of price history')
    parser.add_argument('--end', default='2026-01-01', help='end of price history (exclusive)')
    parser.add_argument('--window', type=int, default=500, help='estimation window in trading days')
    parser.add_argument('--refresh', action='store_true', help='re-download prices')
    parser.add_argument('--figures', action='store_true', help='redraw README charts in docs/img/')
    return parser.parse_args()

def main():
    args = parse_args()
    tickers = TICKERS
    confidence_levels = [0.95, 0.99]
    horizons = [1, 10, 22]  # 1-day, 10-day, and 22-day VaR
    target_return = 0.001  # Target daily return for optimization

    # One-day stress scenarios (shock to each stock's price)
    scenarios = {
        'Market Crash': {ticker: -0.15 for ticker in tickers},
        'Tech Selloff': {'AAPL': -0.20, 'GOOGL': -0.20, 'MSFT': -0.20,
                        'TSLA': -0.25, 'AMZN': -0.20},
        'Recovery': {ticker: 0.10 for ticker in tickers},
        'Mixed': {'AAPL': 0.05, 'GOOGL': -0.05, 'MSFT': 0.05,
                 'TSLA': -0.10, 'AMZN': -0.05}
    }

    os.makedirs('plots', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # ---------------------------------------------------------------- data
    if args.data == 'real':
        print_section_header(f"Loading daily prices {args.start} to {args.end} (Yahoo Finance)")
        all_returns = load_returns(tickers, args.start, args.end, refresh=args.refresh)
        # Point-in-time VaR uses the most recent estimation window
        returns = all_returns.iloc[-args.window:]
        print(f"{len(all_returns)} trading days; snapshot window {returns.index[0].date()} to {returns.index[-1].date()}")
    else:
        print_section_header("Generating synthetic normal returns")
        all_returns = None
        returns = DataGenerator(tickers=tickers, days=args.window, use_real_data=False).generate_returns()
    weights = {t: 1 / len(tickers) for t in tickers}

    var_calc = VaRCalculator(returns=returns, weights=weights, confidence_levels=confidence_levels)
    adv_risk = AdvancedRiskModel(returns=returns, weights=weights,
                                 confidence_levels=confidence_levels, target_return=target_return)

    # ---------------------------------------------------------------- point-in-time VaR
    print_section_header("One-day VaR, four methods")
    hist_var = var_calc.historical_var()
    print("\n1. Historical simulation\n", format_var_results(hist_var))
    param_var = var_calc.parametric_var()
    print("\n2. Parametric (normal)\n", format_var_results(param_var))
    mc_var = var_calc.monte_carlo_var()
    print("\n3. Monte Carlo (normal draws, 10,000 runs)\n", format_var_results(mc_var))
    garch_var = adv_risk.calculate_garch_var()
    print("\n4. GARCH(1,1)\n", format_var_results(garch_var))

    print_section_header("Multi-horizon VaR (square-root-of-time scaling of historical VaR)")
    horizon_vars = adv_risk.scale_var(hist_var, horizons)
    for horizon in horizons:
        print(f"\n{horizon}-day:\n", format_var_results({'Portfolio': horizon_vars[horizon]['Portfolio']}))

    print_section_header("Stressed VaR (volatility x 1.5)")
    stressed_var = adv_risk.calculate_stressed_var()
    print(format_var_results(stressed_var))

    print_section_header("Component VaR at 95% (contribution of each stock)")
    component_var = adv_risk.calculate_component_var()
    print((component_var * 100).round(3).astype(str) + '%')

    print_section_header("Minimum-variance weights for the target return")
    try:
        opt_weights = adv_risk.optimize_portfolio()
        for ticker, weight in opt_weights.items():
            print(f"{ticker}: {weight*100:.1f}%")
    except Exception as e:
        print(f"Optimisation not feasible for this target: {e}")

    print_section_header("Scenario analysis (one-day portfolio return)")
    scenario_results = adv_risk.run_scenario_analysis(scenarios)
    for scenario, result in scenario_results.items():
        print(f"{scenario}: {result*100:.2f}%")

    print_section_header("Bootstrap 95% confidence intervals for 95% historical VaR")
    confidence_intervals = adv_risk.calculate_var_confidence_intervals()
    print((pd.DataFrame(confidence_intervals) * 100).round(2))

    # ---------------------------------------------------------------- normality check
    port = (all_returns if all_returns is not None else returns) @ pd.Series(weights)
    print_section_header("How normal are the portfolio returns?")
    print(f"skewness {stats.skew(port):.2f}, excess kurtosis {stats.kurtosis(port):.2f} (normal: 0, 0)")
    for c in confidence_levels:
        h = -np.quantile(port, 1 - c)
        n = stats.norm.ppf(c) * port.std() - port.mean()
        print(f"{c:.0%} VaR over the full sample: historical {h*100:.2f}%  vs  normal {n*100:.2f}%")

    # ---------------------------------------------------------------- 2025 backtest
    if all_returns is not None and len(all_returns.loc['2025']) > 0:
        print_section_header(f"Out-of-sample backtest, 2025 (rolling {args.window}-day window)")
        forecasts = rolling_var_forecasts(all_returns, weights, confidence_levels, window=args.window)
        summary = summarise_backtest(forecasts, confidence_levels)
        shown = summary.assign(breach_rate=(summary.breach_rate * 100).round(1),
                               avg_var=(summary.avg_var * 100).round(2),
                               kupiec_p=summary.kupiec_p.round(3),
                               independence_p=summary.independence_p.round(3))
        print(shown.to_string(index=False))

        print_section_header("April 2025 tariff sell-off (99% VaR, %)")
        april = forecasts.loc['2025-03-31':'2025-04-15',
                              ['return', 'historical_0.99', 'normal_0.99', 'garch_0.99']] * 100
        april.index = april.index.date
        print(april.round(2))
        forecasts.to_csv('output/backtest_2025.csv')
        summary.to_csv('output/backtest_summary.csv', index=False)

        if args.figures:
            from var_system.plots import plot_backtest, plot_tail, plot_garch_vol
            os.makedirs('docs/img', exist_ok=True)
            plot_backtest(forecasts, 'docs/img/backtest_2025.png')
            plot_tail(port, 'docs/img/fat_tail.png')
            plot_garch_vol(port, 'docs/img/garch_volatility.png')
            print("\n✓ README charts saved to docs/img/")

    # ---------------------------------------------------------------- outputs
    risk_analyzer = RiskAnalyzer(returns=returns, portfolio_returns=var_calc.portfolio_returns,
                                 var_results=hist_var, confidence_levels=confidence_levels)
    risk_analyzer.plot_return_distributions(save_path='plots/return_distributions.png')
    risk_analyzer.plot_correlation_heatmap(save_path='plots/correlation_heatmap.png')
    risk_analyzer.plot_rolling_var(save_path='plots/rolling_var.png')
    adv_risk.export_to_excel(
        filename='output/var_analysis_results.xlsx',
        var_results=hist_var,
        garch_var=garch_var,
        stressed_var=stressed_var,
        component_var=component_var,
        scenario_results=scenario_results,
        confidence_intervals=confidence_intervals
    )
    print("\n✓ Plots saved to plots/, results exported to output/var_analysis_results.xlsx")

if __name__ == "__main__":
    main()
