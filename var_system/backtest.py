"""Rolling out-of-sample VaR backtest with Kupiec and Christoffersen tests."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from scipy.special import xlogy

METHODS = ['historical', 'normal', 'monte_carlo', 'garch']


def rolling_var_forecasts(returns: pd.DataFrame, weights: Dict[str, float],
                          confidence_levels: List[float] = (0.95, 0.99),
                          window: int = 500, start: str = '2025-01-01',
                          end: str = '2025-12-31', simulations: int = 10000,
                          seed: int = 42) -> pd.DataFrame:
    """
    One-day-ahead portfolio VaR for every trading day in [start, end].

    Each forecast for day t uses only the `window` days before t, so the
    test is out of sample: the model never sees the return it is predicting.

    Returns:
        pd.DataFrame: column 'return' (realised portfolio return) plus one
        column per method and level, e.g. 'garch_0.99', holding VaR as a
        positive loss
    """
    w = pd.Series(weights)[returns.columns].values
    port = returns @ w
    rng = np.random.default_rng(seed)
    test_days = port.loc[start:end].index

    rows = []
    for day in test_days:
        i = port.index.get_loc(day)
        if i < window:
            continue
        hist_assets = returns.iloc[i - window:i]
        hist_port = port.iloc[i - window:i]
        row = {'date': day, 'return': port.iloc[i]}

        mu, sd = hist_port.mean(), hist_port.std()

        # Monte Carlo: simulate asset returns from a multivariate normal, then form the portfolio
        sims = rng.multivariate_normal(hist_assets.mean().values, hist_assets.cov().values, simulations) @ w

        # GARCH(1,1) on percentage returns, one-step forecast of mean and volatility
        fit = arch_model(hist_port.values * 100, mean='Constant', vol='Garch', p=1, q=1).fit(disp='off')
        fc = fit.forecast(horizon=1)
        g_mu = fc.mean.iloc[-1, 0] / 100
        g_sd = np.sqrt(fc.variance.iloc[-1, 0]) / 100
        row['garch_sigma'] = g_sd

        for c in confidence_levels:
            z = stats.norm.ppf(c)
            row[f'historical_{c}'] = -np.quantile(hist_port, 1 - c)
            row[f'normal_{c}'] = z * sd - mu
            row[f'monte_carlo_{c}'] = -np.quantile(sims, 1 - c)
            row[f'garch_{c}'] = z * g_sd - g_mu
        rows.append(row)

    return pd.DataFrame(rows).set_index('date')


def kupiec_pof(breaches: np.ndarray, confidence: float) -> Tuple[float, float]:
    """
    Kupiec proportion-of-failures test.

    H0: the breach probability equals 1 - confidence.
    LR = -2 log[ p^x (1-p)^(n-x) / phat^x (1-phat)^(n-x) ] ~ chi2(1)

    Returns:
        Tuple[float, float]: LR statistic and p-value
    """
    b = np.asarray(breaches, dtype=int)
    n, x = len(b), b.sum()
    p, phat = 1 - confidence, x / n
    log_h0 = xlogy(x, p) + xlogy(n - x, 1 - p)
    log_h1 = xlogy(x, phat) + xlogy(n - x, 1 - phat)
    lr = -2 * (log_h0 - log_h1)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_independence(breaches: np.ndarray) -> Tuple[float, float]:
    """
    Christoffersen independence test.

    H0: a breach today is no more likely after a breach yesterday
    (breaches do not cluster). Compares a two-state Markov chain with
    pi01 = P(breach | no breach yesterday), pi11 = P(breach | breach yesterday)
    against a single breach probability. LR ~ chi2(1).

    Returns:
        Tuple[float, float]: LR statistic and p-value (nan if there are no breaches)
    """
    b = np.asarray(breaches, dtype=int)
    prev, curr = b[:-1], b[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))
    if n01 + n11 == 0:
        return float('nan'), float('nan')

    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    log_h0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_h1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = -2 * (log_h0 - log_h1)
    return float(lr), float(stats.chi2.sf(lr, 1))


def summarise_backtest(forecasts: pd.DataFrame,
                       confidence_levels: List[float] = (0.95, 0.99)) -> pd.DataFrame:
    """
    Breach counts and test p-values for every method and confidence level.

    A breach is a day whose loss is bigger than that day's VaR forecast.
    """
    rows = []
    n = len(forecasts)
    for c in confidence_levels:
        for m in METHODS:
            breaches = (-forecasts['return'] > forecasts[f'{m}_{c}']).values
            _, p_kupiec = kupiec_pof(breaches, c)
            _, p_ind = christoffersen_independence(breaches)
            rows.append({
                'method': m, 'confidence': c, 'days': n,
                'expected': round(n * (1 - c), 1), 'breaches': int(breaches.sum()),
                'breach_rate': breaches.mean(), 'avg_var': forecasts[f'{m}_{c}'].mean(),
                'kupiec_p': p_kupiec, 'independence_p': p_ind,
            })
    return pd.DataFrame(rows)
