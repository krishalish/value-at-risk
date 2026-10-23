import numpy as np
import pandas as pd
import pytest

from var_system.backtest import (christoffersen_independence, kupiec_pof,
                                 rolling_var_forecasts, summarise_backtest)


def test_kupiec_exact_rate_is_not_rejected():
    # 5 breaches in 100 days at 95% is exactly the expected rate: LR = 0, p = 1
    breaches = np.zeros(100, dtype=int)
    breaches[::20] = 1
    lr, p = kupiec_pof(breaches, 0.95)
    assert lr == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0)


def test_kupiec_too_many_breaches_is_rejected():
    breaches = np.zeros(250, dtype=int)
    breaches[:15] = 1  # 15 breaches when 2.5 are expected at 99%
    _, p = kupiec_pof(breaches, 0.99)
    assert p < 0.001


def test_kupiec_zero_breaches_is_finite():
    lr, p = kupiec_pof(np.zeros(250, dtype=int), 0.99)
    assert np.isfinite(lr) and 0 < p < 1


def test_christoffersen_detects_clustering():
    spread = np.zeros(250, dtype=int)
    spread[::25] = 1                 # 10 breaches, evenly spaced
    clustered = np.zeros(250, dtype=int)
    clustered[100:110] = 1           # the same 10 breaches, back to back
    assert christoffersen_independence(spread)[1] > 0.05
    assert christoffersen_independence(clustered)[1] < 0.001


def test_rolling_forecasts_are_out_of_sample():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range('2024-01-01', periods=320)
    returns = pd.DataFrame(rng.normal(0, 0.01, (320, 2)), index=idx, columns=['A', 'B'])
    f = rolling_var_forecasts(returns, {'A': 0.5, 'B': 0.5}, window=250,
                              start=str(idx[260].date()), end=str(idx[-1].date()), simulations=2000)
    assert len(f) == 60
    # the forecast for the first test day only uses the 250 days before it
    first = f.index[0]
    port = returns.mean(axis=1)
    prior = port.loc[:first].iloc[:-1].iloc[-250:]
    assert f.loc[first, 'historical_0.95'] == pytest.approx(-np.quantile(prior, 0.05))
    summary = summarise_backtest(f)
    assert set(summary['method']) == {'historical', 'normal', 'monte_carlo', 'garch'}
    assert (summary['breaches'] >= 0).all()
