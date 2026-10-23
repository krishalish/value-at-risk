"""Download and cache daily stock prices from Yahoo Finance."""
import warnings
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']


def load_prices(tickers: List[str] = TICKERS, start: str = '2019-01-01',
                end: str = '2026-01-01', cache_dir: str = 'data',
                refresh: bool = False) -> pd.DataFrame:
    """
    Daily adjusted closing prices, cached to CSV so later runs are offline and repeatable.

    Args:
        tickers (List[str]): Stock tickers
        start (str): First date (inclusive), YYYY-MM-DD
        end (str): Last date (exclusive), YYYY-MM-DD
        cache_dir (str): Folder for the cached CSV
        refresh (bool): Re-download even if a cached file exists

    Returns:
        pd.DataFrame: Prices, one column per ticker
    """
    cache = Path(cache_dir) / f"prices_{'-'.join(tickers)}_{start}_{end}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # yfinance emits pandas deprecation noise
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("No data returned from Yahoo Finance (check your connection)")
    prices = raw['Close'][tickers].dropna()

    cache.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache)
    return prices


def load_returns(tickers: List[str] = TICKERS, start: str = '2019-01-01',
                 end: str = '2026-01-01', cache_dir: str = 'data',
                 refresh: bool = False) -> pd.DataFrame:
    """
    Daily simple returns, P_t / P_{t-1} - 1.

    Simple (not log) returns are used because a portfolio's simple return is
    exactly the weighted sum of its assets' simple returns.
    """
    prices = load_prices(tickers, start, end, cache_dir, refresh)
    return prices.pct_change().dropna()
