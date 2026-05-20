"""
Data Agent — fetches and aligns price/return data for BTC and factors.

Supports:
    - Yahoo Finance live download (yfinance)
    - CSV file loading (for backtesting / offline use)
    - Weekly resampling
    - VIX special handling (level changes, not log returns)
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass


# Ticker mapping for factors
FACTOR_TICKERS = {
    "QQQ": "QQQ",      # Nasdaq 100 / tech risk
    "GLD": "GLD",      # Gold
    "TLT": "TLT",      # Long-term treasuries
    "UUP": "UUP",      # US Dollar index ETF
    "HYG": "HYG",      # High-yield corporate bonds
    "VIX": "^VIX",     # Volatility index
}

BTC_TICKER = "BTC-USD"


@dataclass
class MarketData:
    """Container for aligned market data."""
    prices: pd.DataFrame
    returns: pd.DataFrame
    btc_returns: pd.Series
    factor_returns: pd.DataFrame
    metadata: dict


class DataAgent:
    """
    Fetches and preprocesses market data for the BTC beta model.
    Supports Yahoo Finance download or CSV file loading.
    """

    def __init__(
        self,
        btc_ticker: str = BTC_TICKER,
        factor_tickers: Optional[dict] = None,
    ):
        self.btc_ticker = btc_ticker
        self.factor_tickers = factor_tickers or FACTOR_TICKERS

    def load_csv(
        self,
        filepath: str,
        factors: Optional[list[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        freq: str = "daily",
    ) -> MarketData:
        """Load prices from a CSV file (columns: Date, BTC, QQQ, ...)."""
        prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
        if start:
            prices = prices[prices.index >= start]
        if end:
            prices = prices[prices.index <= end]
        if freq == "weekly":
            prices = prices.resample("W-FRI").last().dropna()

        returns = np.log(prices / prices.shift(1)).dropna()
        btc_ret = returns["BTC"]
        factor_cols = [c for c in returns.columns if c != "BTC"]
        if factors:
            factor_cols = [c for c in factor_cols if c in factors]
        factor_ret = returns[factor_cols]

        metadata = {
            "start": str(returns.index[0].date()),
            "end": str(returns.index[-1].date()),
            "n_obs": len(returns),
            "factors": factor_cols,
            "freq": freq,
            "source": filepath,
        }
        print(f"  Loaded: {filepath}")
        print(f"  Data: {metadata['start']} → {metadata['end']}  ({metadata['n_obs']} obs)")

        return MarketData(
            prices=prices,
            returns=returns,
            btc_returns=btc_ret,
            factor_returns=factor_ret,
            metadata=metadata,
        )

    def fetch(
        self,
        factors: list[str] = ["QQQ"],
        start: str = "2020-01-01",
        end: Optional[str] = None,
        freq: str = "daily",
    ) -> MarketData:
        """
        Fetch BTC + factor prices via yfinance and compute log returns.

        Parameters
        ----------
        factors : list of factor names (keys from FACTOR_TICKERS)
        start / end : date range
        freq : 'daily' or 'weekly'
        """
        import yfinance as yf

        tickers_to_fetch = {self.btc_ticker: "BTC"}
        for f in factors:
            if f in self.factor_tickers:
                tickers_to_fetch[self.factor_tickers[f]] = f
            else:
                raise ValueError(f"Unknown factor: {f}. Available: {list(self.factor_tickers)}")

        ticker_list = list(tickers_to_fetch.keys())
        print(f"  Fetching: {', '.join(tickers_to_fetch.values())} ...")
        raw = yf.download(
            ticker_list,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        # Handle single vs multi ticker
        if len(ticker_list) == 1:
            prices = raw[["Close"]].copy()
            prices.columns = [list(tickers_to_fetch.values())[0]]
        else:
            prices = raw["Close"][ticker_list].copy()
            prices.columns = [tickers_to_fetch[t] for t in ticker_list]

        prices = prices.dropna()

        if freq == "weekly":
            prices = prices.resample("W-FRI").last().dropna()

        returns = np.log(prices / prices.shift(1)).dropna()

        # VIX special handling: level changes, not log returns
        if "VIX" in returns.columns:
            vix_prices = prices["VIX"]
            returns["VIX"] = vix_prices.diff() / vix_prices.shift(1)
            returns = returns.dropna()

        btc_ret = returns["BTC"]
        factor_cols = [c for c in returns.columns if c != "BTC"]
        factor_ret = returns[factor_cols]

        metadata = {
            "start": str(returns.index[0].date()),
            "end": str(returns.index[-1].date()),
            "n_obs": len(returns),
            "factors": factor_cols,
            "freq": freq,
        }

        print(f"  Data: {metadata['start']} → {metadata['end']}  ({metadata['n_obs']} obs)")
        return MarketData(
            prices=prices,
            returns=returns,
            btc_returns=btc_ret,
            factor_returns=factor_ret,
            metadata=metadata,
        )
