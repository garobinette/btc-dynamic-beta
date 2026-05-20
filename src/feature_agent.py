"""
Feature Agent — transforms raw prices/returns into model features.

Computes:
    - Z-scores (rolling standardized returns)
    - Rolling volatility surfaces (short/long windows)
    - Vol ratio (regime indicator)
    - Cross-asset spreads (BTC-QQQ, HY proxy, real rate, dollar-gold)
    - Multi-horizon momentum
    - Rolling BTC-factor correlations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class FeatureSet:
    """Container for computed features."""
    returns: pd.DataFrame          # log returns
    z_scores: pd.DataFrame         # rolling z-scores of returns
    volatility: pd.DataFrame       # rolling realized vol (annualized)
    vol_ratio: pd.DataFrame        # short/long vol ratio (vol regime)
    spreads: pd.DataFrame          # cross-asset spread features
    momentum: pd.DataFrame         # multi-horizon momentum
    correlations: pd.DataFrame     # rolling BTC-factor correlations
    metadata: dict


class FeatureAgent:
    """
    Transforms raw market data into features for the state-space model
    and downstream agents (Risk, Signal, Narrative).
    """

    def __init__(
        self,
        z_window: int = 60,
        vol_short: int = 21,
        vol_long: int = 63,
        corr_window: int = 60,
        momentum_horizons: tuple = (5, 21, 63),
    ):
        self.z_window = z_window
        self.vol_short = vol_short
        self.vol_long = vol_long
        self.corr_window = corr_window
        self.momentum_horizons = momentum_horizons

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        btc_col: str = "BTC",
    ) -> FeatureSet:
        """Compute the full feature set from price and return data."""
        print("  Computing features:")

        z_scores = self._z_scores(returns)
        print(f"    z-scores ({self.z_window}d window)")

        vol_short = self._rolling_vol(returns, window=self.vol_short)
        vol_long = self._rolling_vol(returns, window=self.vol_long)
        vol_ratio = vol_short / vol_long
        print(f"    volatility ({self.vol_short}d / {self.vol_long}d)")

        spreads = self._compute_spreads(prices, returns)
        print(f"    spreads ({len(spreads.columns)} features)")

        momentum = self._compute_momentum(prices)
        print(f"    momentum ({len(self.momentum_horizons)} horizons)")

        correlations = self._rolling_correlations(returns, btc_col)
        print(f"    correlations ({self.corr_window}d window)")

        metadata = {
            "z_window": self.z_window,
            "vol_short": self.vol_short,
            "vol_long": self.vol_long,
            "corr_window": self.corr_window,
            "momentum_horizons": list(self.momentum_horizons),
            "n_features": (
                len(z_scores.columns) +
                len(vol_short.columns) +
                len(vol_ratio.columns) +
                len(spreads.columns) +
                len(momentum.columns) +
                len(correlations.columns)
            ),
        }

        return FeatureSet(
            returns=returns,
            z_scores=z_scores,
            volatility=vol_short,
            vol_ratio=vol_ratio,
            spreads=spreads,
            momentum=momentum,
            correlations=correlations,
            metadata=metadata,
        )

    def _z_scores(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Rolling z-score: (r - rolling_mean) / rolling_std."""
        mu = returns.rolling(self.z_window, min_periods=10).mean()
        sigma = returns.rolling(self.z_window, min_periods=10).std()
        z = (returns - mu) / sigma.replace(0, np.nan)
        z.columns = [f"{c}_z" for c in z.columns]
        return z

    def _rolling_vol(self, returns: pd.DataFrame, window: int) -> pd.DataFrame:
        """Annualized rolling realized volatility."""
        vol = returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252)
        vol.columns = [f"{c}_vol{window}d" for c in vol.columns]
        return vol

    def _compute_spreads(
        self, prices: pd.DataFrame, returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Cross-asset spread features:
        - BTC_QQQ_spread: excess crypto return over tech
        - HY_spread_proxy: HYG - TLT (credit spread movement)
        - Real_rate_proxy: TLT - GLD (real rate signal)
        - Dollar_gold_spread: UUP - GLD (risk appetite)
        """
        spreads = pd.DataFrame(index=returns.index)

        if "BTC" in returns.columns and "QQQ" in returns.columns:
            spreads["BTC_QQQ_spread"] = returns["BTC"] - returns["QQQ"]
        if "HYG" in returns.columns and "TLT" in returns.columns:
            spreads["HY_spread_proxy"] = returns["HYG"] - returns["TLT"]
        if "TLT" in returns.columns and "GLD" in returns.columns:
            spreads["real_rate_proxy"] = returns["TLT"] - returns["GLD"]
        if "UUP" in returns.columns and "GLD" in returns.columns:
            spreads["dollar_gold_spread"] = returns["UUP"] - returns["GLD"]

        return spreads

    def _compute_momentum(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Multi-horizon momentum: cumulative return over N days."""
        momentum = pd.DataFrame(index=prices.index)
        for h in self.momentum_horizons:
            returns_h = np.log(prices / prices.shift(h))
            for col in prices.columns:
                momentum[f"{col}_mom{h}d"] = returns_h[col]
        return momentum

    def _rolling_correlations(
        self, returns: pd.DataFrame, btc_col: str = "BTC"
    ) -> pd.DataFrame:
        """Rolling correlation of BTC with each factor."""
        if btc_col not in returns.columns:
            return pd.DataFrame(index=returns.index)

        corrs = pd.DataFrame(index=returns.index)
        factor_cols = [c for c in returns.columns if c != btc_col]
        for f in factor_cols:
            corrs[f"BTC_{f}_corr"] = (
                returns[btc_col]
                .rolling(self.corr_window, min_periods=15)
                .corr(returns[f])
            )
        return corrs

    def summary(self, features: FeatureSet) -> dict:
        """Latest snapshot of key features for reporting."""
        latest = {}
        for col in features.z_scores.columns:
            val = features.z_scores[col].dropna()
            if len(val) > 0:
                latest[col] = round(float(val.iloc[-1]), 2)

        for col in features.volatility.columns:
            val = features.volatility[col].dropna()
            if len(val) > 0:
                latest[col] = round(float(val.iloc[-1]), 3)

        for col in features.vol_ratio.columns:
            val = features.vol_ratio[col].dropna()
            if len(val) > 0:
                latest[col.replace("vol21d", "vol_ratio")] = round(float(val.iloc[-1]), 2)

        for col in features.correlations.columns:
            val = features.correlations[col].dropna()
            if len(val) > 0:
                latest[col] = round(float(val.iloc[-1]), 3)

        return latest
