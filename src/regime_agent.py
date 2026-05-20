"""
Regime Agent — classifies BTC's current factor exposure into named regimes.

Regimes:
    - risk-on: BTC tracking equity/tech risk appetite
    - gold-like: BTC behaving as store-of-value / inflation hedge
    - liquidity-sensitive: BTC driven by dollar / rates / credit
    - stress: BTC correlating with volatility spikes
    - idiosyncratic: BTC decoupled from traditional factors
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


REGIME_RULES = {
    "risk-on": {
        "description": "BTC tracking equity/tech risk appetite",
        "condition": lambda b: b.get("QQQ", 0) > 0.5 and b.get("GLD", 0) < 0.3,
    },
    "gold-like": {
        "description": "BTC behaving like a store-of-value / inflation hedge",
        "condition": lambda b: b.get("GLD", 0) > 0.3 and b.get("QQQ", 0) < 0.3,
    },
    "liquidity-sensitive": {
        "description": "BTC driven by dollar / rates / credit conditions",
        "condition": lambda b: (
            abs(b.get("UUP", 0)) > 0.3 or
            abs(b.get("TLT", 0)) > 0.3 or
            abs(b.get("HYG", 0)) > 0.3
        ),
    },
    "stress": {
        "description": "BTC correlating with volatility — risk-off behavior",
        "condition": lambda b: b.get("VIX", 0) > 0.02 or (
            b.get("QQQ", 0) > 0.5 and b.get("VIX", 0) > 0.01
        ),
    },
    "idiosyncratic": {
        "description": "BTC decoupled from traditional factors",
        "condition": lambda b: all(abs(v) < 0.3 for v in b.values()),
    },
}


@dataclass
class RegimeSnapshot:
    """Current regime classification."""
    regime: str
    description: str
    confidence: float
    betas: dict
    date: pd.Timestamp


@dataclass
class RegimeHistory:
    """Full history of regime classifications."""
    regimes: pd.Series
    regime_changes: pd.DataFrame
    current: RegimeSnapshot


class RegimeAgent:
    """
    Classifies BTC's market behavior regime based on its factor betas.
    Uses the multi-factor Kalman beta estimates.
    """

    def __init__(self, rules: Optional[dict] = None, smoothing_window: int = 5):
        self.rules = rules or REGIME_RULES
        self.smoothing_window = smoothing_window

    def _classify_single(self, beta_dict: dict) -> tuple[str, str]:
        """Classify a single observation into a regime."""
        for regime_name, rule in self.rules.items():
            if rule["condition"](beta_dict):
                return regime_name, rule["description"]
        return "idiosyncratic", REGIME_RULES["idiosyncratic"]["description"]

    def classify(self, betas: pd.DataFrame) -> RegimeHistory:
        """
        Classify each date into a regime based on beta values.

        Parameters
        ----------
        betas : time-varying betas from Kalman filter (T x K)
        """
        betas_smooth = betas.rolling(self.smoothing_window, min_periods=1).mean()

        regime_labels = []
        for idx in betas_smooth.index:
            beta_dict = betas_smooth.loc[idx].to_dict()
            label, _ = self._classify_single(beta_dict)
            regime_labels.append(label)

        regime_series = pd.Series(regime_labels, index=betas_smooth.index, name="regime")

        changes = regime_series != regime_series.shift(1)
        change_dates = regime_series[changes].reset_index()
        change_dates.columns = ["date", "regime"]

        latest_betas = betas.iloc[-1].to_dict()
        current_label, current_desc = self._classify_single(
            betas_smooth.iloc[-1].to_dict()
        )

        lookback = min(20, len(regime_series))
        recent = regime_series.iloc[-lookback:]
        confidence = (recent == current_label).mean()

        current = RegimeSnapshot(
            regime=current_label, description=current_desc,
            confidence=confidence, betas=latest_betas, date=betas.index[-1],
        )

        return RegimeHistory(
            regimes=regime_series, regime_changes=change_dates, current=current,
        )
