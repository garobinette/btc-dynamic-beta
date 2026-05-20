#!/usr/bin/env python3
"""
run_pipeline.py — Execute the BTC Dynamic Beta pipeline.

Usage:
    python run_pipeline.py                          # Single factor (QQQ)
    python run_pipeline.py --multi                  # Multi-factor
    python run_pipeline.py --full                   # Full pipeline (all agents)
    python run_pipeline.py --start 2022-01-01       # Custom date range
    python run_pipeline.py --full --export ../outputs  # Export all results
"""

import sys
import os
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Add src dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_agent import DataAgent
from feature_agent import FeatureAgent
from model_agent import (
    KalmanBetaFilter, RollingOLSBeta, StaticOLSBeta,
    RiskAgent, SignalAgent, NarrativeAgent,
)
from regime_agent import RegimeAgent


# ── Default paths (relative to project root) ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def _find_csv(data_dir: str) -> str | None:
    """Look for a prices CSV in the data directory."""
    for name in ("prices.csv", "btc_prices.csv", "market_data.csv"):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    # Fall back to any CSV
    csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv")] if os.path.isdir(data_dir) else []
    return os.path.join(data_dir, csvs[0]) if csvs else None


def run_single_factor(start="2020-01-01", end=None, export_dir=None):
    """Phase 1: BTC vs QQQ dynamic beta."""
    print("\n" + "="*60)
    print("PHASE 1: BTC vs QQQ — Dynamic Beta")
    print("="*60)

    print("\n[Data Agent]")
    data_agent = DataAgent()
    csv_path = _find_csv(DATA_DIR)
    if csv_path:
        data = data_agent.load_csv(csv_path, factors=["QQQ"], start=start, end=end)
    else:
        data = data_agent.fetch(factors=["QQQ"], start=start, end=end)

    print("\n[Model Agent — Kalman Filter]")
    kf = KalmanBetaFilter(q_scale=1e-4)
    result = kf.fit(data.btc_returns, data.factor_returns, em_iterations=10)
    print(f"  Log-likelihood: {result.log_likelihood:.2f}")
    print(f"  Latest beta (QQQ): {result.betas['QQQ'].iloc[-1]:.4f}")

    print("\n[Comparison — 90-day Rolling OLS]")
    rolling = RollingOLSBeta(window=90)
    rolling_betas = rolling.fit(data.btc_returns, data.factor_returns)
    print(f"  Latest rolling beta (QQQ): {rolling_betas['QQQ'].iloc[-1]:.4f}")

    print("\n[Comparison — Static OLS]")
    static = StaticOLSBeta()
    static_result = static.fit(data.btc_returns, data.factor_returns)
    print(f"  Static beta (QQQ): {static_result['betas']['QQQ']:.4f}")
    print(f"  R²: {static_result['r_squared']:.4f}")

    print("\n" + "-"*50)
    print("BETA COMPARISON SUMMARY")
    print("-"*50)
    print(f"  {'Method':<25} {'QQQ Beta':>10}")
    print(f"  {'─'*25} {'─'*10}")
    print(f"  {'Kalman (latest)':<25} {result.betas['QQQ'].iloc[-1]:>10.4f}")
    print(f"  {'90d Rolling OLS':<25} {rolling_betas['QQQ'].iloc[-1]:>10.4f}")
    print(f"  {'Static OLS':<25} {static_result['betas']['QQQ']:>10.4f}")
    print(f"\n  Residual vol (21d, ann): {result.residual_vol.iloc[-1] * np.sqrt(252):.1%}")

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        comparison = pd.DataFrame({
            "kalman_beta": result.betas["QQQ"],
            "rolling_90d_beta": rolling_betas["QQQ"],
            "static_ols_beta": static_result["betas"]["QQQ"],
            "kalman_beta_std": result.beta_std["QQQ"],
            "residual": result.residuals,
            "residual_vol_21d": result.residual_vol,
        })
        comparison.to_csv(f"{export_dir}/phase1_single_factor.csv")
        print(f"\n  Exported → {export_dir}/phase1_single_factor.csv")

    return result, rolling_betas, static_result, data


def run_multi_factor(start="2020-01-01", end=None, export_dir=None):
    """Phase 2+3: Multi-factor model with regime classification."""
    print("\n" + "="*60)
    print("PHASE 2+3: Multi-Factor Model + Regime Classification")
    print("="*60)

    print("\n[Data Agent]")
    data_agent = DataAgent()
    factors = ["QQQ", "GLD", "TLT", "UUP", "HYG"]
    csv_path = _find_csv(DATA_DIR)
    if csv_path:
        data = data_agent.load_csv(csv_path, factors=factors, start=start, end=end)
    else:
        data = data_agent.fetch(factors=factors, start=start, end=end)

    print("\n[Model Agent — Kalman Filter (multi-factor)]")
    kf = KalmanBetaFilter(q_scale=5e-5)
    result = kf.fit(data.btc_returns, data.factor_returns, em_iterations=15)
    print(f"  Log-likelihood: {result.log_likelihood:.2f}")
    for f in result.factor_names:
        print(f"    {f:<6}: {result.betas[f].iloc[-1]:>8.4f} (±{result.beta_std[f].iloc[-1]:.4f})")

    print("\n[Comparison — Static OLS]")
    static = StaticOLSBeta()
    static_result = static.fit(data.btc_returns, data.factor_returns)
    print(f"  R²: {static_result['r_squared']:.4f}")

    print("\n[Regime Agent]")
    regime_agent = RegimeAgent(smoothing_window=10)
    regime_history = regime_agent.classify(result.betas)
    current = regime_history.current
    print(f"  Current regime: {current.regime.upper()}")
    print(f"  Confidence: {current.confidence:.0%}")

    print("\n[Narrative Agent]")
    narrative_agent = NarrativeAgent()
    report = narrative_agent.generate(result, regime_history, static_result, data.metadata)
    print(report.full_text)

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        result.betas.to_csv(f"{export_dir}/kalman_betas_multifactor.csv")
        regime_history.regimes.to_csv(f"{export_dir}/regimes.csv")
        with open(f"{export_dir}/narrative_report.txt", "w") as f:
            f.write(report.full_text)
        print(f"\n  Exported → {export_dir}/")

    return result, regime_history, report, data


def run_full_pipeline(start="2020-01-01", end=None, export_dir=None):
    """
    Full pipeline: Data → Feature → Model → Regime → Risk → Signal → Narrative
    """
    print("\n" + "="*60)
    print("FULL PIPELINE: All Agents Active")
    print("="*60)

    factors = ["QQQ", "GLD", "TLT", "UUP", "HYG"]

    # 1. DATA AGENT
    print("\n[1/7 — Data Agent]")
    data_agent = DataAgent()
    csv_path = _find_csv(DATA_DIR)
    if csv_path:
        data = data_agent.load_csv(csv_path, factors=factors, start=start, end=end)
    else:
        data = data_agent.fetch(factors=factors, start=start, end=end)

    # 2. FEATURE AGENT
    print("\n[2/7 — Feature Agent]")
    feature_agent = FeatureAgent(z_window=60, vol_short=21, vol_long=63)
    features = feature_agent.compute(data.prices, data.returns)
    feature_summary = feature_agent.summary(features)
    print(f"    Total features: {features.metadata['n_features']}")

    # 3. STATE-SPACE MODEL
    print("\n[3/7 — State-Space Model (Kalman Filter)]")
    kf = KalmanBetaFilter(q_scale=5e-5)
    kalman_result = kf.fit(data.btc_returns, data.factor_returns, em_iterations=15)
    print(f"  Log-likelihood: {kalman_result.log_likelihood:.2f}")
    for f in kalman_result.factor_names:
        print(f"    {f:<6}: β = {kalman_result.betas[f].iloc[-1]:>8.4f}")

    static = StaticOLSBeta()
    static_result = static.fit(data.btc_returns, data.factor_returns)

    # 4. REGIME AGENT
    print("\n[4/7 — Regime Agent]")
    regime_agent = RegimeAgent(smoothing_window=10)
    regime_history = regime_agent.classify(kalman_result.betas)
    print(f"  Current regime: {regime_history.current.regime.upper()}")
    print(f"  Confidence: {regime_history.current.confidence:.0%}")

    # 5. RISK AGENT
    print("\n[5/7 — Risk Agent]")
    risk_agent = RiskAgent()
    risk_report = risk_agent.analyze(
        btc_returns=data.btc_returns,
        factor_returns=data.factor_returns,
        betas=kalman_result.betas,
        residuals=kalman_result.residuals,
        prices=data.prices,
    )
    print(risk_agent.format_exposure_table(risk_report))

    # 6. SIGNAL AGENT
    print("\n[6/7 — Signal Agent]")
    signal_agent = SignalAgent()
    signal_report = signal_agent.generate(
        betas=kalman_result.betas,
        regimes=regime_history.regimes,
        risk_report=risk_report,
        features=features,
    )
    print(signal_agent.format_signal_table(signal_report))

    # 7. NARRATIVE AGENT
    print("\n[7/7 — Narrative Agent]")
    narrative_agent = NarrativeAgent()
    report = narrative_agent.generate(
        kalman_result, regime_history, static_result, data.metadata,
        risk_report=risk_report, signal_report=signal_report,
    )
    print(report.full_text)

    # EXPORT
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

        kalman_result.betas.to_csv(f"{export_dir}/kalman_betas.csv")
        kalman_result.beta_std.to_csv(f"{export_dir}/kalman_beta_std.csv")
        regime_history.regimes.to_csv(f"{export_dir}/regimes.csv")
        features.z_scores.to_csv(f"{export_dir}/z_scores.csv")
        features.volatility.to_csv(f"{export_dir}/volatility.csv")
        features.correlations.to_csv(f"{export_dir}/correlations.csv")
        features.spreads.to_csv(f"{export_dir}/spreads.csv")

        with open(f"{export_dir}/narrative_report.txt", "w") as f:
            f.write(report.full_text)

        summary = {
            "date": report.date,
            "regime": regime_history.current.regime,
            "regime_confidence": regime_history.current.confidence,
            "betas": {f: round(float(kalman_result.betas[f].iloc[-1]), 4)
                      for f in kalman_result.factor_names},
            "risk": {
                "systematic_pct": round(risk_report.systematic_pct, 4),
                "idiosyncratic_pct": round(risk_report.idiosyncratic_pct, 4),
                "var_95": round(risk_report.var_95, 4),
                "cvar_95": round(risk_report.cvar_95, 4),
                "vol_ann": round(risk_report.vol_ann, 4),
                "vol_regime": risk_report.vol_regime,
                "max_drawdown": round(risk_report.max_drawdown, 4),
                "current_drawdown": round(risk_report.current_drawdown, 4),
            },
            "signal": {
                "composite_score": round(signal_report.composite_score, 4),
                "composite_label": signal_report.composite_label,
                "signals": [
                    {"name": s.name, "direction": s.direction,
                     "conviction": s.conviction, "category": s.category}
                    for s in signal_report.signals
                ],
                "hedging": signal_report.hedging_recommendations,
            },
            "features": feature_summary,
            "static_r_squared": round(static_result["r_squared"], 4),
            "exposure_per_1M": risk_report.exposure_summary,
            "sizing": risk_report.sizing,
        }
        with open(f"{export_dir}/full_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  Exported full pipeline → {export_dir}/")

    return {
        "data": data,
        "features": features,
        "kalman_result": kalman_result,
        "static_result": static_result,
        "regime_history": regime_history,
        "risk_report": risk_report,
        "signal_report": signal_report,
        "narrative": report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC Dynamic Beta Pipeline")
    parser.add_argument("--multi", action="store_true", help="Run multi-factor model")
    parser.add_argument("--full", action="store_true", help="Run full pipeline (all agents)")
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date")
    parser.add_argument("--export", default=None, help="Export directory (default: ../outputs)")
    args = parser.parse_args()

    export = args.export or OUTPUT_DIR

    if args.full:
        run_full_pipeline(start=args.start, end=args.end, export_dir=export)
    elif args.multi:
        run_multi_factor(start=args.start, end=args.end, export_dir=export)
    else:
        run_single_factor(start=args.start, end=args.end, export_dir=export)
