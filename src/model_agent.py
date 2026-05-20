"""
Model Agent — state-space estimation, risk analysis, signal generation, and narrative.

Contains:
    - KalmanBetaFilter: dynamic beta via Kalman filter with EM learning
    - RollingOLSBeta / StaticOLSBeta: comparison baselines
    - RiskAgent: variance decomposition, VaR, drawdown, position sizing
    - SignalAgent: tactical views, composite scoring, hedging recs
    - NarrativeAgent: PM-style text report generation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
#  KALMAN FILTER ENGINE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class KalmanBetaResult:
    """Container for Kalman filter estimation results."""
    dates: pd.DatetimeIndex
    betas: pd.DataFrame          # time-varying betas (T x K)
    beta_std: pd.DataFrame       # std dev of beta estimates (T x K)
    alpha: pd.Series             # time-varying intercept
    residuals: pd.Series         # observation residuals
    residual_vol: pd.Series      # rolling residual volatility
    log_likelihood: float
    factor_names: list


class KalmanBetaFilter:
    """
    State-space model for dynamic regression coefficients.

    Model:
        y_t = alpha_t + beta_t' * x_t + eps_t,   eps_t ~ N(0, H)
        [alpha_t, beta_t]' = [alpha_{t-1}, beta_{t-1}]' + eta_t,  eta_t ~ N(0, Q)

    State vector: [alpha, beta_1, ..., beta_K].
    Transition: random walk (identity).
    EM learning refines Q (diagonal) and H iteratively.
    RTS smoother provides backward-smoothed estimates.
    """

    def __init__(
        self,
        q_scale: float = 1e-4,
        h_init: float = 1e-3,
        residual_vol_window: int = 21,
    ):
        self.q_scale = q_scale
        self.h_init = h_init
        self.residual_vol_window = residual_vol_window

    def fit(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        em_iterations: int = 10,
    ) -> KalmanBetaResult:
        """
        Run Kalman filter + RTS smoother with EM parameter learning.

        Parameters
        ----------
        y : BTC returns (dependent variable)
        X : factor returns (regressors)
        em_iterations : EM iterations to refine Q and H
        """
        y = y.dropna()
        X = X.loc[y.index].dropna()
        common = y.index.intersection(X.index)
        y = y.loc[common].values.astype(np.float64)
        X_vals = X.loc[common].values.astype(np.float64)
        dates = common
        factor_names = list(X.columns)

        T = len(y)
        K = X_vals.shape[1]
        state_dim = K + 1  # [alpha, beta_1, ..., beta_K]

        # OLS warm-start
        X_with_const = np.column_stack([np.ones(T), X_vals])
        try:
            ols_coefs = np.linalg.lstsq(X_with_const[:min(60, T)], y[:min(60, T)], rcond=None)[0]
        except np.linalg.LinAlgError:
            ols_coefs = np.zeros(state_dim)

        state = ols_coefs.copy()
        P = np.eye(state_dim) * 1.0

        Q = np.eye(state_dim) * self.q_scale
        H = self.h_init

        # Storage
        states_filt = np.zeros((T, state_dim))
        P_filt = np.zeros((T, state_dim, state_dim))
        states_pred = np.zeros((T, state_dim))
        P_pred = np.zeros((T, state_dim, state_dim))
        residuals = np.zeros(T)
        log_lik = 0.0

        for em_iter in range(max(em_iterations, 1)):
            state = ols_coefs.copy()
            P = np.eye(state_dim) * 1.0
            log_lik = 0.0

            # Forward pass (filter)
            for t in range(T):
                state_p = state.copy()
                P_p = P + Q

                z_t = np.concatenate([[1.0], X_vals[t]])
                y_pred = z_t @ state_p
                resid = y[t] - y_pred
                S = z_t @ P_p @ z_t + H

                K_gain = P_p @ z_t / S

                state = state_p + K_gain * resid
                P = P_p - np.outer(K_gain, K_gain) * S

                states_pred[t] = state_p
                P_pred[t] = P_p
                states_filt[t] = state
                P_filt[t] = P
                residuals[t] = resid

                log_lik += -0.5 * (np.log(2 * np.pi * S) + resid**2 / S)

            # Backward pass (RTS smoother)
            states_smooth = states_filt.copy()
            P_smooth = P_filt.copy()

            for t in range(T - 2, -1, -1):
                try:
                    J = P_filt[t] @ np.linalg.inv(P_pred[t + 1])
                except np.linalg.LinAlgError:
                    J = P_filt[t] @ np.linalg.pinv(P_pred[t + 1])

                states_smooth[t] = states_filt[t] + J @ (states_smooth[t + 1] - states_pred[t + 1])
                P_smooth[t] = P_filt[t] + J @ (P_smooth[t + 1] - P_pred[t + 1]) @ J.T

            # EM update of Q and H
            if em_iterations > 1:
                resid_smooth = np.zeros(T)
                for t in range(T):
                    z_t = np.concatenate([[1.0], X_vals[t]])
                    resid_smooth[t] = y[t] - z_t @ states_smooth[t]
                H = max(np.mean(resid_smooth**2), 1e-8)

                Q_diag = np.zeros(state_dim)
                for t in range(1, T):
                    diff = states_smooth[t] - states_smooth[t - 1]
                    Q_diag += diff**2 + np.diag(P_smooth[t]) + np.diag(P_smooth[t - 1])
                Q_diag /= (T - 1)
                Q_diag = np.clip(Q_diag, 1e-8, self.q_scale * 10)
                Q = np.diag(Q_diag)

        # Package results
        betas_df = pd.DataFrame(states_smooth[:, 1:], index=dates, columns=factor_names)
        beta_std_df = pd.DataFrame(
            np.sqrt(np.array([np.diag(P_smooth[t])[1:] for t in range(T)])),
            index=dates, columns=factor_names,
        )
        alpha_s = pd.Series(states_smooth[:, 0], index=dates, name="alpha")
        resid_s = pd.Series(residuals, index=dates, name="residual")
        resid_vol = resid_s.rolling(self.residual_vol_window).std()
        resid_vol.name = "residual_vol"

        return KalmanBetaResult(
            dates=dates, betas=betas_df, beta_std=beta_std_df,
            alpha=alpha_s, residuals=resid_s, residual_vol=resid_vol,
            log_likelihood=log_lik, factor_names=factor_names,
        )


class RollingOLSBeta:
    """Rolling-window OLS beta for comparison."""

    def __init__(self, window: int = 90):
        self.window = window

    def fit(self, y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
        common = y.dropna().index.intersection(X.dropna().index)
        y = y.loc[common]
        X = X.loc[common]

        results = {}
        for col in X.columns:
            betas = []
            for i in range(len(y)):
                if i < self.window - 1:
                    betas.append(np.nan)
                else:
                    y_w = y.iloc[i - self.window + 1 : i + 1].values
                    x_w = X[col].iloc[i - self.window + 1 : i + 1].values
                    X_mat = np.column_stack([np.ones(self.window), x_w])
                    try:
                        coef = np.linalg.lstsq(X_mat, y_w, rcond=None)[0]
                        betas.append(coef[1])
                    except:
                        betas.append(np.nan)
            results[col] = betas

        return pd.DataFrame(results, index=common)


class StaticOLSBeta:
    """Full-sample OLS beta for comparison."""

    def fit(self, y: pd.Series, X: pd.DataFrame) -> dict:
        common = y.dropna().index.intersection(X.dropna().index)
        y_vals = y.loc[common].values
        X_vals = X.loc[common].values
        X_mat = np.column_stack([np.ones(len(y_vals)), X_vals])

        coefs = np.linalg.lstsq(X_mat, y_vals, rcond=None)[0]
        residuals = y_vals - X_mat @ coefs
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_vals - y_vals.mean())**2)
        r_squared = 1 - ss_res / ss_tot

        return {
            "alpha": coefs[0],
            "betas": dict(zip(X.columns, coefs[1:])),
            "r_squared": r_squared,
            "residual_std": np.std(residuals),
            "n_obs": len(y_vals),
        }


# ═══════════════════════════════════════════════════════════════════
#  RISK AGENT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RiskReport:
    """Container for risk analysis results."""
    factor_var_contrib: dict
    systematic_pct: float
    idiosyncratic_pct: float
    var_95: float
    var_99: float
    cvar_95: float
    max_drawdown: float
    current_drawdown: float
    vol_ann: float
    vol_regime: str
    sizing: dict
    exposure_summary: dict
    date: str


class RiskAgent:
    """
    Evaluates BTC's risk profile using dynamic betas, returns, and features.
    Answers: "How much risk am I taking, and where is it coming from?"
    """

    def __init__(
        self,
        vol_window: int = 21,
        var_window: int = 252,
        risk_targets: tuple = (0.10, 0.15, 0.20, 0.25),
    ):
        self.vol_window = vol_window
        self.var_window = var_window
        self.risk_targets = risk_targets

    def analyze(
        self,
        btc_returns: pd.Series,
        factor_returns: pd.DataFrame,
        betas: pd.DataFrame,
        residuals: pd.Series,
        prices: Optional[pd.DataFrame] = None,
    ) -> RiskReport:
        """Run full risk analysis."""
        print("  Risk analysis:")

        latest_betas = betas.iloc[-1].to_dict()
        factor_names = list(latest_betas.keys())

        # Variance decomposition
        factor_cov = factor_returns.tail(self.var_window).cov().values
        beta_vec = np.array([latest_betas[f] for f in factor_names])

        systematic_var = beta_vec @ factor_cov @ beta_vec * 252
        idio_var = float(residuals.tail(self.vol_window).var()) * 252
        total_var = systematic_var + idio_var

        systematic_pct = systematic_var / total_var if total_var > 0 else 0
        idiosyncratic_pct = 1 - systematic_pct

        factor_var_contrib = {}
        for i, f in enumerate(factor_names):
            fvar = beta_vec[i] ** 2 * factor_cov[i, i] * 252
            factor_var_contrib[f] = float(fvar / total_var) if total_var > 0 else 0
        print(f"    Systematic: {systematic_pct:.1%} | Idiosyncratic: {idiosyncratic_pct:.1%}")

        # VaR / CVaR
        recent_returns = btc_returns.tail(self.var_window).dropna().values
        vol_ann = float(np.std(recent_returns) * np.sqrt(252))
        daily_vol = np.std(recent_returns)
        var_95 = float(1.645 * daily_vol)
        var_99 = float(2.326 * daily_vol)

        sorted_rets = np.sort(recent_returns)
        n5 = max(1, int(len(sorted_rets) * 0.05))
        cvar_95 = float(-np.mean(sorted_rets[:n5]))
        print(f"    VaR(95%): {var_95:.2%} | CVaR(95%): {cvar_95:.2%}")

        # Drawdown
        if prices is not None and "BTC" in prices.columns:
            btc_prices = prices["BTC"].dropna()
            running_max = btc_prices.cummax()
            drawdowns = (btc_prices - running_max) / running_max
            max_drawdown = float(drawdowns.min())
            current_drawdown = float(drawdowns.iloc[-1])
        else:
            cum_ret = np.cumsum(btc_returns.dropna().values)
            running_max = np.maximum.accumulate(cum_ret)
            dd = cum_ret - running_max
            max_drawdown = float(dd.min())
            current_drawdown = float(dd[-1])
        print(f"    Max DD: {max_drawdown:.1%} | Current DD: {current_drawdown:.1%}")

        # Vol regime
        vol_short = float(btc_returns.tail(21).std() * np.sqrt(252))
        vol_long = float(btc_returns.tail(63).std() * np.sqrt(252))
        vol_ratio = vol_short / vol_long if vol_long > 0 else 1

        if vol_ratio < 0.8:
            vol_regime = "low"
        elif vol_ratio < 1.1:
            vol_regime = "normal"
        elif vol_ratio < 1.5:
            vol_regime = "elevated"
        else:
            vol_regime = "high"
        print(f"    Vol regime: {vol_regime} (ratio: {vol_ratio:.2f})")

        # Position sizing
        sizing = {}
        for target in self.risk_targets:
            weight = target / vol_ann if vol_ann > 0 else 0
            sizing[f"{target:.0%}_vol_target"] = {
                "weight": round(float(min(weight, 1.0)), 4),
                "notional_per_1M": round(float(min(weight, 1.0) * 1_000_000), 0),
            }

        # Exposure summary per $1M notional
        exposure_summary = {}
        for f in factor_names:
            beta_f = latest_betas[f]
            exposure_summary[f] = {
                "beta": round(beta_f, 4),
                "equivalent_notional": round(beta_f * 1_000_000, 0),
                "direction": "long" if beta_f > 0 else "short",
            }

        date_str = str(betas.index[-1].date()) if hasattr(betas.index[-1], 'date') else str(betas.index[-1])

        return RiskReport(
            factor_var_contrib=factor_var_contrib,
            systematic_pct=float(systematic_pct),
            idiosyncratic_pct=float(idiosyncratic_pct),
            var_95=var_95, var_99=var_99, cvar_95=cvar_95,
            max_drawdown=max_drawdown, current_drawdown=current_drawdown,
            vol_ann=vol_ann, vol_regime=vol_regime,
            sizing=sizing, exposure_summary=exposure_summary, date=date_str,
        )

    def format_exposure_table(self, report: RiskReport) -> str:
        """Format exposure summary as a readable table."""
        lines = [
            "FACTOR EXPOSURE (per $1M BTC notional)",
            "-" * 55,
            f"  {'Factor':<8} {'Beta':>8} {'Direction':>10} {'Equiv $':>12}",
            f"  {'─'*8} {'─'*8} {'─'*10} {'─'*12}",
        ]
        for f, exp in report.exposure_summary.items():
            lines.append(
                f"  {f:<8} {exp['beta']:>8.3f} {exp['direction']:>10} "
                f"${exp['equivalent_notional']:>10,.0f}"
            )
        lines.append("")
        lines.append(f"  Systematic risk: {report.systematic_pct:.1%}")
        lines.append(f"  Idiosyncratic:   {report.idiosyncratic_pct:.1%}")
        lines.append(f"  Vol regime:      {report.vol_regime}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  SIGNAL AGENT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    """A single actionable signal."""
    name: str
    direction: str          # 'bullish', 'bearish', 'neutral'
    conviction: float       # 0 to 1
    rationale: str
    category: str           # 'beta_momentum', 'regime', 'risk', 'hedging'


@dataclass
class SignalReport:
    """Collection of signals with overall view."""
    signals: list[Signal]
    composite_score: float
    composite_label: str
    hedging_recommendations: list[str]
    date: str


class SignalAgent:
    """
    Generates tactical views by analyzing beta momentum, regime context,
    risk conditions, and cross-asset divergences.
    """

    def __init__(
        self,
        beta_momentum_window: int = 20,
        regime_stability_window: int = 10,
    ):
        self.beta_momentum_window = beta_momentum_window
        self.regime_stability_window = regime_stability_window

    def generate(
        self,
        betas: pd.DataFrame,
        regimes: pd.Series,
        risk_report: RiskReport,
        features=None,
    ) -> SignalReport:
        """Generate signals from model outputs."""
        print("  Generating signals:")
        signals = []

        signals.extend(self._beta_momentum(betas))
        signals.extend(self._regime_signals(regimes))
        signals.extend(self._risk_signals(risk_report))
        if features is not None:
            signals.extend(self._feature_signals(features))

        # Composite score
        if signals:
            direction_map = {"bullish": 1, "bearish": -1, "neutral": 0}
            weighted = sum(direction_map[s.direction] * s.conviction for s in signals)
            composite = max(-1, min(1, weighted / len(signals)))
        else:
            composite = 0.0

        if composite > 0.5:
            label = "strong buy"
        elif composite > 0.2:
            label = "buy"
        elif composite > -0.2:
            label = "neutral"
        elif composite > -0.5:
            label = "sell"
        else:
            label = "strong sell"

        hedging = self._hedging_recs(betas, risk_report, regimes)
        date_str = str(betas.index[-1].date()) if hasattr(betas.index[-1], 'date') else str(betas.index[-1])

        print(f"    Composite: {composite:+.2f} ({label})")
        print(f"    Signals: {len(signals)} total")

        return SignalReport(
            signals=signals, composite_score=float(composite),
            composite_label=label, hedging_recommendations=hedging, date=date_str,
        )

    def _beta_momentum(self, betas: pd.DataFrame) -> list[Signal]:
        """Detect trending betas (z-scored changes)."""
        signals = []
        w = self.beta_momentum_window
        if len(betas) < w + 5:
            return signals

        for col in betas.columns:
            recent = betas[col].iloc[-w:]
            prior = betas[col].iloc[-2 * w : -w] if len(betas) >= 2 * w else betas[col].iloc[:w]

            delta = recent.mean() - prior.mean()
            recent_std = recent.std()
            z_move = delta / recent_std if recent_std > 0 else 0

            if abs(z_move) > 1.5:
                direction = "bullish" if delta > 0 else "bearish"
                if col == "GLD" and delta > 0:
                    direction = "bearish"
                elif col == "TLT" and delta > 0:
                    direction = "bearish"
                elif col == "UUP" and delta > 0:
                    direction = "bearish"

                conviction = min(abs(z_move) / 3, 1.0)
                signals.append(Signal(
                    name=f"{col}_beta_momentum", direction=direction,
                    conviction=round(conviction, 2),
                    rationale=f"{col} beta moved {delta:+.3f} over {w}d (z={z_move:+.1f})",
                    category="beta_momentum",
                ))
        return signals

    def _regime_signals(self, regimes: pd.Series) -> list[Signal]:
        """Regime stability and type signals."""
        signals = []
        w = self.regime_stability_window
        recent = regimes.iloc[-w:] if len(regimes) >= w else regimes
        current = recent.iloc[-1]
        n_changes = (recent != recent.shift(1)).sum() - 1

        if n_changes == 0:
            signals.append(Signal("regime_stable", "neutral", 0.3,
                f"Regime ({current}) stable for {w}+ periods", "regime"))
        elif n_changes >= 3:
            signals.append(Signal("regime_unstable", "bearish", 0.6,
                f"{n_changes} regime changes in last {w} periods — unstable", "regime"))

        if current == "risk-on":
            signals.append(Signal("risk_on_regime", "bullish", 0.4,
                "Risk-on regime: BTC tracking equity risk appetite", "regime"))
        elif current == "stress":
            signals.append(Signal("stress_regime", "bearish", 0.7,
                "Stress regime: BTC correlating with volatility spikes", "regime"))
        elif current == "idiosyncratic":
            signals.append(Signal("idiosyncratic_regime", "neutral", 0.2,
                "Idiosyncratic regime: BTC decoupled from macro factors", "regime"))

        return signals

    def _risk_signals(self, risk_report: RiskReport) -> list[Signal]:
        """Signals from risk conditions."""
        signals = []

        if risk_report.vol_regime == "high":
            signals.append(Signal("high_vol", "bearish", 0.6,
                f"Vol regime is HIGH (ann vol: {risk_report.vol_ann:.1%})", "risk"))
        elif risk_report.vol_regime == "low":
            signals.append(Signal("low_vol", "bullish", 0.3,
                "Vol regime is LOW — potential for expansion", "risk"))

        if risk_report.current_drawdown < -0.20:
            signals.append(Signal("deep_drawdown", "bearish", 0.5,
                f"Current drawdown: {risk_report.current_drawdown:.1%}", "risk"))

        if risk_report.idiosyncratic_pct > 0.80:
            signals.append(Signal("high_idio_risk", "neutral", 0.4,
                f"Idiosyncratic risk is {risk_report.idiosyncratic_pct:.0%} — macro hedges less effective", "risk"))

        return signals

    def _feature_signals(self, features) -> list[Signal]:
        """Signals from z-scores (if FeatureSet provided)."""
        signals = []
        btc_z_col = "BTC_z"
        if btc_z_col in features.z_scores.columns:
            z = features.z_scores[btc_z_col].dropna()
            if len(z) > 0:
                latest_z = float(z.iloc[-1])
                if latest_z > 2.0:
                    signals.append(Signal("btc_z_score_high", "bearish",
                        min(abs(latest_z) / 4, 0.7),
                        f"BTC z-score at {latest_z:+.1f} — overbought", "feature"))
                elif latest_z < -2.0:
                    signals.append(Signal("btc_z_score_low", "bullish",
                        min(abs(latest_z) / 4, 0.7),
                        f"BTC z-score at {latest_z:+.1f} — oversold", "feature"))
        return signals

    def _hedging_recs(self, betas, risk_report, regimes) -> list[str]:
        """Generate hedging recommendations based on current exposures."""
        recs = []
        latest = betas.iloc[-1].to_dict()

        if latest.get("QQQ", 0) > 0.5:
            notional = abs(latest["QQQ"]) * 100
            recs.append(f"QQQ exposure: Short ~${notional:.0f}k QQQ per $100k BTC "
                        f"to neutralize equity beta ({latest['QQQ']:.2f})")
        if latest.get("GLD", 0) > 0.3:
            recs.append(f"Gold overlap: BTC provides {latest['GLD']:.2f}x GLD exposure — "
                        f"consider reducing standalone gold allocation")
        if latest.get("TLT", 0) > 0.2:
            recs.append(f"Duration: BTC carries {latest['TLT']:.2f}x TLT beta — "
                        f"factor into portfolio duration budget")
        if abs(latest.get("UUP", 0)) > 0.2:
            dir_label = "long" if latest["UUP"] > 0 else "short"
            recs.append(f"Dollar: BTC is effectively {dir_label} USD (UUP beta: {latest['UUP']:.2f})")
        if risk_report.idiosyncratic_pct > 0.70:
            recs.append(f"Warning: {risk_report.idiosyncratic_pct:.0%} of risk is idiosyncratic — "
                        f"traditional factor hedges cover only {risk_report.systematic_pct:.0%} of variance")
        if not recs:
            recs.append("No significant factor exposures requiring hedging action")

        return recs

    def format_signal_table(self, report: SignalReport) -> str:
        """Format signals as a readable table."""
        lines = [
            f"SIGNAL DASHBOARD — {report.date}",
            f"Composite: {report.composite_score:+.2f} ({report.composite_label.upper()})",
            "-" * 65,
            f"  {'Signal':<30} {'Dir':>8} {'Conv':>6} {'Category':>15}",
            f"  {'─'*30} {'─'*8} {'─'*6} {'─'*15}",
        ]
        for s in report.signals:
            lines.append(f"  {s.name:<30} {s.direction:>8} {s.conviction:>6.2f} {s.category:>15}")
        lines.append("")
        lines.append("HEDGING RECOMMENDATIONS:")
        for i, rec in enumerate(report.hedging_recommendations, 1):
            lines.append(f"  {i}. {rec}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  NARRATIVE AGENT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class NarrativeReport:
    """Structured narrative output."""
    title: str
    date: str
    regime_summary: str
    beta_commentary: str
    risk_commentary: str
    outlook: str
    full_text: str


class NarrativeAgent:
    """
    Generates human-readable PM-style narrative from model outputs.
    Rule-based for now; can be swapped with LLM agent later.
    """

    def generate(
        self,
        kalman_result: KalmanBetaResult,
        regime_history,
        static_ols: dict,
        metadata: dict,
        risk_report: Optional[RiskReport] = None,
        signal_report: Optional[SignalReport] = None,
    ) -> NarrativeReport:
        current = regime_history.current
        latest_betas = kalman_result.betas.iloc[-1]
        latest_std = kalman_result.beta_std.iloc[-1]
        resid_vol = kalman_result.residual_vol.iloc[-1]

        date_str = str(current.date.date())

        # Regime
        regime_str = (
            f"BTC is currently in a **{current.regime.upper()}** regime "
            f"({current.description}). "
            f"Regime confidence: {current.confidence:.0%} "
            f"(based on last 20 trading days)."
        )
        recent_changes = regime_history.regime_changes[
            regime_history.regime_changes["date"] > (current.date - pd.Timedelta(days=60))
        ]
        if len(recent_changes) > 2:
            regime_str += f" Note: {len(recent_changes)} regime changes in the last 60 days — elevated instability."

        # Betas
        beta_lines = []
        for factor in kalman_result.factor_names:
            b = latest_betas[factor]
            s = latest_std[factor]
            static_b = static_ols["betas"].get(factor, 0)
            delta = b - static_b
            direction = "above" if delta > 0 else "below"
            magnitude = "significantly " if abs(delta) > 0.15 else ""
            beta_lines.append(
                f"  • {factor}: β = {b:.3f} (±{s:.3f}), "
                f"{magnitude}{direction} full-sample OLS ({static_b:.3f})"
            )
        beta_str = "Current factor exposures (Kalman-filtered):\n" + "\n".join(beta_lines)

        # Risk
        ann_resid_vol = resid_vol * np.sqrt(252) if not np.isnan(resid_vol) else 0
        risk_str = f"Residual (idiosyncratic) volatility: {ann_resid_vol:.1%} annualized. "
        if ann_resid_vol > 0.5:
            risk_str += "Elevated — crypto-specific drivers dominate over macro factors."
        elif ann_resid_vol > 0.3:
            risk_str += "Moderate — mixed macro/crypto signal."
        else:
            risk_str += "Low — BTC well-explained by macro factors currently."

        r2 = static_ols.get("r_squared", 0)
        risk_str += f"\nFull-sample R² = {r2:.3f}."

        if risk_report is not None:
            risk_str += (
                f"\n\nRisk decomposition: {risk_report.systematic_pct:.0%} systematic / "
                f"{risk_report.idiosyncratic_pct:.0%} idiosyncratic."
                f"\nVaR(95%): {risk_report.var_95:.2%} daily | "
                f"CVaR(95%): {risk_report.cvar_95:.2%} daily"
                f"\nVol regime: {risk_report.vol_regime.upper()} "
                f"(ann vol: {risk_report.vol_ann:.1%})"
                f"\nMax drawdown: {risk_report.max_drawdown:.1%} | "
                f"Current drawdown: {risk_report.current_drawdown:.1%}"
            )

        # Signals
        signal_str = ""
        if signal_report is not None:
            signal_str = (
                f"Composite signal: {signal_report.composite_score:+.2f} "
                f"({signal_report.composite_label.upper()})"
            )
            if signal_report.signals:
                top_signals = sorted(signal_report.signals, key=lambda s: s.conviction, reverse=True)[:3]
                signal_str += "\nTop signals:"
                for s in top_signals:
                    signal_str += f"\n  • {s.name}: {s.direction} ({s.conviction:.0%}) — {s.rationale}"
            if signal_report.hedging_recommendations:
                signal_str += "\n\nHedging recommendations:"
                for i, rec in enumerate(signal_report.hedging_recommendations, 1):
                    signal_str += f"\n  {i}. {rec}"

        # Outlook
        outlook_map = {
            "risk-on": "BTC is tracking equity risk appetite. Watch for QQQ momentum shifts "
                       "and Fed commentary as potential catalysts for beta regime change.",
            "gold-like": "BTC is trading like a store-of-value asset. Monitor real yields "
                         "and gold price action for confirmation or divergence.",
            "stress": "BTC in stress/risk-off mode. Correlation with VIX elevated. "
                      "Consider reducing gross exposure until regime stabilizes.",
            "liquidity-sensitive": "BTC sensitive to dollar/rates/credit. Watch DXY, yield curve, "
                                   "and credit spreads for directional cues.",
        }
        outlook_str = outlook_map.get(
            current.regime,
            "BTC is decoupled from macro factors — idiosyncratic drivers "
            "(on-chain, regulatory, flows) are dominant. "
            "Standard factor hedges may be less effective."
        )

        # Full text
        title = f"BTC Dynamic Beta Report — {date_str}"
        sections = [
            f"{'='*60}", title, f"{'='*60}", "",
            "REGIME", "------", regime_str, "",
            "FACTOR EXPOSURES", "----------------", beta_str, "",
            "RISK", "----", risk_str,
        ]
        if signal_str:
            sections += ["", "SIGNALS", "-------", signal_str]
        sections += [
            "", "OUTLOOK", "-------", outlook_str, "",
            f"{'='*60}",
            f"Data: {metadata.get('start', '?')} → {metadata.get('end', '?')}  |  {metadata.get('n_obs', '?')} observations",
            f"{'='*60}",
        ]

        return NarrativeReport(
            title=title, date=date_str, regime_summary=regime_str,
            beta_commentary=beta_str, risk_commentary=risk_str,
            outlook=outlook_str, full_text="\n".join(sections),
        )
