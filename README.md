# BTC Dynamic Beta Framework

A state-space model framework for estimating time-varying BTC factor betas using Kalman filtering with an agent-based architecture.

## Architecture

```
Data Agent → Feature Agent → State-Space Model → ┬─ Regime Agent ─┐
                                                  ├─ Risk Agent   ─┤→ Narrative Agent → PM Dashboard / PDF
                                                  └─ Signal Agent ─┘
```

## Project Structure

```
btc-dynamic-beta/
├── data/                  # Price CSVs (BTC + factors)
├── outputs/               # Pipeline exports (betas, regimes, reports, PDFs)
├── notebooks/             # Jupyter notebooks for exploration
├── src/
│   ├── data_agent.py      # Market data fetch (yfinance) + CSV loading
│   ├── feature_agent.py   # Z-scores, vol surfaces, spreads, momentum, correlations
│   ├── model_agent.py     # Kalman filter, OLS baselines, Risk, Signal, Narrative agents
│   ├── regime_agent.py    # 5-regime classifier (risk-on, gold-like, liquidity, stress, idiosyncratic)
│   └── run_pipeline.py    # CLI pipeline orchestrator
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Single factor (BTC vs QQQ)
cd src && python run_pipeline.py

# Multi-factor with regime classification
python run_pipeline.py --multi

# Full pipeline (all 7 agents)
python run_pipeline.py --full

# Custom date range + export
python run_pipeline.py --full --start 2022-01-01 --export ../outputs
```

## Data

Place a CSV in `data/` with columns: `Date, BTC, QQQ, GLD, TLT, UUP, HYG` (adjusted close prices).  
If no CSV is found, the pipeline fetches live data via `yfinance`.

## Agents

| Agent | Role |
|-------|------|
| **Data Agent** | Fetches/loads price data, computes log returns, handles VIX |
| **Feature Agent** | 50+ features: z-scores, rolling vol, cross-asset spreads, momentum, correlations |
| **Model Agent** | Kalman filter with EM learning, RTS smoother, OLS warm-start |
| **Regime Agent** | Classifies BTC into 5 macro regimes based on beta patterns |
| **Risk Agent** | Variance decomposition, VaR/CVaR, drawdowns, position sizing |
| **Signal Agent** | Composite tactical score (-1 to +1), hedging recommendations |
| **Narrative Agent** | PM-style text reports (rule-based, LLM-ready) |

## Kalman Filter Details

The state-space model estimates dynamic regression coefficients:

```
Observation:  y_t = α_t + β_t' · x_t + ε_t,     ε_t ~ N(0, H)
Transition:   [α_t, β_t]' = [α_{t-1}, β_{t-1}]' + η_t,   η_t ~ N(0, Q)
```

- **EM learning** refines observation noise (H) and process noise (Q) iteratively
- **Diagonal Q** with capping for stability (full Q caused exploding uncertainty)
- **RTS smoother** provides backward-smoothed beta estimates
- **OLS warm-start** initializes state from first 60 observations

## Outputs

The `--full --export` run produces:

| File | Contents |
|------|----------|
| `kalman_betas.csv` | Time-varying factor betas |
| `kalman_beta_std.csv` | Beta uncertainty (std dev) |
| `regimes.csv` | Daily regime classifications |
| `z_scores.csv` | Rolling z-scored returns |
| `volatility.csv` | Rolling realized vol |
| `correlations.csv` | Rolling BTC-factor correlations |
| `spreads.csv` | Cross-asset spread features |
| `narrative_report.txt` | Full PM-style narrative |
| `full_summary.json` | Complete metrics for dashboards/PDFs |
