# Vantage V9.1 — Dual-Filter Regime-Aware Portfolio System

> **Walk-forward validated | OOS Sharpe 0.969 | Max Drawdown −17.47% | 2007–2026**

Vantage is a systematic, rules-based portfolio strategy that uses three market signals to detect the current regime and dynamically adjust asset allocation. Built and validated across 18 years of genuine out-of-sample data covering five distinct market environments — GFC, Bull Market, COVID, Rate Hike, and Post-Hike.

---

## Performance Summary

| Metric | **Vantage V9.1** | V8 (VIX only) | SPY (Buy & Hold) |
|---|---|---|---|
| OOS Sharpe | **0.969** | 0.748 | 0.624 |
| Ann. Return | **8.65%** | 8.17% | 12.48% |
| Ann. Volatility | **8.93%** | 10.92% | 20.00% |
| Max Drawdown | **−17.47%** | −24.98% | −53.89% |
| Calmar Ratio | **0.495** | 0.327 | 0.232 |
| Total Return | **341.76%** | 290.84% | 559.58% |

> V9.1 trades raw return for dramatically lower risk. SPY's 559% comes with a −53.89% drawdown. V9.1 delivers 341% with only −17.47% max drawdown and less than half SPY's volatility.

---

## How It Works

### Three-State Regime Engine

```
CRISIS    →  VIX ≥ 90th percentile (1yr lookback)          →  100% Cash (BIL)
DEFENSIVE →  SPY below MA200 OR yield curve inverted        →  50% Optimised + 50% TLT
NORMAL    →  Everything else                                →  100% Sharpe-Optimised
```

### Signal 1 — VIX Percentile (Crisis Filter)
Compares today's VIX to the trailing 252-day distribution. At or above the 90th percentile = CRISIS. Single-snapshot signal — VIX spikes are instantaneous and need no confirmation window.

### Signal 2 — SPY vs 200-Day MA (Trend Filter)
If SPY closes below its 200-day moving average for **45 consecutive days**, the market is in a confirmed downtrend → DEFENSIVE. The 45-day confirmation window is the key V9.1 improvement over V9.0 — it eliminates false triggers from brief dips that quickly recover.

### Signal 3 — Yield Curve Slope (Recession Filter)
The 10Y−2Y Treasury spread has preceded every US recession since 1970. If the curve stays inverted for **45 consecutive days** → DEFENSIVE. Uses `^TNX` (10Y) and `^IRX` (13-week T-bill proxy) from Yahoo Finance.

### Portfolio Optimisation
In NORMAL and DEFENSIVE regimes, weights are determined by maximising the Sharpe ratio over the trailing 3-year training window using `scipy.optimize`. Per-asset bounds: 5%–40%. Weights are re-optimised at each annual fold boundary in the walk-forward.

### Walk-Forward Validation
No in-sample fitting. Every regime decision and weight is computed using only past data, then tested on the next year it never saw. 18 folds from 2007 to 2026.

---

## Asset Universe

| Ticker | Asset Class |
|---|---|
| SPY | US Large Cap Equities |
| TLT | Long-Term US Treasuries (20yr+) |
| GLD | Gold |
| XLE | Energy Sector |
| EEM | Emerging Markets Equities |
| BIL | Cash / T-Bills (CRISIS proxy) |

---

## Regime Stress Test

| Period | V9.1 Sharpe | V9.1 Max DD | V8 Sharpe | SPY Sharpe |
|---|---|---|---|---|
| GFC (2008–09) | −0.246* | **−6.56%** | 0.222 | −0.150 |
| Bull Market (2010–19) | **1.157** | −10.30% | 0.992 | 0.933 |
| COVID (2020–21) | 0.937 | −17.47% | 0.982 | 0.957 |
| Rate Hike (2022–23) | **0.372** | −12.35% | −0.247 ❌ | 0.180 |
| Post-Hike (2024–26) | 1.409 | −10.96% | 1.790 | 1.285 |

*\*GFC Sharpe appears negative due to near-zero BIL yields in 2008–09, not capital loss. V9.1 was 100% cash with only −6.56% drawdown while SPY fell −51.87%. Capital was fully preserved.*

**Key win:** V8 posted −0.247 Sharpe during Rate Hike (2022–23) — it had no signal to detect the slow-bleed regime. V9.1 posted +0.372 Sharpe in the same period because MA200 + yield curve caught it.

---

## Robustness Checks

| Check | Status | Detail |
|---|---|---|
| VIX threshold sensitivity spread < 0.15 | ✅ PASS | spread = 0.110 |
| Sharpe > 0.50 at 30bps transaction cost | ✅ PASS | Sharpe = 0.961 |
| Capital preserved during GFC (DD < 10%) | ✅ PASS | Max DD = −6.56% |
| Positive Sharpe in Rate Hike era | ✅ PASS | Sharpe = 0.372 |
| V9.1 Sharpe > V8 Sharpe | ✅ PASS | 0.969 vs 0.748 |
| Max Drawdown < 30% | ✅ PASS | −17.47% |

---

## Repo Structure

```
vantage/
├── README.md
│
├── backtest/
│   ├── vantage_v8.py          ← Baseline: VIX-only 2-state filter
│   ├── vantage_v9.py          ← V9.0: Dual-filter with snapshot signals
│   └── vantage_v9_1.py        ← V9.1: Dual-filter + 45-day confirmation (MAIN)
│
├── robustness/
│   └── vantage_v10.py         ← V10: Expanded universe (8 assets) stress test
│
└── live/
    └── vantage_v91_live.py    ← Daily runner: fetches data, outputs HTML dashboard
```

---

## Version History

| Version | Key Change | Sharpe |
|---|---|---|
| V8 | VIX-only crisis filter (2-state: NORMAL / CRISIS) | 0.748 |
| V9.0 | Added MA200 + yield curve signals (3-state), snapshot evaluation | ~0.90 |
| V9.1 | 45-day confirmation window on MA200 + yield curve (eliminates false triggers) | **0.969** |
| V10 | Universe expanded to 8 assets (VNQ, DBC, IEF) — robustness validation only | — |

---

## Running the Backtest

```bash
pip install yfinance scipy numpy pandas matplotlib

# Run the main V9.1 backtest
python backtest/vantage_v9_1.py

# Run the V10 universe robustness test
python robustness/vantage_v10.py
```

---

## Running the Live Dashboard

```bash
pip install yfinance scipy numpy pandas

# Basic — virtual portfolio at $100k
python live/vantage_v91_live.py

# With your actual capital
python live/vantage_v91_live.py --capital 250000

# With your actual holdings (for real P&L tracking)
python live/vantage_v91_live.py --capital 100000 --portfolio SPY:0.40,TLT:0.05,GLD:0.386,XLE:0.114,EEM:0.05

# More history on the chart
python live/vantage_v91_live.py --history 90
```

The script fetches live data from Yahoo Finance, applies the V9.1 regime logic, and writes `vantage_dashboard.html` — which auto-opens in your browser. Run it every trading day morning.

---

## Dashboard Features

- **Regime banner** — NORMAL / DEFENSIVE / CRISIS with live pulse animation
- **Signal cards** — VIX percentile, SPY vs MA200 (with 45-day confirmation status), yield curve spread
- **Allocation bars** — weights shift automatically based on current regime
- **Recent equity curve** — V9.1 vs SPY for the last N trading days (real data)
- **Live P&L table** — price, day change, position value, and daily P&L per asset

---

## Important Notes

- **Rebalancing frequency:** The backtest rebalances annually. In live use, re-run the script daily to check for regime changes. Actual rebalancing trades should only happen when the regime changes.
- **Recency bias in weights:** The optimiser uses the trailing 3 years. In periods where one asset has run strongly (e.g. GLD in 2024–25), it will receive higher allocation. This is by design — the regime filters provide the downside protection, not the weights.
- **Not financial advice.** This is a research and educational project. Past performance does not guarantee future results.

---

## Requirements

```
python >= 3.9
yfinance
scipy
numpy
pandas
matplotlib
```

---

*Built with walk-forward discipline. No look-ahead bias. No curve fitting.*
