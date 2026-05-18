"""
Vantage V9.1 — Live Signal Runner
==================================
Run this script daily. It fetches live market data via yfinance,
applies the exact V9.1 regime logic, computes today's recommended
weights, and writes/updates `vantage_dashboard.html`.

Usage:
    python vantage_v91_live.py

Optional flags:
    --capital 100000        Set your portfolio capital (default 100000)
    --portfolio SPY:0.35,TLT:0.25,GLD:0.20,XLE:0.10,EEM:0.10
                            Override current holdings for P&L calc
    --history 30            Days of recent equity curve to show (default 60)

Requirements:
    pip install yfinance scipy numpy pandas
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# ── Configuration (must match backtest) ──────────────────────────────────────
TICKERS              = ['SPY', 'TLT', 'GLD', 'XLE', 'EEM']
CASH_PROXY           = 'BIL'
LOOKBACK_DAYS        = 252 * 3          # 3 years training window
VIX_LB               = 252
MA_WINDOW            = 200
BASE_THRESH          = 0.90
BOUNDS               = (0.05, 0.40)
DEFENSIVE_EQUITY_FRAC = 0.50
CONFIRM_DAYS         = 45

TICKER_NAMES = {
    'SPY': 'US Large Cap',
    'TLT': 'Long Bonds',
    'GLD': 'Gold',
    'XLE': 'Energy',
    'EEM': 'Emerging Mkts',
    'BIL': 'Cash (T-Bills)',
}

TICKER_COLORS = {
    'SPY': '#00e5ff',
    'TLT': '#f0c040',
    'GLD': '#ffd700',
    'XLE': '#50e080',
    'EEM': '#bb80ff',
    'BIL': '#aaaaaa',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _confirmed(bool_series, n=CONFIRM_DAYS):
    """True if the last n values are all True."""
    if len(bool_series) < n:
        return False
    return bool(bool_series.iloc[-n:].all())


def optimise(train_df):
    n      = len(TICKERS)
    bounds = tuple([BOUNDS] * n)
    cons   = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    def neg_sharpe(w):
        p_ret = np.sum(train_df.mean() * w) * 252
        p_vol = np.sqrt(np.dot(w.T, np.dot(train_df.cov() * 252, w)))
        return -p_ret / (p_vol + 1e-9)

    res = minimize(neg_sharpe, [1/n]*n, bounds=bounds, constraints=cons,
                   options={'maxiter': 1000})
    return res.x if res.success else np.array([1/n]*n)


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_live_data():
    end   = datetime.today()
    # LOOKBACK_DAYS is in trading days (~252/yr).
    # timedelta counts calendar days, so multiply by 1.5 to be safe
    # (accounts for weekends, holidays, and any gaps in data).
    calendar_days = int(LOOKBACK_DAYS * 1.5) + 120
    start = end - timedelta(days=calendar_days)

    print("⏳  Fetching market data …")
    raw = yf.download(
        TICKERS + [CASH_PROXY, '^VIX', '^TNX', '^IRX'],
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        progress=False
    )['Close']

    returns      = raw[TICKERS].pct_change().dropna()
    cash_returns = raw[CASH_PROXY].pct_change().fillna(0)
    vix          = raw['^VIX'].reindex(returns.index).ffill()
    spy_price    = raw['SPY'].reindex(returns.index).ffill()
    ma200        = spy_price.rolling(MA_WINDOW).mean()
    yield_10y    = raw['^TNX'].reindex(returns.index).ffill()
    yield_2y     = raw['^IRX'].reindex(returns.index).ffill()
    yield_curve  = (yield_10y - yield_2y).ffill()

    print(f"✅  Loaded {len(returns)} days  ({returns.index[0].date()} → {returns.index[-1].date()})")
    return returns, cash_returns, vix, spy_price, ma200, yield_curve


# ── Signal computation ────────────────────────────────────────────────────────

def compute_signals(returns, vix, spy_price, ma200, yield_curve):
    # Need at least LOOKBACK_DAYS rows
    if len(returns) < LOOKBACK_DAYS:
        raise ValueError(f"Not enough data: {len(returns)} days < {LOOKBACK_DAYS} required")

    train_slice = returns.iloc[-LOOKBACK_DAYS:]

    # VIX percentile (single snapshot)
    hist_vix       = vix.iloc[-VIX_LB - 1 : -1]
    current_vix    = float(vix.iloc[-1])
    vix_percentile = float((hist_vix < current_vix).mean())

    # MA200 — 45-day confirmation
    below_ma_bool  = spy_price < ma200
    below_ma200    = _confirmed(below_ma_bool)

    # Yield curve — 45-day confirmation
    curve_inv_bool = yield_curve < 0
    curve_inverted = _confirmed(curve_inv_bool)

    # Regime
    if vix_percentile >= BASE_THRESH:
        regime = 'CRISIS'
    elif below_ma200 or curve_inverted:
        regime = 'DEFENSIVE'
    else:
        regime = 'NORMAL'

    # Weights
    w = optimise(train_slice)

    # Current prices & 1-day changes
    current_prices = {}
    daily_changes  = {}
    for t in TICKERS + [CASH_PROXY]:
        try:
            import yfinance as yf2
            tk   = yf2.Ticker(t)
            hist = tk.history(period='2d')
            if len(hist) >= 2:
                current_prices[t] = round(float(hist['Close'].iloc[-1]), 2)
                daily_changes[t]  = round(float((hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100), 2)
            else:
                current_prices[t] = None
                daily_changes[t]  = None
        except Exception:
            current_prices[t] = None
            daily_changes[t]  = None

    return {
        'regime'         : regime,
        'vix_percentile' : round(vix_percentile * 100, 1),
        'current_vix'    : round(current_vix, 2),
        'below_ma200'    : bool(below_ma200),
        'curve_inverted' : bool(curve_inverted),
        'current_spy'    : round(float(spy_price.iloc[-1]), 2),
        'current_ma200'  : round(float(ma200.iloc[-1]), 2),
        'current_yc'     : round(float(yield_curve.iloc[-1]), 3),
        'weights'        : {t: round(float(w[i]), 4) for i, t in enumerate(TICKERS)},
        'prices'         : current_prices,
        'daily_changes'  : daily_changes,
        'as_of'          : returns.index[-1].strftime('%Y-%m-%d'),
        'generated_at'   : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


# ── Recent equity curve (for mini-chart) ─────────────────────────────────────

def compute_recent_curve(returns, vix, spy_price, ma200, yield_curve, days=60):
    """Walk forward over the last `days` of data to build a daily equity curve."""
    if len(returns) < LOOKBACK_DAYS + days:
        days = len(returns) - LOOKBACK_DAYS
    if days <= 0:
        return [], []

    curve  = [1.0]
    dates  = []
    spy_c  = [1.0]

    below_ma_bool  = spy_price < ma200
    curve_inv_bool = yield_curve < 0

    start_i = len(returns) - days

    # Build weights from full training window before the slice
    train_slice = returns.iloc[start_i - LOOKBACK_DAYS : start_i]
    w = optimise(train_slice)

    spy_base = None

    for i in range(start_i, len(returns)):
        # Regime check at each step
        hist_vix       = vix.iloc[max(0, i - VIX_LB) : i]
        current_vix    = float(vix.iloc[i])
        vix_pct        = float((hist_vix < current_vix).mean()) if len(hist_vix) > 0 else 0

        bm  = _confirmed(below_ma_bool.iloc[:i])
        ci  = _confirmed(curve_inv_bool.iloc[:i])

        if vix_pct >= BASE_THRESH:
            r = 0.0  # cash (approx 0 daily)
        elif bm or ci:
            port_r = float((returns.iloc[i] * w).sum())
            tlt_r  = float(returns['TLT'].iloc[i])
            r = DEFENSIVE_EQUITY_FRAC * port_r + (1 - DEFENSIVE_EQUITY_FRAC) * tlt_r
        else:
            r = float((returns.iloc[i] * w).sum())

        spy_r = float(returns['SPY'].iloc[i])
        if spy_base is None:
            spy_base = 1.0

        curve.append(curve[-1] * (1 + r))
        spy_c.append(spy_c[-1] * (1 + spy_r))
        dates.append(returns.index[i].strftime('%Y-%m-%d'))

    return dates, curve[1:], spy_c[1:]


# ── HTML generation ───────────────────────────────────────────────────────────

def write_dashboard(signals, dates, curve, spy_curve, capital, holdings, output_path):

    regime        = signals['regime']
    regime_color  = {'CRISIS': '#e05050', 'DEFENSIVE': '#f0c040', 'NORMAL': '#50e080'}[regime]
    regime_emoji  = {'CRISIS': '🔴', 'DEFENSIVE': '🟡', 'NORMAL': '🟢'}[regime]
    regime_desc   = {
        'CRISIS'   : 'VIX spike detected. Portfolio is 100% Cash (BIL). Preserve capital.',
        'DEFENSIVE': 'Downtrend or inverted yield curve confirmed. 50% optimised portfolio + 50% TLT.',
        'NORMAL'   : 'All clear. Deployed 100% in Sharpe-optimised portfolio.',
    }[regime]

    # Weights for display
    if regime == 'CRISIS':
        display_weights = {'BIL': 1.0}
    elif regime == 'DEFENSIVE':
        display_weights = {}
        for t, w in signals['weights'].items():
            display_weights[t] = round(w * DEFENSIVE_EQUITY_FRAC, 4)
        display_weights['TLT'] = round(display_weights.get('TLT', 0) + (1 - DEFENSIVE_EQUITY_FRAC), 4)
    else:
        display_weights = signals['weights']

    # P&L from holdings
    pnl_rows = ''
    total_value = 0.0
    if holdings:
        for ticker, alloc in holdings.items():
            price   = signals['prices'].get(ticker)
            chg     = signals['daily_changes'].get(ticker)
            val     = capital * alloc
            day_pnl = val * (chg / 100) if chg is not None else 0
            total_value += val
            chg_class = 'pos' if (chg or 0) >= 0 else 'neg'
            pnl_rows += f"""
            <tr>
              <td><span class="ticker-badge" style="background:{TICKER_COLORS.get(ticker,'#555')}">{ticker}</span>
                  {TICKER_NAMES.get(ticker, ticker)}</td>
              <td>{alloc*100:.1f}%</td>
              <td>${price if price else '—'}</td>
              <td class="{chg_class}">{f'{chg:+.2f}%' if chg is not None else '—'}</td>
              <td>${val:,.0f}</td>
              <td class="{chg_class}">{f'${day_pnl:+,.0f}' if chg is not None else '—'}</td>
            </tr>"""
    else:
        # Show recommended weights as virtual positions
        for ticker, w in display_weights.items():
            price   = signals['prices'].get(ticker)
            chg     = signals['daily_changes'].get(ticker)
            val     = capital * w
            day_pnl = val * (chg / 100) if chg is not None else 0
            chg_class = 'pos' if (chg or 0) >= 0 else 'neg'
            pnl_rows += f"""
            <tr>
              <td><span class="ticker-badge" style="background:{TICKER_COLORS.get(ticker,'#555')}">{ticker}</span>
                  {TICKER_NAMES.get(ticker, ticker)}</td>
              <td>{w*100:.1f}%</td>
              <td>${price if price else '—'}</td>
              <td class="{chg_class}">{f'{chg:+.2f}%' if chg is not None else '—'}</td>
              <td>${val*w if False else capital*w:,.0f}</td>
              <td class="{chg_class}">{f'${day_pnl:+,.0f}' if chg is not None else '—'}</td>
            </tr>"""

    # Weight bars
    weight_bars = ''
    for ticker, w in display_weights.items():
        color = TICKER_COLORS.get(ticker, '#888')
        weight_bars += f"""
        <div class="weight-row">
          <span class="wlabel">{ticker}</span>
          <div class="wbar-wrap">
            <div class="wbar" style="width:{w*100:.1f}%;background:{color}"></div>
          </div>
          <span class="wpct">{w*100:.1f}%</span>
        </div>"""

    # Chart data
    chart_dates   = json.dumps(dates)
    chart_curve   = json.dumps([round(v, 4) for v in curve])
    chart_spy     = json.dumps([round(v, 4) for v in spy_curve])
    total_ret     = round((curve[-1] - 1) * 100, 2) if curve else 0
    spy_ret       = round((spy_curve[-1] - 1) * 100, 2) if spy_curve else 0
    total_ret_cls = 'pos' if total_ret >= 0 else 'neg'
    spy_ret_cls   = 'pos' if spy_ret >= 0 else 'neg'

    # Signal boxes
    vix_status = 'ELEVATED' if signals['vix_percentile'] >= BASE_THRESH * 100 else 'NORMAL'
    vix_color  = '#e05050' if vix_status == 'ELEVATED' else '#50e080'
    ma_status  = 'BELOW ⚠' if signals['below_ma200'] else 'ABOVE ✓'
    ma_color   = '#e05050' if signals['below_ma200'] else '#50e080'
    yc_status  = 'INVERTED ⚠' if signals['curve_inverted'] else 'NORMAL ✓'
    yc_color   = '#e05050' if signals['curve_inverted'] else '#50e080'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vantage V9.1 — Live Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:       #080c10;
    --surface:  #0e1420;
    --border:   #1e2a38;
    --text:     #c8d8e8;
    --muted:    #4a5a6a;
    --accent:   #00e5ff;
    --gold:     #f0c040;
    --green:    #50e080;
    --red:      #e05050;
    --yellow:   #f0c040;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    background-image:
      radial-gradient(ellipse 80% 40% at 50% -10%, rgba(0,229,255,0.07) 0%, transparent 60%),
      repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(30,42,56,0.4) 39px, rgba(30,42,56,0.4) 40px),
      repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(30,42,56,0.2) 39px, rgba(30,42,56,0.2) 40px);
  }}
  header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 36px;
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(8px);
    position: sticky; top: 0; z-index: 100;
    background: rgba(8,12,16,0.88);
  }}
  .logo {{ font-size: 1.4rem; font-weight: 800; letter-spacing: 0.08em; color: var(--accent); }}
  .logo span {{ color: var(--gold); }}
  .header-meta {{ font-family: 'Space Mono', monospace; font-size: 0.72rem; color: var(--muted); text-align: right; line-height: 1.6; }}
  main {{ max-width: 1400px; margin: 0 auto; padding: 28px 36px 60px; }}

  /* REGIME BANNER */
  .regime-banner {{
    border: 2px solid {regime_color};
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    background: rgba(0,0,0,0.4);
    box-shadow: 0 0 40px {regime_color}22, inset 0 0 60px {regime_color}08;
    display: flex;
    align-items: center;
    gap: 24px;
  }}
  .regime-indicator {{
    width: 80px; height: 80px;
    border-radius: 50%;
    background: {regime_color}22;
    border: 3px solid {regime_color};
    display: flex; align-items: center; justify-content: center;
    font-size: 2.2rem;
    flex-shrink: 0;
    box-shadow: 0 0 30px {regime_color}44;
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{
    0%,100% {{ box-shadow: 0 0 20px {regime_color}44; }}
    50% {{ box-shadow: 0 0 50px {regime_color}88; }}
  }}
  .regime-text h2 {{ font-size: 2rem; font-weight: 800; color: {regime_color}; letter-spacing: 0.1em; }}
  .regime-text p {{ color: var(--text); margin-top: 6px; font-size: 0.95rem; opacity: 0.85; }}

  /* GRID */
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 20px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .grid-1 {{ margin-bottom: 20px; }}

  /* CARDS */
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 24px;
  }}
  .card-title {{
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.15em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 14px;
  }}

  /* SIGNAL CARDS */
  .signal-card {{ position: relative; overflow: hidden; }}
  .signal-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
  }}
  .signal-val {{
    font-size: 2.2rem;
    font-weight: 800;
    font-family: 'Space Mono', monospace;
    line-height: 1;
    margin-bottom: 6px;
  }}
  .signal-status {{
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 10px;
  }}
  .signal-sub {{ font-size: 0.82rem; color: var(--muted); }}

  /* WEIGHTS */
  .weight-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
  .wlabel {{ font-family: 'Space Mono', monospace; font-size: 0.78rem; width: 36px; color: var(--text); }}
  .wbar-wrap {{ flex: 1; background: rgba(255,255,255,0.05); border-radius: 4px; height: 12px; overflow: hidden; }}
  .wbar {{ height: 100%; border-radius: 4px; transition: width 0.6s ease; }}
  .wpct {{ font-family: 'Space Mono', monospace; font-size: 0.78rem; width: 44px; text-align: right; color: var(--text); }}

  /* CHART */
  .chart-wrap {{ position: relative; height: 220px; }}

  /* TABLE */
  .pnl-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  .pnl-table th {{
    text-align: left;
    padding: 8px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
  }}
  .pnl-table td {{ padding: 10px 12px; border-bottom: 1px solid rgba(30,42,56,0.5); }}
  .pnl-table tr:hover td {{ background: rgba(0,229,255,0.03); }}
  .ticker-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    margin-right: 8px;
    color: #000;
  }}
  .pos {{ color: var(--green); }}
  .neg {{ color: var(--red); }}

  /* STAT PILLS */
  .stat-row {{ display: flex; gap: 16px; margin-top: 12px; }}
  .stat-pill {{
    background: rgba(0,0,0,0.3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px;
    text-align: center;
    flex: 1;
  }}
  .stat-pill .sv {{ font-size: 1.2rem; font-weight: 800; font-family: 'Space Mono', monospace; }}
  .stat-pill .sl {{ font-size: 0.65rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 2px; }}

  /* FOOTER */
  footer {{
    text-align: center;
    padding: 24px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    margin-top: 40px;
  }}

  @media (max-width: 900px) {{
    .grid-3 {{ grid-template-columns: 1fr; }}
    .grid-2 {{ grid-template-columns: 1fr; }}
    main {{ padding: 16px; }}
  }}
</style>
</head>
<body>

<header>
  <div class="logo">VANTAGE <span>V9.1</span></div>
  <div class="header-meta">
    As of: {signals['as_of']}<br>
    Generated: {signals['generated_at']}<br>
    Capital: ${capital:,.0f}
  </div>
</header>

<main>

  <!-- REGIME BANNER -->
  <div class="regime-banner">
    <div class="regime-indicator">{regime_emoji}</div>
    <div class="regime-text">
      <h2>REGIME: {regime}</h2>
      <p>{regime_desc}</p>
    </div>
  </div>

  <!-- SIGNALS ROW -->
  <div class="grid-3">

    <div class="card signal-card">
      <div class="card-title">Signal 1 — VIX Percentile</div>
      <div class="signal-val" style="color:{vix_color}">{signals['vix_percentile']:.1f}<span style="font-size:1rem">%ile</span></div>
      <div class="signal-status" style="background:{vix_color}22;color:{vix_color}">{vix_status}</div>
      <div class="signal-sub">
        VIX level: <strong>{signals['current_vix']:.2f}</strong><br>
        Crisis threshold: ≥ {BASE_THRESH*100:.0f}th percentile
      </div>
    </div>

    <div class="card signal-card">
      <div class="card-title">Signal 2 — SPY vs MA200</div>
      <div class="signal-val" style="color:{ma_color}">${signals['current_spy']:.2f}</div>
      <div class="signal-status" style="background:{ma_color}22;color:{ma_color}">{ma_status}</div>
      <div class="signal-sub">
        MA200: <strong>${signals['current_ma200']:.2f}</strong><br>
        45-day confirmation: <strong>{'ACTIVE' if signals['below_ma200'] else 'NOT TRIGGERED'}</strong>
      </div>
    </div>

    <div class="card signal-card">
      <div class="card-title">Signal 3 — Yield Curve (10Y–2Y)</div>
      <div class="signal-val" style="color:{yc_color}">{signals['current_yc']:+.3f}<span style="font-size:1rem">%</span></div>
      <div class="signal-status" style="background:{yc_color}22;color:{yc_color}">{yc_status}</div>
      <div class="signal-sub">
        Spread: <strong>{signals['current_yc']:+.3f}%</strong><br>
        45-day inversion: <strong>{'CONFIRMED' if signals['curve_inverted'] else 'NOT CONFIRMED'}</strong>
      </div>
    </div>

  </div>

  <!-- WEIGHTS + CHART -->
  <div class="grid-2">

    <div class="card">
      <div class="card-title">Recommended Allocation — {regime} Mode</div>
      {weight_bars}
      <div class="stat-row" style="margin-top:18px">
        <div class="stat-pill">
          <div class="sv" style="color:var(--accent)">{regime}</div>
          <div class="sl">Current Regime</div>
        </div>
        <div class="stat-pill">
          <div class="sv">{len(display_weights)}</div>
          <div class="sl">Active Positions</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Recent Performance — V9.1 vs SPY ({len(dates)} days)</div>
      <div class="chart-wrap">
        <canvas id="curveChart"></canvas>
      </div>
      <div class="stat-row">
        <div class="stat-pill">
          <div class="sv {total_ret_cls}">{total_ret:+.2f}%</div>
          <div class="sl">V9.1 Return</div>
        </div>
        <div class="stat-pill">
          <div class="sv {spy_ret_cls}">{spy_ret:+.2f}%</div>
          <div class="sl">SPY Return</div>
        </div>
        <div class="stat-pill">
          <div class="sv {'pos' if total_ret >= spy_ret else 'neg'}">{total_ret - spy_ret:+.2f}%</div>
          <div class="sl">Alpha</div>
        </div>
      </div>
    </div>

  </div>

  <!-- P&L TABLE -->
  <div class="card grid-1">
    <div class="card-title">Live P&L — {'Current Holdings' if holdings else 'Virtual (Recommended Weights @ $'+f'{capital:,.0f}'+')'}</div>
    <table class="pnl-table">
      <thead>
        <tr>
          <th>Asset</th>
          <th>Allocation</th>
          <th>Price</th>
          <th>Day Chg</th>
          <th>Value</th>
          <th>Day P&L</th>
        </tr>
      </thead>
      <tbody>
        {pnl_rows}
      </tbody>
    </table>
  </div>

</main>

<footer>
  VANTAGE V9.1 — Dual-Filter Regime Detection | VIX + MA200 + Yield Curve | 45-Day Confirmation Window<br>
  Not financial advice. For informational and research purposes only.
</footer>

<script>
const ctx = document.getElementById('curveChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: {chart_dates},
    datasets: [
      {{
        label: 'V9.1',
        data: {chart_curve},
        borderColor: '#00e5ff',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
        backgroundColor: 'rgba(0,229,255,0.06)',
      }},
      {{
        label: 'SPY',
        data: {chart_spy},
        borderColor: '#666666',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
        borderDash: [4,3],
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{
        labels: {{ color: '#8899aa', font: {{ family: 'Space Mono', size: 10 }} }}
      }},
      tooltip: {{
        mode: 'index',
        intersect: false,
        backgroundColor: '#0e1420',
        borderColor: '#1e2a38',
        borderWidth: 1,
        titleColor: '#c8d8e8',
        bodyColor: '#8899aa',
        callbacks: {{
          label: ctx => ` ${{ctx.dataset.label}}: ${{((ctx.raw - 1)*100).toFixed(2)}}%`
        }}
      }}
    }},
    scales: {{
      x: {{
        ticks: {{ color: '#4a5a6a', maxTicksLimit: 6, font: {{ family: 'Space Mono', size: 9 }} }},
        grid: {{ color: 'rgba(30,42,56,0.5)' }}
      }},
      y: {{
        ticks: {{
          color: '#4a5a6a',
          font: {{ family: 'Space Mono', size: 9 }},
          callback: v => ((v-1)*100).toFixed(1) + '%'
        }},
        grid: {{ color: 'rgba(30,42,56,0.5)' }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅  Dashboard written → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_holdings(s):
    out = {}
    for part in s.split(','):
        ticker, pct = part.strip().split(':')
        out[ticker.strip().upper()] = float(pct)
    total = sum(out.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Holdings must sum to 1.0, got {total:.3f}")
    return out


def main():
    parser = argparse.ArgumentParser(description='Vantage V9.1 Live Dashboard')
    parser.add_argument('--capital',   type=float, default=100_000, help='Portfolio capital in USD')
    parser.add_argument('--portfolio', type=str,   default='',
                        help='Current holdings e.g. SPY:0.35,TLT:0.25,GLD:0.20,XLE:0.10,EEM:0.10')
    parser.add_argument('--history',   type=int,   default=60, help='Days of history for mini-chart')
    parser.add_argument('--output',    type=str,   default='vantage_dashboard.html')
    args = parser.parse_args()

    holdings = {}
    if args.portfolio:
        try:
            holdings = parse_holdings(args.portfolio)
        except Exception as e:
            print(f"⚠️  Could not parse --portfolio: {e}")

    try:
        returns, cash_returns, vix, spy_price, ma200, yield_curve = fetch_live_data()
        signals = compute_signals(returns, vix, spy_price, ma200, yield_curve)

        print(f"\n{'='*55}")
        print(f"  VANTAGE V9.1  —  {signals['as_of']}")
        print(f"{'='*55}")
        print(f"  Regime         : {signals['regime']}")
        print(f"  VIX Percentile : {signals['vix_percentile']:.1f}%  (level: {signals['current_vix']:.2f})")
        print(f"  SPY vs MA200   : {'BELOW ⚠' if signals['below_ma200'] else 'ABOVE ✓'}"
              f"  (SPY ${signals['current_spy']:.2f} / MA ${signals['current_ma200']:.2f})")
        print(f"  Yield Curve    : {'INVERTED ⚠' if signals['curve_inverted'] else 'NORMAL ✓'}"
              f"  ({signals['current_yc']:+.3f}%)")
        print(f"\n  Recommended Weights:")
        for t, w in signals['weights'].items():
            print(f"    {t:<5}  {w*100:.1f}%")
        print(f"{'='*55}\n")

        print("⏳  Computing recent equity curve …")
        dates, curve, spy_curve = compute_recent_curve(
            returns, vix, spy_price, ma200, yield_curve, days=args.history
        )

        write_dashboard(signals, dates, curve, spy_curve,
                        args.capital, holdings, args.output)

        # Auto-open in browser
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(args.output)}")

    except Exception as e:
        print(f"\n❌  Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
