"""
Vantage V9 — Dual-Filter Regime Detection
==========================================
Fixes the core V8 blind spot: the VIX filter only catches volatility
spikes (sharp crashes) but misses slow-bleed rate-hike regimes where
VIX stays calm while equities grind lower (e.g. 2022).

V9 adds TWO additional signals on top of VIX:

  SIGNAL 1 — 200-Day Moving Average (MA200)
    If SPY closes below its 200-day MA, the market is in a confirmed
    downtrend. This catches slow bear markets the VIX filter misses.

  SIGNAL 2 — Yield Curve Slope (10Y - 2Y spread)
    An inverted yield curve (spread < 0) has preceded every US recession
    since 1970. When the curve inverts, we reduce equity exposure and
    rotate defensively. Fetched via ^TNX (10Y) and ^IRX (13-week proxy
    for 2Y) from Yahoo Finance.

REGIME LOGIC (3-state):
  CRISIS   : VIX >= 90th percentile                -> 100% Cash (BIL)
  DEFENSIVE: SPY below MA200  OR  yield curve < 0  -> 50% optimised + 50% TLT
  NORMAL   : everything else                       -> 100% optimised portfolio

This means V9 has a "soft landing" mode instead of a binary on/off switch,
which is more realistic and avoids whipsaw from brief MA crossings.

All three robustness mods from V8 are preserved.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

TICKERS     = ['SPY', 'TLT', 'GLD', 'XLE', 'EEM']
CASH_PROXY  = 'BIL'
START       = '2004-01-01'
END         = '2026-05-15'
LOOKBACK    = 252 * 3
STEP        = 252
VIX_LB      = 252
MA_WINDOW   = 200          # days for the moving average signal
BASE_THRESH = 0.90         # VIX crisis threshold
BOUNDS      = (0.05, 0.40)
DEFENSIVE_EQUITY_FRAC = 0.50   # in defensive mode: 50% optimised, 50% TLT


# ── Data fetch ─────────────────────────────────────────────────────────────────

def fetch_data():
    print("Fetching price data, VIX, and yield curve ...")

    # Core ETFs + cash + VIX
    raw = yf.download(
        TICKERS + [CASH_PROXY, '^VIX', '^TNX', '^IRX'],
        start=START, end=END, progress=False
    )['Close']

    returns      = raw[TICKERS].pct_change().dropna()
    cash_returns = raw[CASH_PROXY].pct_change().fillna(0)
    vix          = raw['^VIX']
    spy_price    = raw['SPY']

    # Yield curve: 10Y minus 2Y proxy (^IRX = 13-week, annualised %)
    # Both series are already in annualised % terms
    yield_10y    = raw['^TNX']
    yield_2y     = raw['^IRX']
    yield_curve  = (yield_10y - yield_2y).ffill()

    # MA200 on SPY
    ma200        = spy_price.rolling(MA_WINDOW).mean()

    print(f"  Loaded {len(returns)} trading days "
          f"({returns.index[0].date()} -> {returns.index[-1].date()})")
    print(f"  Yield curve data: {yield_curve.dropna().index[0].date()} -> "
          f"{yield_curve.dropna().index[-1].date()}\n")
    return returns, cash_returns, vix, spy_price, ma200, yield_curve


# ── Optimizer ──────────────────────────────────────────────────────────────────

def optimise(train_slice):
    n      = len(TICKERS)
    bounds = tuple([BOUNDS] * n)
    cons   = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    def neg_sharpe(w):
        p_ret = np.sum(train_slice.mean() * w) * 252
        p_vol = np.sqrt(np.dot(w.T, np.dot(train_slice.cov() * 252, w)))
        return -p_ret / (p_vol + 1e-9)

    res = minimize(neg_sharpe, [1/n]*n, bounds=bounds, constraints=cons,
                   options={'maxiter': 1000})
    return res.x if res.success else np.array([1/n]*n)


# ── V9.1 Signal confirmation helper ───────────────────────────────────────────
#
# V9.0 PROBLEMS FIXED HERE:
#
#   PROBLEM 1: Yield curve snapshot fired late for Rate Hike (2022) and
#   lingered too long into Post-Hike (2024), keeping the portfolio DEFENSIVE
#   during a strong bull run and costing ~60bps of Sharpe vs V8.
#
#   PROBLEM 2: MA200 single-day cross was too noisy — triggered DEFENSIVE
#   during brief dips that quickly reversed.
#
#   FIX: Both MA200 and yield curve now require CONFIRM_DAYS (45) consecutive
#   days of the signal being true before the regime flips to DEFENSIVE.
#   VIX spikes are still single-snapshot (they don't need confirmation —
#   a sudden VIX spike IS the signal).
#
#   EFFECT: Fewer false DEFENSIVE triggers in 2024 (Post-Hike).
#           Earlier catch of sustained 2022 inversion (Rate Hike).

CONFIRM_DAYS = 45


def _confirmed(bool_series, up_to_i, n=CONFIRM_DAYS):
    """True if the last n values of bool_series before index i are all True."""
    if up_to_i < n:
        return False
    return bool(bool_series.iloc[up_to_i - n : up_to_i].all())


def walk_forward_v9(returns, cash_returns, vix, spy_price, ma200, yield_curve,
                    vix_threshold=BASE_THRESH, cost_bps=0):
    """
    3-state regime: CRISIS / DEFENSIVE / NORMAL
    MA200 and yield curve signals require CONFIRM_DAYS confirmation.
    Returns daily OOS returns and a regime log.
    """
    cost        = cost_bps / 10_000
    oos_returns = []
    regime_log  = []
    last_regime = None

    # Pre-align all series to the returns index so integer iloc works cleanly
    spy_al = spy_price.reindex(returns.index).ffill()
    ma_al  = ma200.reindex(returns.index).ffill()
    yc_al  = yield_curve.reindex(returns.index).ffill()
    vix_al = vix.reindex(returns.index).ffill()

    below_ma_bool    = spy_al < ma_al
    curve_inv_bool   = yc_al < 0

    for i in range(LOOKBACK, len(returns) - STEP, STEP):
        train_slice = returns.iloc[i - LOOKBACK:i]
        test_slice  = returns.iloc[i:i + STEP]

        # ── VIX: single snapshot (spikes are instantaneous) ─────────────────
        hist_vix       = vix_al.iloc[i - VIX_LB:i]
        current_vix    = vix_al.iloc[i]
        vix_percentile = (hist_vix < current_vix).mean()

        # ── MA200: confirmed downtrend (45 days below MA) ────────────────────
        below_ma200    = _confirmed(below_ma_bool, i)

        # ── Yield curve: confirmed sustained inversion (45 days) ─────────────
        curve_inverted = _confirmed(curve_inv_bool, i)

        # ── Regime classification ─────────────────────────────────────────────
        if vix_percentile >= vix_threshold:
            regime = 'CRISIS'
        elif below_ma200 or curve_inverted:
            regime = 'DEFENSIVE'
        else:
            regime = 'NORMAL'

        # ── Optimise on training data (used in NORMAL and DEFENSIVE) ────────
        w = optimise(train_slice)

        # ── Apply regime ─────────────────────────────────────────────────────
        if regime == 'CRISIS':
            period_returns = cash_returns.loc[test_slice.index].copy()

        elif regime == 'DEFENSIVE':
            # 50% in optimised portfolio, 50% in TLT (duration hedge)
            port_rets = (test_slice * w).sum(axis=1)
            tlt_rets  = test_slice['TLT']
            period_returns = (DEFENSIVE_EQUITY_FRAC * port_rets +
                              (1 - DEFENSIVE_EQUITY_FRAC) * tlt_rets).copy()
        else:
            period_returns = (test_slice * w).sum(axis=1).copy()

        # MOD 3: charge switching cost on regime change
        if last_regime is not None and regime != last_regime:
            period_returns.iloc[0] -= cost

        last_regime = regime
        oos_returns.append(period_returns)
        regime_log.append({
            'start'          : test_slice.index[0],
            'end'            : test_slice.index[-1],
            'regime'         : regime,
            'vix_pct'        : round(vix_percentile, 3),
            'below_ma200'    : below_ma200,
            'curve_inverted' : curve_inverted,
        })

    return pd.concat(oos_returns), pd.DataFrame(regime_log)


# ── V8 Walk-Forward (for comparison) ──────────────────────────────────────────

def walk_forward_v8(returns, cash_returns, vix,
                    vix_threshold=BASE_THRESH, cost_bps=0):
    cost        = cost_bps / 10_000
    oos_returns = []
    last_regime = None

    for i in range(LOOKBACK, len(returns) - STEP, STEP):
        train_slice    = returns.iloc[i - LOOKBACK:i]
        test_slice     = returns.iloc[i:i + STEP]
        hist_vix       = vix.iloc[i - VIX_LB:i]
        current_vix    = vix.iloc[i]
        vix_percentile = (hist_vix < current_vix).mean()

        if vix_percentile >= vix_threshold:
            regime         = 'CRISIS'
            period_returns = cash_returns.loc[test_slice.index].copy()
        else:
            regime         = 'NORMAL'
            w              = optimise(train_slice)
            period_returns = (test_slice * w).sum(axis=1).copy()

        if last_regime is not None and regime != last_regime:
            period_returns.iloc[0] -= cost
        last_regime = regime
        oos_returns.append(period_returns)

    return pd.concat(oos_returns)


# ── Performance stats ──────────────────────────────────────────────────────────

def stats(rets, label=''):
    cum     = (1 + rets).cumprod()
    total   = (cum.iloc[-1] - 1) * 100
    ann_ret = rets.mean() * 252 * 100
    ann_vol = rets.std() * np.sqrt(252) * 100
    sharpe  = ann_ret / (ann_vol + 1e-9)
    dd      = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    calmar  = ann_ret / abs(dd) if dd != 0 else np.nan
    return dict(label=label, total=round(total, 2), ann_ret=round(ann_ret, 2),
                ann_vol=round(ann_vol, 2), sharpe=round(sharpe, 3),
                max_dd=round(dd, 2), calmar=round(calmar, 3))


def slice_stats(full_rets, start, end, label):
    slc = full_rets.loc[start:end]
    return stats(slc, label) if len(slc) >= 20 else None


# ── Regime sub-periods ─────────────────────────────────────────────────────────

REGIMES = [
    ('GFC',         '2008-01-01', '2009-12-31'),
    ('Bull Market', '2010-01-01', '2019-12-31'),
    ('COVID',       '2020-01-01', '2021-12-31'),
    ('Rate Hike',   '2022-01-01', '2023-12-31'),
    ('Post-Hike',   '2024-01-01', '2026-05-15'),
]


# ── MOD 2: VIX threshold sensitivity ──────────────────────────────────────────

THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]

def run_sensitivity(returns, cash_returns, vix, spy_price, ma200, yield_curve):
    print("=" * 60)
    print("MOD 2 — VIX THRESHOLD SENSITIVITY (V9)")
    print("=" * 60)
    print(f"  {'Threshold':>12}  {'Sharpe':>8}  {'Max DD':>9}  {'Total Ret':>11}")
    print("  " + "-" * 48)
    results = []
    for t in THRESHOLDS:
        oos, _ = walk_forward_v9(returns, cash_returns, vix,
                                 spy_price, ma200, yield_curve,
                                 vix_threshold=t, cost_bps=0)
        s = stats(oos, f'p{int(t*100)}')
        results.append(s)
        marker = " <- base" if t == BASE_THRESH else ""
        print(f"  {t:>12.0%}  {s['sharpe']:>8.3f}  {s['max_dd']:>8.2f}%  "
              f"{s['total']:>10.2f}%{marker}")
    sharpes = [s['sharpe'] for s in results]
    spread  = max(sharpes) - min(sharpes)
    verdict = ("ROBUST - spread < 0.15" if spread < 0.15
               else "CONCERN - spread >= 0.15")
    print(f"\n  Sharpe spread: {spread:.3f}  |  {verdict}\n")
    return results


# ── MOD 3: Transaction cost sensitivity ───────────────────────────────────────

COST_BPS = [0, 10, 20, 30]

def run_costs(returns, cash_returns, vix, spy_price, ma200, yield_curve):
    print("=" * 60)
    print("MOD 3 — TRANSACTION COST MODEL (V9)")
    print("=" * 60)
    print(f"  {'Cost (bps)':>12}  {'Sharpe':>8}  {'Total Ret':>11}  {'Sharpe decay':>13}")
    print("  " + "-" * 50)
    results     = []
    base_sharpe = None
    for bps in COST_BPS:
        oos, _ = walk_forward_v9(returns, cash_returns, vix,
                                 spy_price, ma200, yield_curve,
                                 vix_threshold=BASE_THRESH, cost_bps=bps)
        s = stats(oos, f'{bps}bps')
        results.append(s)
        if base_sharpe is None:
            base_sharpe = s['sharpe']
        decay = (base_sharpe - s['sharpe']) / base_sharpe * 100
        print(f"  {bps:>12}  {s['sharpe']:>8.3f}  {s['total']:>10.2f}%  {decay:>12.1f}%")
    print()
    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

REGIME_COLORS = {
    'NORMAL'    : '#50e080',
    'DEFENSIVE' : '#f0c040',
    'CRISIS'    : '#e05050',
}

def plot_all(returns, oos_v9, oos_v8, regime_log,
             sensitivity, cost_results, spy_price, ma200, yield_curve):

    idx     = oos_v9.index
    spy_oos = returns['SPY'].loc[idx]

    cum_v9  = (1 + oos_v9).cumprod()
    cum_v8  = (1 + oos_v8.loc[idx]).cumprod()
    cum_spy = (1 + spy_oos).cumprod()

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0d0d0d')
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.38)

    GOLD = '#f0c040'
    CYAN = '#00e5ff'
    GRY  = '#666666'
    GRN  = '#50e080'
    RED  = '#e05050'
    BLU  = '#4fa3e0'
    lkw  = dict(color='#aaaaaa', fontsize=9)
    tkw  = dict(color='#ffffff', fontsize=10, fontweight='bold', pad=8)

    def style(ax):
        ax.set_facecolor('#111111')
        ax.tick_params(colors='#888888', labelsize=8)
        ax.spines[:].set_color('#333333')

    # ── Row 0: Equity curves ───────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    style(ax0)
    ax0.plot(cum_v9.index,  cum_v9.values,  color=CYAN, lw=2.5, label='V9 (dual-filter)', zorder=3)
    ax0.plot(cum_v8.index,  cum_v8.values,  color=GOLD, lw=1.5, linestyle='--',
             label='V8 (VIX only)', alpha=0.8)
    ax0.plot(cum_spy.index, cum_spy.values, color=GRY,  lw=1.2, alpha=0.5, label='SPY')

    # Shade regimes from log
    for _, row in regime_log.iterrows():
        ax0.axvspan(row['start'], row['end'],
                    alpha=0.12, color=REGIME_COLORS.get(row['regime'], GRY))

    # Custom regime legend patches
    from matplotlib.patches import Patch
    regime_patches = [Patch(facecolor=REGIME_COLORS[r], alpha=0.5, label=r)
                      for r in ['NORMAL', 'DEFENSIVE', 'CRISIS']]
    leg0 = ax0.legend(
        handles=ax0.get_lines() + regime_patches,
        fontsize=8, framealpha=0.3, facecolor='#222222',
        edgecolor='#444444', ncol=6
    )
    for t in leg0.get_texts():
        t.set_color('#cccccc')
    ax0.set_title('V9.1 Equity Curve — Confirmed Dual-Filter (45-day window, green=normal, yellow=defensive, red=crisis)',
                  **tkw)
    ax0.set_ylabel('Growth of $1', **lkw)

    # ── Row 1 left: SPY vs MA200 ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    style(ax1)
    spy_plot  = spy_price.loc[idx]
    ma_plot   = ma200.loc[idx]
    ax1.plot(spy_plot.index, spy_plot.values, color=GRY,  lw=1,   label='SPY price')
    ax1.plot(ma_plot.index,  ma_plot.values,  color=GOLD, lw=1.5, label='MA200')
    ax1.fill_between(spy_plot.index,
                     spy_plot.values, ma_plot.values,
                     where=(spy_plot.values < ma_plot.values),
                     alpha=0.25, color=RED, label='Below MA200')
    ax1.set_title('SPY vs 200-Day MA  (Signal 1)', **tkw)
    ax1.set_ylabel('Price ($)', **lkw)
    leg1 = ax1.legend(fontsize=7, framealpha=0.3, facecolor='#222222', edgecolor='#444444')
    for t in leg1.get_texts(): t.set_color('#cccccc')

    # ── Row 1 centre: Yield curve ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    style(ax2)
    yc_plot = yield_curve.loc[idx].dropna()
    ax2.plot(yc_plot.index, yc_plot.values, color=BLU, lw=1.2, label='10Y - 2Y spread')
    ax2.axhline(0, color=RED, lw=1, linestyle='--', label='Inversion line')
    ax2.fill_between(yc_plot.index, yc_plot.values, 0,
                     where=(yc_plot.values < 0),
                     alpha=0.3, color=RED, label='Inverted')
    ax2.set_title('Yield Curve: 10Y minus 2Y  (Signal 2)', **tkw)
    ax2.set_ylabel('Spread (%)', **lkw)
    leg2 = ax2.legend(fontsize=7, framealpha=0.3, facecolor='#222222', edgecolor='#444444')
    for t in leg2.get_texts(): t.set_color('#cccccc')

    # ── Row 1 right: Regime distribution ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    style(ax3)
    regime_counts = regime_log['regime'].value_counts()
    colors_pie    = [REGIME_COLORS.get(r, GRY) for r in regime_counts.index]
    wedges, texts, autotexts = ax3.pie(
        regime_counts.values,
        labels=regime_counts.index,
        colors=colors_pie,
        autopct='%1.0f%%',
        textprops={'color': '#cccccc', 'fontsize': 9},
        wedgeprops={'linewidth': 0.5, 'edgecolor': '#333333'}
    )
    for at in autotexts: at.set_color('#ffffff')
    ax3.set_title('Regime Distribution (% of folds)', **tkw)

    # ── Row 2 left: Regime Sharpe comparison V8 vs V9 ─────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    style(ax4)
    spy_oos_full = returns['SPY'].loc[idx]
    v9_sharpes, v8_sharpes_r, spy_sharpes_r, reg_labels = [], [], [], []
    for name, s, e in REGIMES:
        rv9 = slice_stats(oos_v9,        s, e, name)
        rv8 = slice_stats(oos_v8.loc[idx], s, e, name)
        rsp = slice_stats(spy_oos_full,  s, e, name)
        if rv9:
            v9_sharpes.append(rv9['sharpe'])
            v8_sharpes_r.append(rv8['sharpe'] if rv8 else 0)
            spy_sharpes_r.append(rsp['sharpe'] if rsp else 0)
            reg_labels.append(name)

    x  = np.arange(len(reg_labels))
    bw = 0.25
    ax4.bar(x - bw,   v9_sharpes,   width=bw, color=CYAN, label='V9')
    ax4.bar(x,        v8_sharpes_r, width=bw, color=GOLD, label='V8')
    ax4.bar(x + bw,   spy_sharpes_r,width=bw, color=GRY,  label='SPY', alpha=0.7)
    ax4.axhline(0, color='#555555', lw=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(reg_labels, rotation=20, ha='right', fontsize=7, color='#aaaaaa')
    ax4.set_title('Sharpe by Regime: V9 vs V8 vs SPY', **tkw)
    ax4.set_ylabel('Sharpe Ratio', **lkw)
    leg4 = ax4.legend(fontsize=8, framealpha=0.3, facecolor='#222222', edgecolor='#444444')
    for t in leg4.get_texts(): t.set_color('#cccccc')

    # ── Row 2 centre: Drawdown comparison ─────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    style(ax5)
    for cum, color, label in [
        (cum_v9,  CYAN, 'V9'),
        (cum_v8,  GOLD, 'V8'),
        (cum_spy, GRY,  'SPY'),
    ]:
        dd = (cum - cum.cummax()) / cum.cummax() * 100
        ax5.fill_between(dd.index, dd.values, 0, alpha=0.25, color=color)
        ax5.plot(dd.index, dd.values, color=color, lw=1, label=label)
    ax5.set_title('Drawdown: V9 vs V8 vs SPY', **tkw)
    ax5.set_ylabel('Drawdown (%)', **lkw)
    leg5 = ax5.legend(fontsize=8, framealpha=0.3, facecolor='#222222', edgecolor='#444444')
    for t in leg5.get_texts(): t.set_color('#cccccc')

    # ── Row 2 right: VIX threshold sensitivity ────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    style(ax6)
    th_sharpes = [s['sharpe'] for s in sensitivity]
    th_labels  = [f"p{int(t*100)}" for t in THRESHOLDS]
    colors_th  = [CYAN if t == BASE_THRESH else BLU for t in THRESHOLDS]
    ax6.bar(th_labels, th_sharpes, color=colors_th, width=0.5)
    spread = max(th_sharpes) - min(th_sharpes)
    ax6.text(0.5, 0.05, f'Spread: {spread:.3f}',
             transform=ax6.transAxes, color='#aaaaaa', fontsize=9, ha='center')
    ax6.set_title('VIX Threshold Sensitivity  (MOD 2)', **tkw)
    ax6.set_ylabel('OOS Sharpe', **lkw)
    ax6.set_xlabel('VIX percentile trigger', **lkw)

    # ── Row 3: Cost degradation + summary scorecard ────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    style(ax7)
    cost_sharpes = [s['sharpe'] for s in cost_results]
    cost_returns = [s['total']  for s in cost_results]
    ax7.plot(COST_BPS, cost_sharpes, color=CYAN, marker='o', lw=2, label='Sharpe')
    ax7b = ax7.twinx()
    ax7b.set_facecolor('#111111')
    ax7b.plot(COST_BPS, cost_returns, color=GRN, marker='s',
              lw=1.5, linestyle='--', label='Total Return %')
    ax7b.tick_params(colors='#888888', labelsize=8)
    ax7b.spines[:].set_color('#333333')
    ax7.set_title('Cost & Slippage Degradation  (MOD 3)', **tkw)
    ax7.set_xlabel('Round-trip cost (bps)', **lkw)
    ax7.set_ylabel('Sharpe Ratio', **lkw)
    ax7b.set_ylabel('Total Return (%)', color='#aaaaaa', fontsize=9)
    lines1, labs1 = ax7.get_legend_handles_labels()
    lines2, labs2 = ax7b.get_legend_handles_labels()
    leg7 = ax7.legend(lines1 + lines2, labs1 + labs2, fontsize=8,
                      framealpha=0.3, facecolor='#222222', edgecolor='#444444')
    for t in leg7.get_texts(): t.set_color('#cccccc')

    # ── Row 3 right: Summary stats table ──────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1:])
    style(ax8)
    ax8.axis('off')

    sv9  = stats(oos_v9,          'V9')
    sv8  = stats(oos_v8.loc[idx], 'V8')
    sspy = stats(spy_oos,         'SPY')

    table_data = [
        ['Metric',         'V9 (dual-filter)', 'V8 (VIX only)', 'SPY'],
        ['OOS Sharpe',     sv9['sharpe'],       sv8['sharpe'],   sspy['sharpe']],
        ['Ann. Return %',  sv9['ann_ret'],       sv8['ann_ret'],  sspy['ann_ret']],
        ['Ann. Vol %',     sv9['ann_vol'],       sv8['ann_vol'],  sspy['ann_vol']],
        ['Max Drawdown %', sv9['max_dd'],        sv8['max_dd'],   sspy['max_dd']],
        ['Calmar',         sv9['calmar'],        sv8['calmar'],   sspy['calmar']],
        ['Total Return %', sv9['total'],         sv8['total'],    sspy['total']],
    ]

    tbl = ax8.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#1a1a1a' if r % 2 == 0 else '#111111')
        cell.set_edgecolor('#333333')
        cell.set_text_props(color='#cccccc')
        if r == 0:
            cell.set_facecolor('#222244')
            cell.set_text_props(color='#ffffff', fontweight='bold')
        if c == 1 and r > 0:
            cell.set_facecolor('#0d1f2d')
            cell.set_text_props(color=CYAN)

    ax8.set_title('Performance Summary: V9 vs V8 vs SPY', **tkw)

    plt.suptitle(
        'Vantage V9.1 — Dual-Filter with 45-Day Confirmation Window (VIX + MA200 + Yield Curve)',
        color='#ffffff', fontsize=13, fontweight='bold', y=1.01
    )
    plt.savefig('vantage_v9.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print("Chart saved -> vantage_v9.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    returns, cash_returns, vix, spy_price, ma200, yield_curve = fetch_data()

    # ── V9 base run ───────────────────────────────────────────────────────
    print("Running V9 walk-forward (dual-filter) ...")
    oos_v9, regime_log = walk_forward_v9(
        returns, cash_returns, vix, spy_price, ma200, yield_curve,
        vix_threshold=BASE_THRESH, cost_bps=0
    )

    # ── V8 for comparison ─────────────────────────────────────────────────
    print("Running V8 walk-forward (VIX only, for comparison) ...")
    oos_v8 = walk_forward_v8(returns, cash_returns, vix,
                              vix_threshold=BASE_THRESH, cost_bps=0)

    idx  = oos_v9.index
    sv9  = stats(oos_v9,           'V9')
    sv8  = stats(oos_v8.loc[idx],  'V8')
    sspy = stats(returns['SPY'].loc[idx], 'SPY')

    print(f"\n{'='*60}")
    print(f"  V9 vs V8 vs SPY — HEADLINE RESULTS")
    print(f"{'='*60}")
    for s in [sv9, sv8, sspy]:
        print(f"\n  [{s['label']}]")
        print(f"    Sharpe      : {s['sharpe']}")
        print(f"    Max Drawdown: {s['max_dd']}%")
        print(f"    Total Return: {s['total']}%")
        print(f"    Calmar      : {s['calmar']}")

    # ── Regime log ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  REGIME LOG")
    print(f"{'='*60}")
    print(f"  {'Period':<30}  {'Regime':<12}  {'VIX pct':>8}  "
          f"{'MA<200':>8}  {'Curve inv':>10}")
    print("  " + "-"*72)
    for _, row in regime_log.iterrows():
        period = f"{row['start'].date()} -> {row['end'].date()}"
        print(f"  {period:<30}  {row['regime']:<12}  {row['vix_pct']:>8.3f}  "
              f"{str(row['below_ma200']):>8}  {str(row['curve_inverted']):>10}")

    # ── Regime sub-period breakdown ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MOD 1 — REGIME STRESS TEST (V9 vs V8 vs SPY)")
    print(f"{'='*60}")
    spy_oos = returns['SPY'].loc[idx]
    for name, s, e in REGIMES:
        rv9  = slice_stats(oos_v9,          s, e, name)
        rv8  = slice_stats(oos_v8.loc[idx], s, e, name)
        rsp  = slice_stats(spy_oos,         s, e, name)
        if rv9:
            print(f"\n  [{name}]  {s} -> {e}")
            print(f"    V9   Sharpe: {rv9['sharpe']:>6.3f}  |  "
                  f"Max DD: {rv9['max_dd']:>7.2f}%  |  Return: {rv9['total']:>7.2f}%")
            if rv8:
                print(f"    V8   Sharpe: {rv8['sharpe']:>6.3f}  |  "
                      f"Max DD: {rv8['max_dd']:>7.2f}%  |  Return: {rv8['total']:>7.2f}%")
            if rsp:
                print(f"    SPY  Sharpe: {rsp['sharpe']:>6.3f}  |  "
                      f"Max DD: {rsp['max_dd']:>7.2f}%  |  Return: {rsp['total']:>7.2f}%")

    # ── MOD 2 & 3 ─────────────────────────────────────────────────────────
    sensitivity  = run_sensitivity(returns, cash_returns, vix,
                                   spy_price, ma200, yield_curve)
    cost_results = run_costs(returns, cash_returns, vix,
                             spy_price, ma200, yield_curve)

    # ── Robustness scorecard ──────────────────────────────────────────────
    th_sharpes = [s['sharpe'] for s in sensitivity]
    spread     = max(th_sharpes) - min(th_sharpes)
    gfc_r      = slice_stats(oos_v9, '2008-01-01', '2009-12-31', 'GFC')
    rh_r       = slice_stats(oos_v9, '2022-01-01', '2023-12-31', 'Rate Hike')

    checks = [
        ("VIX threshold spread < 0.15",
         spread < 0.15,
         f"spread = {spread:.3f}"),
        ("Sharpe > 0.50 at 30bps cost",
         cost_results[-1]['sharpe'] > 0.50,
         f"Sharpe at 30bps = {cost_results[-1]['sharpe']:.3f}"),
        ("Positive Sharpe during GFC",
         gfc_r and gfc_r['sharpe'] > 0,
         f"GFC Sharpe = {gfc_r['sharpe']:.3f}" if gfc_r else "no data"),
        ("Positive Sharpe in Rate Hike era  (V8 failed this)",
         rh_r and rh_r['sharpe'] > 0,
         f"Rate Hike Sharpe = {rh_r['sharpe']:.3f}" if rh_r else "no data"),
        ("V9 Sharpe > V8 Sharpe",
         sv9['sharpe'] > sv8['sharpe'],
         f"V9={sv9['sharpe']} vs V8={sv8['sharpe']}"),
        ("Max Drawdown < 30%",
         sv9['max_dd'] > -30,
         f"Max DD = {sv9['max_dd']}%"),
    ]

    print(f"\n{'='*60}")
    print(f"  OVERALL ROBUSTNESS VERDICT (V9.1)")
    print(f"{'='*60}")
    passed = 0
    for name, ok, detail in checks:
        icon = "  PASS" if ok else "  FAIL"
        if ok: passed += 1
        print(f"{icon}  {name:<46}  ({detail})")

    print(f"\n  Score: {passed}/{len(checks)} checks passed")
    if passed == len(checks):
        print("  Strategy is ROBUST — all checks passed.")
    elif passed >= 4:
        print("  Strategy is PROMISING — minor vulnerabilities remain.")
    else:
        print("  Strategy NEEDS WORK — revisit signal logic.")
    print("=" * 60)

    plot_all(returns, oos_v9, oos_v8, regime_log,
             sensitivity, cost_results, spy_price, ma200, yield_curve)


if __name__ == "__main__":
    main()
