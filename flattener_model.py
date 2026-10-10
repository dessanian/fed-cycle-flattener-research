from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

SERIES = ["DGS2", "DGS10", "DFF"]
CACHE_DIR = Path("data")
EVAL_START = "1981-07-14"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_fred(series_id: str) -> pd.Series:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{series_id}.csv"
    if not path.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        pd.read_csv(url).to_csv(path, index=False)
    df = pd.read_csv(path, na_values=".")
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    idx = pd.to_datetime(df.iloc[:, 0])
    return pd.Series(vals.values, index=idx, name=series_id).dropna()


def to_daily(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def prepare(raw: dict) -> pd.DataFrame:
    idx = raw["DGS10"].index.intersection(raw["DGS2"].index)
    idx = idx[idx >= "1980-01-01"]
    df = pd.DataFrame(
        {
            "y2": raw["DGS2"].reindex(idx),
            "y10": raw["DGS10"].reindex(idx),
            "ff": to_daily(raw["DFF"], idx),
        }
    ).dropna()
    df["ff_s"] = df["ff"].rolling(5, min_periods=1).median()
    df["gap"] = df["y2"] - df["ff_s"]
    return df


# ---------------------------------------------------------------------------
# Bond maths (from original model)
# ---------------------------------------------------------------------------
def par_bond_price(c, y, n):
    """Price per 1 face of an n-year semiannual bond, coupon c, yield y (decimals)."""
    v = (1 + y / 2) ** (-2 * n)
    return c / y * (1 - v) + v


def mod_duration(y_pct, n):
    y = np.maximum(y_pct / 100, 1e-4)
    return (1 / y) * (1 - (1 + y / 2) ** (-2 * n))


def excess_returns(y_pct: pd.Series, cash_pct: pd.Series, n: int) -> pd.Series:
    """Daily excess return (over cash) of a constant-maturity n-year bond."""
    y = np.maximum(y_pct / 100, 1e-4)
    c_prev = y.shift(1)
    dt = y.index.to_series().diff().dt.days / 365.0
    price_ret = par_bond_price(c_prev, y, n) - 1
    carry = c_prev * dt
    financing = (cash_pct / 100).shift(1) * dt
    return price_ret + carry - financing


def flattener_returns(df: pd.DataFrame):
    """Return of a DV01-neutral FLATTENER (long 10y, short 2y) per 1 unit of 10y notional."""
    r2 = excess_returns(df["y2"], df["ff_s"], 2)
    r10 = excess_returns(df["y10"], df["ff_s"], 10)
    n2_lag = mod_duration(df["y10"].shift(1), 10) / mod_duration(df["y2"].shift(1), 2)
    return r10 - n2_lag * r2, n2_lag


# ---------------------------------------------------------------------------
# Strategy backtest (matching original model exactly)
# ---------------------------------------------------------------------------
def backtest(
    df: pd.DataFrame,
    min_gap_pp: float = 0.0,
    target_vol: float = 0.10,
    vol_window: int = 60,
    max_lev: float = 5.0,
    rebal_days: int = 5,
    cost10_bps: float = 1.0,
    cost2_bps: float = 0.3,
) -> pd.DataFrame:
    """
    Backtest matching the original flattener_model.py logic exactly.
    Returns dataframe with strategy returns, positions, and diagnostic info.
    """
    flat_ret, n2_lag = flattener_returns(df)
    signal = (df["gap"] > min_gap_pp).astype(float)
    vol = flat_ret.rolling(vol_window).std() * np.sqrt(252)
    w = (signal * target_vol / vol).clip(0, max_lev)
    w = w.iloc[::rebal_days].reindex(w.index).ffill()
    pos = w.shift(1)
    cost = pos.diff().abs() * (cost10_bps + cost2_bps * n2_lag.shift(1)) / 1e4
    strat = pos * flat_ret - cost
    out = pd.DataFrame(
        {
            "strategy": strat,
            "position": pos,
            "signal": signal,
            "gap": df["gap"],
            "y2": df["y2"],
            "y10": df["y10"],
            "ff_s": df["ff_s"],
            "flat_ret": flat_ret,
            "vol": vol,
            "cost": cost,
            "n2_lag": n2_lag,
        }
    ).dropna(subset=["strategy"])
    return out.loc[EVAL_START:]


def metrics(r: pd.Series) -> dict:
    """Calculate performance metrics from return series."""
    r = r.dropna()
    if len(r) == 0:
        return {
            "CAGR %": np.nan,
            "Vol %": np.nan,
            "Sharpe": np.nan,
            "MaxDD %": np.nan,
            "Calmar": np.nan,
        }

    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1

    sharpe = ann / vol if vol > 0 else np.nan
    max_dd = dd.min()
    calmar = ann / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "CAGR %": ann * 100,
        "Vol %": vol * 100,
        "Sharpe": sharpe,
        "MaxDD %": max_dd * 100,
        "Calmar": calmar,
    }


def strategy_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe comparing 0 bp and 25 bp threshold strategies."""
    out0 = backtest(df, min_gap_pp=0.0)
    out25 = backtest(df, min_gap_pp=0.25)

    rows = []
    for label, out in [("Gap > 0 bp (base)", out0), ("Gap > 25 bp (threshold)", out25)]:
        m = metrics(out["strategy"])
        rows.append(
            {
                "Strategy": label,
                "CAGR %": m["CAGR %"],
                "Vol %": m["Vol %"],
                "Sharpe": m["Sharpe"],
                "MaxDD %": m["MaxDD %"],
                "Calmar": m["Calmar"],
                "Trading Days": len(out),
                "% Days Invested": (out["position"] > 0).mean() * 100,
                "Final Equity": (1 + out["strategy"]).cumprod().iloc[-1],
            }
        )

    table = pd.DataFrame(rows).set_index("Strategy")
    table = table.round({
        "CAGR %": 2,
        "Vol %": 2,
        "Sharpe": 2,
        "MaxDD %": 2,
        "Calmar": 2,
        "% Days Invested": 1,
        "Final Equity": 2,
    })
    return table


def print_metrics_table(df: pd.DataFrame):
    table = strategy_metrics_table(df)
    print("\n=== Strategy Metrics Table ===")
    print(table.to_string())
    return table


# ---------------------------------------------------------------------------
# Save trade-level CSV
# ---------------------------------------------------------------------------
def save_trades_csv(out: pd.DataFrame, threshold_bp: float, filename: str, append: bool = False):
    """Export trade-level records to a CSV. Append rows if requested."""
    export = out[["gap", "y2", "y10", "ff_s", "position", "flat_ret", "strategy", "cost"]].copy()
    export.columns = [
        "gap_pp",
        "y2_yield_%",
        "y10_yield_%",
        "ff_s_%",
        "position_size",
        "flattener_ret",
        "strategy_ret",
        "cost_bps",
    ]
    export["threshold_bp"] = threshold_bp
    export["strategy_name"] = f"gap_gt_{threshold_bp}_bp"
    export["in_market"] = (export["position_size"] > 0).astype(int)

    if append and os.path.exists(filename):
        existing = pd.read_csv(filename)
        combined = pd.concat([existing, export], ignore_index=True)
        combined.to_csv(filename, index=False)
    else:
        export.to_csv(filename, index=False)

    print(f"Saved: {filename}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_strategy_comparison(df: pd.DataFrame):
    """Create side-by-side comparison of gap > 0 bp vs gap > 25 bp."""
    out0 = backtest(df, min_gap_pp=0.0)
    out25 = backtest(df, min_gap_pp=0.25)

    eq0 = (1 + out0["strategy"]).cumprod()
    eq25 = (1 + out25["strategy"]).cumprod()
    dd0 = eq0 / eq0.cummax() - 1
    dd25 = eq25 / eq25.cummax() - 1

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.30)
    ax_eq = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)

    SURF = "#fcfcfb"
    INK = "#0b0b0b"
    INK2 = "#52514e"
    MUTED = "#898781"
    GRID = "#e1e0d9"
    AXIS = "#c3c2b7"
    C_BASE = "#2a78d6"
    C_THRESHOLD = "#eb6834"

    fig.set_facecolor(SURF)
    for ax in [ax_eq, ax_dd]:
        ax.set_facecolor(SURF)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.grid(axis="y", color=GRID, linewidth=0.5, linestyle="-")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, labelsize=9, labelcolor=MUTED)

    ax_eq.plot(eq0.index, eq0.values, color=C_BASE, linewidth=2.0, label="Gap > 0 bp (base)", solid_capstyle="round")
    ax_eq.plot(eq25.index, eq25.values, color=C_THRESHOLD, linewidth=2.0, label="Gap > 25 bp (threshold)", solid_capstyle="round")

    ax_eq.scatter([eq0.index[-1]], [eq0.iloc[-1]], s=22, color=C_BASE, edgecolor=SURF, linewidth=1.0, zorder=5, clip_on=False)
    ax_eq.scatter([eq25.index[-1]], [eq25.iloc[-1]], s=22, color=C_THRESHOLD, edgecolor=SURF, linewidth=1.0, zorder=5, clip_on=False)
    ax_eq.annotate(f"{eq0.iloc[-1]:.2f}x", (eq0.index[-1], eq0.iloc[-1]), xytext=(9, 0), textcoords="offset points",
                   va="center", fontsize=10, color=INK, annotation_clip=False)
    ax_eq.annotate(f"{eq25.iloc[-1]:.2f}x", (eq25.index[-1], eq25.iloc[-1]), xytext=(9, 6), textcoords="offset points",
                   va="center", fontsize=10, color=INK, annotation_clip=False)

    ax_eq.set_yscale("log")
    lo = min(eq0.min(), eq25.min()) * 0.85
    hi = max(eq0.max(), eq25.max()) * 1.2
    ax_eq.set_ylim(lo, hi)
    ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8, 16, 32) if lo <= t <= hi]
    if len(ticks) >= 2:
        ax_eq.set_yticks(ticks)
        ax_eq.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))

    ax_eq.set_title("2s10s DV01-neutral flattener: Gap > 0 bp vs Gap > 25 bp", loc="left", fontsize=12, color=INK2, fontweight="bold", pad=8)
    ax_eq.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK2, handlelength=1.8)
    plt.setp(ax_eq.get_xticklabels(), visible=False)

    ax_dd.plot(dd0.index, dd0.values * 100, color=C_BASE, linewidth=2.0, solid_capstyle="round")
    ax_dd.plot(dd25.index, dd25.values * 100, color=C_THRESHOLD, linewidth=2.0, solid_capstyle="round")
    ax_dd.axhline(0, color=AXIS, lw=0.5)
    ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0%" if v == 0 else f"−{abs(v):.0f}%"))
    ax_dd.set_title("Drawdown from prior high", loc="left", fontsize=10, color=INK2, fontweight="bold", pad=8)
    ax_dd.set_ylabel("Drawdown", fontsize=9, color=MUTED)

    ax_dd.xaxis.set_major_locator(mdates.YearLocator(5))
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_dd.set_xlim(eq0.index[0], eq0.index[-1])
    plt.setp(ax_dd.get_xticklabels(), rotation=45, ha="right")

    fig.text(0.075, 0.955, "Fed-cycle flattener: 0 bp vs 25 bp threshold comparison",
             fontsize=14, color=INK, fontweight="bold", va="top")
    fig.text(0.075, 0.918,
             f"DV01-neutral long 10Y / short 2Y. Signal: 2Y > Fed Funds (+ threshold). Vol-targeted 10%. {EVAL_START[:4]}–2026.\n"
             "Excess returns over cash, after costs. Not a recommendation to trade.",
             fontsize=9.5, color=INK2, va="top", linespacing=1.5)

    fig.savefig("flattener_comparison.png", dpi=180, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print("Saved: flattener_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    raw = {sid: load_fred(sid) for sid in SERIES}
    df = prepare(raw)

    out_base = backtest(df, min_gap_pp=0.0)
    out_25 = backtest(df, min_gap_pp=0.25)

    save_trades_csv(out_base, threshold_bp=0, filename="flattener_trades.csv", append=False)
    save_trades_csv(out_25, threshold_bp=25, filename="flattener_trades.csv", append=True)

    print_metrics_table(df)
    plot_strategy_comparison(df)


if __name__ == "__main__":
    main()
