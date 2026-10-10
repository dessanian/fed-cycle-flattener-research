from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = ["DGS2", "DGS10", "DFF"]
CACHE_DIR = Path("data")
THRESHOLDS_BP = [0, 25, 50, 75, 100]


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
    df["flattener_gap"] = df["y2"] - df["ff_s"]
    return df


# ---------------------------------------------------------------------------
# Bond maths
# ---------------------------------------------------------------------------
def par_bond_price(coupon: pd.Series, yield_pct: pd.Series, years: int) -> pd.Series:
    y = np.maximum(yield_pct / 100.0, 1e-4)
    v = (1.0 + y / 2.0) ** (-2.0 * years)
    return coupon / y * (1.0 - v) + v


def mod_duration(y_pct: pd.Series, years: int) -> pd.Series:
    y = np.maximum(y_pct / 100.0, 1e-4)
    return (1.0 / y) * (1.0 - (1.0 + y / 2.0) ** (-2.0 * years))


def excess_returns(y_pct: pd.Series, cash_pct: pd.Series, years: int) -> pd.Series:
    y = np.maximum(y_pct / 100.0, 1e-4)
    c_prev = y.shift(1)
    dt = y.index.to_series().diff().dt.days / 365.0
    price_ret = par_bond_price(c_prev, y, years) - 1.0
    carry = c_prev * dt
    financing = (cash_pct / 100.0).shift(1) * dt
    return price_ret + carry - financing


def flattener_returns(df: pd.DataFrame):
    r2 = excess_returns(df["y2"], df["ff_s"], 2)
    r10 = excess_returns(df["y10"], df["ff_s"], 10)
    n2_lag = mod_duration(df["y10"].shift(1), 10) / mod_duration(df["y2"].shift(1), 2)
    flat_ret = r10 - n2_lag * r2
    return flat_ret, n2_lag


# ---------------------------------------------------------------------------
# Strategy backtests
# ---------------------------------------------------------------------------
def backtest_binary_threshold(df: pd.DataFrame, threshold_bp: float, cost10_bps: float = 1.0, cost2_bps: float = 0.3) -> pd.DataFrame:
    flat_ret, n2_lag = flattener_returns(df)
    signal = (df["flattener_gap"] > threshold_bp / 10000.0).astype(float)
    position = (-signal.shift(1).fillna(0.0)).astype(float)
    exposure = -position
    cost = position.diff().abs().fillna(0.0) * (cost10_bps + cost2_bps * n2_lag.shift(1)) / 1e4
    strategy_ret = exposure * flat_ret - cost
    out = pd.DataFrame(
        {
            "strategy_ret": strategy_ret,
            "flat_ret": flat_ret,
            "position": position,
            "exposure": exposure,
            "gap": df["flattener_gap"],
            "signal": signal,
            "y2": df["y2"],
            "y10": df["y10"],
            "ff_s": df["ff_s"],
        }
    ).dropna()
    return out


def metrics(ret: pd.Series, pos: pd.Series | None = None) -> dict:
    ret = ret.dropna()
    if ret.empty:
        return {
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe": np.nan,
            "MaxDD": np.nan,
            "Calmar": np.nan,
            "TradingDays": 0,
            "PctInvested": np.nan,
            "t_stat": np.nan,
        }

    cumprod = (1.0 + ret).cumprod()
    total_ret = cumprod.iloc[-1] - 1.0
    cagr = (1.0 + total_ret) ** (252.0 / len(ret)) - 1.0

    vol = ret.std(ddof=0) * np.sqrt(252.0)
    sharpe = (ret.mean() * 252.0) / vol if vol > 0 else np.nan

    dd = cumprod / cumprod.cummax() - 1.0
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    pct_invested = (pos != 0).mean() * 100.0 if pos is not None else np.nan
    t_stat = sharpe * np.sqrt(len(ret) / 252.0) if np.isfinite(sharpe) else np.nan

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Calmar": calmar,
        "TradingDays": len(ret),
        "PctInvested": pct_invested,
        "t_stat": t_stat,
    }


def date_mask(df: pd.DataFrame, start: str | None = None, end: str | None = None, exclude_ranges=None) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df.index >= start
    if end is not None:
        mask &= df.index <= end
    if exclude_ranges is not None:
        for s, e in exclude_ranges:
            mask &= ~((df.index >= s) & (df.index <= e))
    return mask


# ---------------------------------------------------------------------------
# Period/robustness outputs
# ---------------------------------------------------------------------------
def threshold_results(df: pd.DataFrame):
    periods = [
        ("Full sample", None, None, []),
        ("Since 1994", "1994-02-04", None, []),
        ("Since 2000", "2000-01-01", None, []),
    ]
    robustness = {
        "Exclude 2009-2015": [("2009-01-01", "2015-12-31")],
        "Exclude 2020-2022": [("2020-01-01", "2022-12-31")],
        "Exclude both": [("2009-01-01", "2015-12-31"), ("2020-01-01", "2022-12-31")],
    }

    output = {}
    for period_label, start, end, _ in periods:
        output[period_label] = {}
        for thr in THRESHOLDS_BP:
            sub = df.loc[date_mask(df, start=start, end=end)]
            out = backtest_binary_threshold(sub, thr)
            output[period_label][thr] = metrics(out["strategy_ret"], out["position"])
        for rob_label, ranges in robustness.items():
            label = f"{period_label} | {rob_label}"
            output[label] = {}
            for thr in THRESHOLDS_BP:
                sub = df.loc[date_mask(df, start=start, end=end, exclude_ranges=ranges)]
                out = backtest_binary_threshold(sub, thr)
                output[label][thr] = metrics(out["strategy_ret"], out["position"])

    return output


def print_threshold_table(output: dict):
    print("\n=== Strict binary flattener threshold results ===")
    print("Signal rule: if (DGS2 - DFF) > threshold then position = -1 else 0")
    print("Long 10Y / short 2Y DV01-neutral flattener return, same cost assumptions as original model\n")

    for label, by_threshold in output.items():
        print(f"--- {label} ---")
        rows = []
        for thr in THRESHOLDS_BP:
            m = by_threshold[thr]
            rows.append(
                {
                    "Threshold bp": thr,
                    "CAGR": m["CAGR"],
                    "Volatility": m["Volatility"],
                    "Sharpe": m["Sharpe"],
                    "MaxDD": m["MaxDD"],
                    "Calmar": m["Calmar"],
                    "TradingDays": m["TradingDays"],
                    "PctInvested": m["PctInvested"],
                    "t_stat": m["t_stat"],
                }
            )
        tbl = pd.DataFrame(rows).set_index("Threshold bp")
        tbl["CAGR"] = tbl["CAGR"] * 100.0
        tbl["Volatility"] = tbl["Volatility"] * 100.0
        tbl["MaxDD"] = tbl["MaxDD"] * 100.0
        print(tbl.round(3).to_string())
        print()


# ---------------------------------------------------------------------------
# Marginal contribution by gap bucket
# ---------------------------------------------------------------------------
def bucket_contribution(df: pd.DataFrame):
    flat_ret, _ = flattener_returns(df)
    gap = df["flattener_gap"]
    positive_gap = gap > 0

    bucket_defs = [
        ("0-25 bp", lambda g: (g > 0.0 / 10000.0) & (g <= 25.0 / 10000.0)),
        ("25-50 bp", lambda g: (g > 25.0 / 10000.0) & (g <= 50.0 / 10000.0)),
        ("50-75 bp", lambda g: (g > 50.0 / 10000.0) & (g <= 75.0 / 10000.0)),
        ("75-100 bp", lambda g: (g > 75.0 / 10000.0) & (g <= 100.0 / 10000.0)),
        ("100 bp+", lambda g: (g > 100.0 / 10000.0)),
    ]

    bucket_rows = []
    bucket_rets = []
    for name, cond_fn in bucket_defs:
        mask = cond_fn(gap) & positive_gap
        if not mask.any():
            continue
        ret = flat_ret[mask].dropna()
        avg_daily = ret.mean()
        annualized = (1.0 + ret).prod() ** (252.0 / len(ret)) - 1.0 if len(ret) > 0 else np.nan
        vol = ret.std(ddof=0) * np.sqrt(252.0)
        sharpe = (ret.mean() * 252.0) / vol if vol > 0 else np.nan
        cumul = (1.0 + ret).prod() - 1.0 if len(ret) > 0 else np.nan
        bucket_rows.append(
            {
                "Bucket": name,
                "Average daily return": avg_daily,
                "Annualized return": annualized,
                "Sharpe": sharpe,
                "Cumulative contribution": cumul,
            }
        )
        bucket_rets.append(ret)

    tbl = pd.DataFrame(bucket_rows)
    if tbl.empty:
        return tbl

    total_pnl = pd.concat(bucket_rets) if bucket_rets else pd.Series(dtype=float)
    total_pnl_cum = (1.0 + total_pnl).prod() - 1.0 if len(total_pnl) > 0 else np.nan
    if np.isfinite(total_pnl_cum):
        tbl["% of total P&L"] = tbl["Cumulative contribution"] / total_pnl_cum * 100.0
    else:
        tbl["% of total P&L"] = np.nan
    return tbl


def print_bucket_table(df: pd.DataFrame):
    print("\n=== Marginal contribution by signal-strength bucket ===")
    tbl = bucket_contribution(df)
    if tbl.empty:
        print("No positive-gap days in the sample.")
        return

    tbl["Average daily return"] = tbl["Average daily return"] * 100.0
    tbl["Annualized return"] = tbl["Annualized return"] * 100.0
    tbl["Cumulative contribution"] = tbl["Cumulative contribution"] * 100.0
    tbl["% of total P&L"] = tbl["% of total P&L"]
    print(tbl.round(3).to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def save_plot(fig, path: str):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_returns_comparison(df: pd.DataFrame):
    """Compare cumulative returns for gap > 0 bp and gap > 25 bp."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    thresholds = [0, 25]
    colors = ["#2E86AB", "#A23B72"]

    for ax, threshold, color in zip(axes, thresholds, colors):
        out = backtest_binary_threshold(df, threshold)
        out = out.loc["1981-07-14":]

        eq = (1.0 + out["strategy_ret"]).cumprod()
        ax.plot(eq.index, eq.values, linewidth=2.2, color=color, label=f"Gap > {threshold} bp")
        ax.fill_between(eq.index, eq.values, eq.cummax(), color=color, alpha=0.12)

        m = metrics(out["strategy_ret"], out["position"])
        text = (
            f"CAGR: {m['CAGR'] * 100:.2f}%\n"
            f"Sharpe: {m['Sharpe']:.2f}\n"
            f"Vol: {m['Volatility'] * 100:.2f}%\n"
            f"MaxDD: {m['MaxDD'] * 100:.2f}%\n"
            f"Calmar: {m['Calmar']:.2f}\n"
            f"Days: {m['TradingDays']}\n"
            f"% Invested: {m['PctInvested']:.1f}%"
        )
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
            family="monospace",
        )

        ax.set_yscale("log")
        ax.set_title(f"Gap > {threshold} bp", fontsize=13, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Growth of $1 (log scale)", fontsize=11)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig("cumulative_returns_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved: cumulative_returns_comparison.png")


def plot_threshold_summary(output: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    labels = [
        "Full sample",
        "Since 1994",
        "Since 2000",
        "Full sample | Exclude both",
    ]

    for ax, label in zip(axes, labels):
        if label not in output:
            continue
        arr = []
        for thr in THRESHOLDS_BP:
            m = output[label][thr]
            arr.append({"thr": thr, "CAGR": m["CAGR"], "Sharpe": m["Sharpe"], "MaxDD": m["MaxDD"]})
        arr = pd.DataFrame(arr)
        if arr.empty:
            continue
        ax.plot(arr["thr"], arr["CAGR"] * 100.0, marker="o", label="CAGR", linewidth=2)
        ax.plot(arr["thr"], arr["Sharpe"], marker="s", label="Sharpe", linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(arr["thr"], arr["MaxDD"] * 100.0, marker="^", color="C2", label="MaxDD", linewidth=2)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Threshold (bp)")
        ax.set_ylabel("CAGR % / Sharpe")
        ax2.set_ylabel("MaxDD %")
        ax.grid(True, alpha=0.25)

    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[0].get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=3)
    fig.suptitle("Threshold Performance Summary", fontsize=14, fontweight="bold", y=0.995)
    save_plot(fig, "threshold_summary.png")
    print("Saved: threshold_summary.png")


def plot_bucket_contribution(df: pd.DataFrame):
    tbl = bucket_contribution(df)
    if tbl.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(tbl["Bucket"], tbl["Cumulative contribution"] * 100.0, 
                  color=["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974"], edgecolor="black", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Cumulative Contribution by Gap-Strength Bucket", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Contribution (%)", fontsize=11)
    ax.set_xlabel("Gap Bucket", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    for b, v in zip(bars, tbl["Cumulative contribution"] * 100.0):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5 if v > 0 else v - 1.5, 
                f"{v:.1f}%", ha="center", va="bottom" if v > 0 else "top", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "bucket_contribution.png")
    print("Saved: bucket_contribution.png")


def plot_bucket_sharpe(df: pd.DataFrame):
    tbl = bucket_contribution(df)
    if tbl.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(tbl["Bucket"], tbl["Sharpe"], 
                  color=["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974"], edgecolor="black", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Sharpe Ratio by Gap-Strength Bucket", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe", fontsize=11)
    ax.set_xlabel("Gap Bucket", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    for b, v in zip(bars, tbl["Sharpe"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05 if v > 0 else v - 0.05, 
                f"{v:.2f}", ha="center", va="bottom" if v > 0 else "top", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "bucket_sharpe.png")
    print("Saved: bucket_sharpe.png")


def plot_threshold_regime_lines(df: pd.DataFrame):
    labels = [
        ("Full sample", None, None, []),
        ("Since 1994", "1994-02-04", None, []),
        ("Since 2000", "2000-01-01", None, []),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (label, start, end, exclude) in zip(axes, labels):
        sub = df.loc[date_mask(df, start=start, end=end, exclude_ranges=exclude)]
        rows = []
        for thr in THRESHOLDS_BP:
            out = backtest_binary_threshold(sub, thr)
            m = metrics(out["strategy_ret"], out["position"])
            rows.append({"thr": thr, "Sharpe": m["Sharpe"], "CAGR": m["CAGR"] * 100.0, "MaxDD": m["MaxDD"] * 100.0})
        rows = pd.DataFrame(rows)
        ax.plot(rows["thr"], rows["Sharpe"], marker="o", label="Sharpe", linewidth=2, markersize=8)
        ax2 = ax.twinx()
        ax2.plot(rows["thr"], rows["CAGR"], marker="s", color="C1", label="CAGR", linewidth=2, markersize=8)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Threshold (bp)", fontsize=10)
        ax.set_ylabel("Sharpe", fontsize=10)
        ax2.set_ylabel("CAGR %", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xticks(THRESHOLDS_BP)
    fig.suptitle("Performance Across Thresholds by Period", fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    save_plot(fig, "threshold_regime_lines.png")
    print("Saved: threshold_regime_lines.png")


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
def interpret_edge(df: pd.DataFrame):
    tbl = bucket_contribution(df)
    if tbl.empty:
        return
    strong = tbl[tbl["Bucket"].isin(["25-50 bp", "50-75 bp", "75-100 bp", "100 bp+"])]["% of total P&L"].sum()
    weak = tbl[tbl["Bucket"] == "0-25 bp"]["% of total P&L"].sum() if (tbl["Bucket"] == "0-25 bp").any() else 0.0
    print("\n=== Economic interpretation ===")
    print(f"Strong-gap buckets (25-100 bp and above) account for {strong:.1f}% of total P&L.")
    print(f"Weak-gap bucket (0-25 bp) accounts for {weak:.1f}% of total P&L.")
    if strong > weak:
        print("Interpretation: the return is concentrated in strong-gap environments rather than the broad flattener signal.")
    else:
        print("Interpretation: weaker gap states are still economically important, so the edge is not purely concentrated in strong gaps.")
    print("The economically robust conclusion is the one that survives the ZIRP and pandemic exclusions without being driven by a single threshold.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    raw = {sid: load_fred(sid) for sid in SERIES}
    df = prepare(raw)

    output = threshold_results(df)
    print_threshold_table(output)
    print_bucket_table(df)

    print("\n=== Generating plots ===")
    plot_cumulative_returns_comparison(df)
    plot_threshold_summary(output)
    plot_bucket_contribution(df)
    plot_bucket_sharpe(df)
    plot_threshold_regime_lines(df)

    interpret_edge(df)


if __name__ == "__main__":
    main()
