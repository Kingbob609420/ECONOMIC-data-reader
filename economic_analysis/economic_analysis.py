"""
Economic Data Analysis
======================
Analyzes US inflation and unemployment trends and their relationship
using data from the Federal Reserve Economic Database (FRED).

Data sources (via pandas_datareader):
  CPIAUCSL  - Consumer Price Index, All Urban Consumers
  CPILFESL  - Core CPI (ex. food & energy)
  UNRATE    - Civilian Unemployment Rate
  FEDFUNDS  - Federal Funds Rate (context)
"""

import sys
import os
from pathlib import Path

# Force UTF-8 output so Unicode characters don't crash on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
import pandas_datareader.data as web
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Output folder is always next to this script, regardless of where you run it from
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────
START = "1960-01-01"
END   = "2025-01-01"

SERIES = {
    "CPI":      "CPIAUCSL",   # headline CPI
    "CORE_CPI": "CPILFESL",   # core CPI
    "UNRATE":   "UNRATE",     # unemployment rate
    "FEDFUNDS": "FEDFUNDS",   # fed funds rate
}

STYLE = {
    "inflation":    "#d62728",
    "core_infl":    "#ff7f0e",
    "unemployment": "#1f77b4",
    "fedfunds":     "#2ca02c",
    "scatter":      "#9467bd",
    "recession":    "#d0d0d0",
}

# NBER recession start/end dates (approximate)
RECESSIONS = [
    ("1960-04", "1961-02"), ("1969-12", "1970-11"), ("1973-11", "1975-03"),
    ("1980-01", "1980-07"), ("1981-07", "1982-11"), ("1990-07", "1991-03"),
    ("2001-03", "2001-11"), ("2007-12", "2009-06"), ("2020-02", "2020-04"),
]

# ── Data Loading ─────────────────────────────────────────────────────────────
def load_data(start: str, end: str) -> pd.DataFrame:
    """Download series from FRED and build a monthly analysis dataframe."""
    print("Downloading data from FRED...")
    raw = {}
    for name, symbol in SERIES.items():
        try:
            raw[name] = web.DataReader(symbol, "fred", start, end)[symbol]
            print(f"  {symbol} ({name}): {len(raw[name])} observations")
        except Exception as e:
            print(f"  WARNING: could not load {symbol}: {e}")

    df = pd.DataFrame(raw).resample("MS").last()  # month-start frequency

    # Year-over-year inflation rates
    df["inflation"]  = df["CPI"].pct_change(12) * 100
    df["core_infl"]  = df["CORE_CPI"].pct_change(12) * 100

    df.dropna(subset=["inflation", "UNRATE"], inplace=True)
    print(f"\nFinal dataset: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} months\n")
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────
def shade_recessions(ax: plt.Axes, start: str, end: str) -> None:
    """Shade NBER recession periods on an axes."""
    for rs, re in RECESSIONS:
        rs_dt = pd.to_datetime(rs)
        re_dt = pd.to_datetime(re)
        if rs_dt >= pd.to_datetime(start) and re_dt <= pd.to_datetime(end):
            ax.axvspan(rs_dt, re_dt, color=STYLE["recession"], alpha=0.5, lw=0)


def period_stats(df: pd.DataFrame, col: str) -> dict:
    return {
        "mean":  df[col].mean(),
        "max":   df[col].max(),
        "min":   df[col].min(),
        "std":   df[col].std(),
        "max_date": df[col].idxmax().strftime("%Y-%m"),
        "min_date": df[col].idxmin().strftime("%Y-%m"),
    }


# ── Plot 1: Inflation Over Time ───────────────────────────────────────────────
def plot_inflation_trends(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    shade_recessions(ax, START, END)

    ax.plot(df.index, df["inflation"],  color=STYLE["inflation"],  lw=1.5,
            label="Headline Inflation (YoY %)")
    ax.plot(df.index, df["core_infl"], color=STYLE["core_infl"], lw=1.5,
            ls="--", label="Core Inflation (YoY %)")
    ax.axhline(2, color="black", lw=0.8, ls=":", label="2% Fed Target")
    ax.axhline(0, color="black", lw=0.5)

    ax.set_title("US Inflation Over Time (1960–2025)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Year-over-Year Change (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # Annotate major events
    annotations = {
        "1973-11": ("Oil Crisis", 10),
        "1979-06": ("Volcker Era", 14),
        "2022-06": ("Post-COVID\nInflation", 9),
    }
    for date_str, (label, y_offset) in annotations.items():
        dt = pd.to_datetime(date_str)
        if dt in df.index:
            ax.annotate(label, xy=(dt, df.loc[dt, "inflation"]),
                        xytext=(dt, df.loc[dt, "inflation"] + y_offset),
                        fontsize=7, ha="center", color="gray",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "inflation_trends.png", dpi=150)
    plt.close()
    print("Saved: inflation_trends.png")


# ── Plot 2: Unemployment Over Time ────────────────────────────────────────────
def plot_unemployment_trends(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    shade_recessions(ax, START, END)

    ax.fill_between(df.index, df["UNRATE"], alpha=0.25, color=STYLE["unemployment"])
    ax.plot(df.index, df["UNRATE"], color=STYLE["unemployment"], lw=1.5,
            label="Unemployment Rate")

    # Rolling average
    roll = df["UNRATE"].rolling(24).mean()
    ax.plot(df.index, roll, color="navy", lw=1, ls="--", label="24-month Moving Avg")

    ax.set_title("US Unemployment Rate Over Time (1960–2025)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "unemployment_trends.png", dpi=150)
    plt.close()
    print("Saved: unemployment_trends.png")


# ── Plot 3: Combined Dashboard ────────────────────────────────────────────────
def plot_combined_dashboard(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4)

    # Panel 1: Inflation
    ax1 = fig.add_subplot(gs[0])
    shade_recessions(ax1, START, END)
    ax1.plot(df.index, df["inflation"], color=STYLE["inflation"], lw=1.4, label="Inflation (YoY %)")
    ax1.axhline(2, color="black", lw=0.7, ls=":")
    ax1.set_title("US Macroeconomic Dashboard (1960–2025)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Inflation (%)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    # Panel 2: Unemployment
    ax2 = fig.add_subplot(gs[1])
    shade_recessions(ax2, START, END)
    ax2.fill_between(df.index, df["UNRATE"], alpha=0.2, color=STYLE["unemployment"])
    ax2.plot(df.index, df["UNRATE"], color=STYLE["unemployment"], lw=1.4, label="Unemployment (%)")
    ax2.set_ylabel("Unemployment (%)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.25)

    # Panel 3: Fed Funds Rate (context)
    ax3 = fig.add_subplot(gs[2])
    shade_recessions(ax3, START, END)
    ax3.plot(df.index, df["FEDFUNDS"], color=STYLE["fedfunds"], lw=1.4, label="Fed Funds Rate (%)")
    ax3.set_ylabel("Fed Funds Rate (%)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(axis="y", alpha=0.25)

    # Shared x-label
    ax3.set_xlabel("Year")

    # Grey box label for recessions
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=STYLE["recession"], label="NBER Recession")]
    fig.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    fig.savefig(OUTPUT_DIR / "macro_dashboard.png", dpi=150)
    plt.close()
    print("Saved: macro_dashboard.png")


# ── Plot 4: Phillips Curve ────────────────────────────────────────────────────
def plot_phillips_curve(df: pd.DataFrame) -> None:
    """
    The Phillips Curve: inverse relationship between inflation & unemployment.
    We split into eras to show how the relationship has shifted over time.
    """
    eras = {
        "1960s–70s": (df["1960":"1979"], "#e41a1c"),
        "1980s–90s": (df["1980":"1999"], "#377eb8"),
        "2000s–10s": (df["2000":"2019"], "#4daf4a"),
        "2020s":     (df["2020":],       "#ff7f00"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Phillips Curve: Inflation vs. Unemployment", fontsize=14, fontweight="bold")

    # Left: full period
    ax = axes[0]
    scatter = ax.scatter(df["UNRATE"], df["inflation"],
                         c=df.index.year, cmap="plasma", alpha=0.5, s=12)
    plt.colorbar(scatter, ax=ax, label="Year")

    # OLS trend
    slope, intercept, r, p, _ = stats.linregress(df["UNRATE"], df["inflation"])
    x_range = np.linspace(df["UNRATE"].min(), df["UNRATE"].max(), 100)
    ax.plot(x_range, slope * x_range + intercept, color="black", lw=1.5,
            label=f"OLS fit  r={r:.2f}")
    ax.set_xlabel("Unemployment Rate (%)")
    ax.set_ylabel("Inflation Rate (YoY %)")
    ax.set_title(f"Full Period (1960–2025)\nr = {r:.2f}, p = {p:.3f}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axhline(2, color="gray", lw=0.5, ls=":")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Right: by era
    ax2 = axes[1]
    for era_label, (era_df, color) in eras.items():
        if len(era_df) < 5:
            continue
        ax2.scatter(era_df["UNRATE"], era_df["inflation"],
                    color=color, alpha=0.6, s=14, label=era_label)
        sl, ic, rv, *_ = stats.linregress(era_df["UNRATE"], era_df["inflation"])
        xr = np.linspace(era_df["UNRATE"].min(), era_df["UNRATE"].max(), 50)
        ax2.plot(xr, sl * xr + ic, color=color, lw=1.5)

    ax2.set_xlabel("Unemployment Rate (%)")
    ax2.set_ylabel("Inflation Rate (YoY %)")
    ax2.set_title("By Era (shifting Phillips Curve)")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.axhline(2, color="gray", lw=0.5, ls=":")
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phillips_curve.png", dpi=150)
    plt.close()
    print("Saved: phillips_curve.png")


# ── Plot 5: Decade Summaries ──────────────────────────────────────────────────
def plot_decade_summary(df: pd.DataFrame) -> None:
    df2 = df.copy()
    df2["decade"] = (df2.index.year // 10 * 10).astype(str) + "s"
    summary = df2.groupby("decade")[["inflation", "UNRATE"]].mean().reset_index()

    x = np.arange(len(summary))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width/2, summary["inflation"],  width, label="Avg Inflation (%)",
                   color=STYLE["inflation"], alpha=0.85)
    bars2 = ax.bar(x + width/2, summary["UNRATE"],     width, label="Avg Unemployment (%)",
                   color=STYLE["unemployment"], alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["decade"])
    ax.set_ylabel("Average Rate (%)")
    ax.set_title("Average Inflation & Unemployment by Decade", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "decade_summary.png", dpi=150)
    plt.close()
    print("Saved: decade_summary.png")


# ── Statistical Report ────────────────────────────────────────────────────────
def print_report(df: pd.DataFrame) -> None:
    infl_s  = period_stats(df, "inflation")
    unemp_s = period_stats(df, "UNRATE")
    corr_all = df["inflation"].corr(df["UNRATE"])

    # Era correlations
    eras = {
        "1960–1979": df["1960":"1979"],
        "1980–1999": df["1980":"1999"],
        "2000–2019": df["2000":"2019"],
        "2020–2025": df["2020":],
    }

    report = f"""
============================================================
         US ECONOMIC DATA ANALYSIS REPORT
         {START[:4]} - {END[:4]}
============================================================

INFLATION (CPI Year-over-Year)
  Mean:    {infl_s['mean']:6.2f}%
  Std Dev: {infl_s['std']:6.2f}%
  Peak:    {infl_s['max']:6.2f}%  ({infl_s['max_date']})
  Trough:  {infl_s['min']:6.2f}%  ({infl_s['min_date']})

UNEMPLOYMENT RATE
  Mean:    {unemp_s['mean']:6.2f}%
  Std Dev: {unemp_s['std']:6.2f}%
  Peak:    {unemp_s['max']:6.2f}%  ({unemp_s['max_date']})
  Trough:  {unemp_s['min']:6.2f}%  ({unemp_s['min_date']})

PHILLIPS CURVE - Correlation (inflation vs. unemployment)
  Full period:  r = {corr_all:+.3f}  {'(negative = expected)' if corr_all < 0 else '(positive = stagflation effects)'}
"""
    for era_label, era_df in eras.items():
        if len(era_df) >= 5:
            r = era_df["inflation"].corr(era_df["UNRATE"])
            slope, _, _, p, _ = stats.linregress(era_df["UNRATE"], era_df["inflation"])
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "  ")
            report += f"  {era_label}:  r = {r:+.3f}  slope = {slope:+.2f}  p = {p:.4f} {sig}\n"

    report += """
KEY INSIGHTS
  - The classic Phillips Curve (inverse trade-off) held best in the 1960s.
  - The 1970s stagflation broke the simple trade-off: high inflation AND
    high unemployment coexisted, driven by oil supply shocks.
  - The Volcker disinflation (1979-1983) drove unemployment to ~10.8%
    to break inflation expectations.
  - 2000s-2010s: "Great Moderation" - low & stable inflation + low unemployment.
  - 2020-2022: Post-pandemic supply chain shocks caused inflation to surge
    despite falling unemployment (supply-side, not demand-side).

Outputs saved to economic_analysis/output/
  inflation_trends.png    - headline vs. core inflation
  unemployment_trends.png - unemployment with moving average
  macro_dashboard.png     - 3-panel dashboard
  phillips_curve.png      - scatter + era breakdown
  decade_summary.png      - average rates by decade
"""
    print(report)

    with open(OUTPUT_DIR / "report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved: report.txt")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(START, END)

    plot_inflation_trends(df)
    plot_unemployment_trends(df)
    plot_combined_dashboard(df)
    plot_phillips_curve(df)
    plot_decade_summary(df)
    print_report(df)

    print("\nDone. All outputs in economic_analysis/output/")
