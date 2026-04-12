"""
US Economic Data Dashboard - Streamlit
======================================
Run with:  streamlit run economic_analysis/streamlit_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from scipy import stats

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="US Economic Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config ────────────────────────────────────────────────────────────────────
START = "1960-01-01"
END   = "2025-01-01"

SERIES = {
    "CPI":      "CPIAUCSL",
    "CORE_CPI": "CPILFESL",
    "UNRATE":   "UNRATE",
    "FEDFUNDS": "FEDFUNDS",
}

STYLE = {
    "inflation":    "#d62728",
    "core_infl":    "#ff7f0e",
    "unemployment": "#1f77b4",
    "fedfunds":     "#2ca02c",
    "recession":    "#d0d0d0",
}

RECESSIONS = [
    ("1960-04", "1961-02"), ("1969-12", "1970-11"), ("1973-11", "1975-03"),
    ("1980-01", "1980-07"), ("1981-07", "1982-11"), ("1990-07", "1991-03"),
    ("2001-03", "2001-11"), ("2007-12", "2009-06"), ("2020-02", "2020-04"),
]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e2a3a;
    }
    [data-testid="stSidebar"] * {
        color: #c8d6e5 !important;
    }
    /* Radio buttons */
    [data-testid="stSidebar"] .stRadio label {
        font-size: 15px;
        padding: 6px 0;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #2e86de;
    }
    /* Explanation box */
    .expl-box {
        background-color: #f8f9fa;
        border-left: 4px solid #2e86de;
        border-radius: 4px;
        padding: 16px 20px;
        font-size: 14px;
        line-height: 1.7;
        white-space: pre-wrap;
        font-family: monospace;
    }
    /* Section headers */
    .section-header {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #7f8c9a;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    raw = {}
    for name, symbol in SERIES.items():
        raw[name] = web.DataReader(symbol, "fred", START, END)[symbol]
    df = pd.DataFrame(raw).resample("MS").last()
    df["inflation"] = df["CPI"].pct_change(12) * 100
    df["core_infl"] = df["CORE_CPI"].pct_change(12) * 100
    df.dropna(subset=["inflation", "UNRATE"], inplace=True)
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────
def shade_recessions(ax, df):
    for rs, re in RECESSIONS:
        rs_dt, re_dt = pd.to_datetime(rs), pd.to_datetime(re)
        if rs_dt >= df.index[0] and re_dt <= df.index[-1]:
            ax.axvspan(rs_dt, re_dt, color=STYLE["recession"], alpha=0.5, lw=0)


def make_fig(h=5):
    fig, ax = plt.subplots(figsize=(12, h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    return fig, ax


# ── Chart Builders ────────────────────────────────────────────────────────────
def chart_inflation(df, start_yr, end_yr):
    sub = df[str(start_yr):str(end_yr)]
    fig, ax = make_fig(5)
    shade_recessions(ax, sub)
    ax.plot(sub.index, sub["inflation"], color=STYLE["inflation"],
            lw=2, label="Headline Inflation (YoY %)")
    ax.plot(sub.index, sub["core_infl"], color=STYLE["core_infl"],
            lw=2, ls="--", label="Core Inflation (excl. food & energy)")
    ax.axhline(2, color="#555", lw=1, ls=":", label="2% Fed Target")
    ax.axhline(0, color="#bbb", lw=0.6)
    ax.set_title("US Inflation Over Time", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Year-over-Year Change (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    extra = [Patch(facecolor=STYLE["recession"], label="NBER Recession")]
    h_, l_ = ax.get_legend_handles_labels()
    ax.legend(h_ + extra, l_ + ["NBER Recession"], fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    return fig


def chart_unemployment(df, start_yr, end_yr):
    sub = df[str(start_yr):str(end_yr)]
    fig, ax = make_fig(5)
    shade_recessions(ax, sub)
    ax.fill_between(sub.index, sub["UNRATE"], alpha=0.15, color=STYLE["unemployment"])
    ax.plot(sub.index, sub["UNRATE"], color=STYLE["unemployment"],
            lw=2, label="Unemployment Rate")
    roll = sub["UNRATE"].rolling(24).mean()
    ax.plot(sub.index, roll, color="navy", lw=1.4, ls="--",
            label="24-month Moving Average")
    ax.set_title("US Unemployment Rate Over Time", fontsize=13,
                 fontweight="bold", pad=10)
    ax.set_ylabel("Unemployment Rate (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    extra = [Patch(facecolor=STYLE["recession"], label="NBER Recession")]
    h_, l_ = ax.get_legend_handles_labels()
    ax.legend(h_ + extra, l_ + ["NBER Recession"], fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    return fig


def chart_dashboard(df, start_yr, end_yr):
    sub = df[str(start_yr):str(end_yr)]
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.5)

    panels = [
        (gs[0], "inflation",  STYLE["inflation"],    "Inflation (YoY %)"),
        (gs[1], "UNRATE",     STYLE["unemployment"], "Unemployment (%)"),
        (gs[2], "FEDFUNDS",   STYLE["fedfunds"],     "Fed Funds Rate (%)"),
    ]
    for i, (slot, col, color, ylabel) in enumerate(panels):
        ax = fig.add_subplot(slot)
        ax.set_facecolor("#fafafa")
        shade_recessions(ax, sub)
        ax.plot(sub.index, sub[col], color=color, lw=1.8, label=ylabel)
        if col == "inflation":
            ax.axhline(2, color="#555", lw=0.8, ls=":")
            ax.set_title("US Macroeconomic Dashboard", fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(axis="y", alpha=0.25, ls="--")

    fig.legend([Patch(facecolor=STYLE["recession"])], ["NBER Recession"],
               loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def chart_phillips(df, start_yr, end_yr):
    sub = df[str(start_yr):str(end_yr)]
    eras = {
        "1960s-70s": (sub["1960":"1979"], "#e41a1c"),
        "1980s-90s": (sub["1980":"1999"], "#377eb8"),
        "2000s-10s": (sub["2000":"2019"], "#4daf4a"),
        "2020s":     (sub["2020":],       "#ff7f00"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle("Phillips Curve: Inflation vs. Unemployment",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.set_facecolor("#fafafa")
    sc = ax.scatter(sub["UNRATE"], sub["inflation"],
                    c=sub.index.year, cmap="plasma", alpha=0.5, s=14, zorder=3)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Year", fontsize=8)
    sl, ic, r, p, _ = stats.linregress(sub["UNRATE"], sub["inflation"])
    xr = np.linspace(sub["UNRATE"].min(), sub["UNRATE"].max(), 100)
    ax.plot(xr, sl * xr + ic, color="black", lw=2, label=f"Trend  r = {r:.2f}")
    ax.axhline(0, color="#bbb", lw=0.6)
    ax.axhline(2, color="#bbb", lw=0.6, ls=":")
    ax.set_xlabel("Unemployment Rate (%)")
    ax.set_ylabel("Inflation Rate (YoY %)")
    ax.set_title(f"Selected Period\nr = {r:.2f},  p = {p:.4f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, ls="--")

    ax2 = axes[1]
    ax2.set_facecolor("#fafafa")
    for era_label, (era_df, color) in eras.items():
        if len(era_df) < 5:
            continue
        ax2.scatter(era_df["UNRATE"], era_df["inflation"],
                    color=color, alpha=0.65, s=16, label=era_label, zorder=3)
        sl2, ic2, *_ = stats.linregress(era_df["UNRATE"], era_df["inflation"])
        xr2 = np.linspace(era_df["UNRATE"].min(), era_df["UNRATE"].max(), 50)
        ax2.plot(xr2, sl2 * xr2 + ic2, color=color, lw=1.8)
    ax2.axhline(0, color="#bbb", lw=0.6)
    ax2.axhline(2, color="#bbb", lw=0.6, ls=":")
    ax2.set_xlabel("Unemployment Rate (%)")
    ax2.set_ylabel("Inflation Rate (YoY %)")
    ax2.set_title("By Era  (the curve shifts over time)")
    ax2.legend(fontsize=8, framealpha=0.95)
    ax2.grid(alpha=0.25, ls="--")

    fig.tight_layout()
    return fig


def chart_decade(df):
    df2 = df.copy()
    df2["decade"] = (df2.index.year // 10 * 10).astype(str) + "s"
    summary = df2.groupby("decade")[["inflation", "UNRATE"]].mean().reset_index()

    fig, ax = make_fig(5)
    x = np.arange(len(summary))
    w = 0.35
    b1 = ax.bar(x - w/2, summary["inflation"], w,
                label="Avg Inflation (%)", color=STYLE["inflation"], alpha=0.85)
    b2 = ax.bar(x + w/2, summary["UNRATE"], w,
                label="Avg Unemployment (%)", color=STYLE["unemployment"], alpha=0.85)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["decade"], fontsize=10)
    ax.set_ylabel("Average Rate (%)")
    ax.set_title("Average Inflation & Unemployment by Decade",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.set_ylim(0, max(summary["inflation"].max(), summary["UNRATE"].max()) + 2)
    fig.tight_layout()
    return fig


# ── Explanations ──────────────────────────────────────────────────────────────
EXPLANATIONS = {
    "📈 Inflation Trends": {
        "what": """This chart shows US inflation from 1960 to 2025, measured as the year-over-year percentage change in the Consumer Price Index (CPI).

**Two lines are shown:**
- 🔴 **Headline Inflation** — all goods including food & energy
- 🟠 **Core Inflation** (dashed) — strips out volatile food & energy prices
- ⬛ **2% Target** (dotted) — the Fed's official goal since the 1990s

Gray bands mark NBER-defined US recessions.""",
        "insights": """- **1973–74:** Arab oil embargo causes inflation to surge past 10%
- **1980:** Inflation peaks at **14.6%** — the highest in modern history
- **1983–2019:** The "Great Moderation" — inflation stays low & stable
- **2021–22:** Post-pandemic supply shocks push inflation to **9%**, a 40-year high
- **2023–24:** Fed rate hikes bring inflation back toward the 2% target""",
        "why": """**Why Core Inflation Matters**

Core CPI strips out food and energy because they're extremely volatile — a single hurricane can spike gas prices, but that doesn't mean the whole economy is overheating. The Fed uses core inflation to understand the true underlying trend.""",
    },
    "📉 Unemployment": {
        "what": """This chart shows the US civilian unemployment rate — the percentage of people actively looking for work but unable to find it.

The **dashed line** is a 24-month rolling average, smoothing out month-to-month noise to reveal the longer trend.

Gray bands mark NBER-defined US recessions.""",
        "insights": """- **1968:** Historic low of **3.4%** during the Vietnam-era economic boom
- **1982:** Peaks at **10.8%** — worst since the Great Depression — caused deliberately by the Fed to kill 1970s inflation
- **2009:** Financial Crisis drives unemployment to **10%**
- **2020:** COVID-19: fastest spike ever — 3.5% → **14.8%** in just 2 months, then the fastest recovery ever""",
        "why": """**The Asymmetric Pattern**

Unemployment rises sharply during recessions (notice the steep cliffs into the gray bands) but falls slowly during recoveries. Economists call this "asymmetric" — it takes years to rebuild jobs but only months to destroy them.""",
    },
    "🖥️ Macro Dashboard": {
        "what": """Three key indicators stacked together to reveal how they interact:

- **Panel 1 — Inflation:** Year-over-year CPI change (%)
- **Panel 2 — Unemployment:** Civilian unemployment rate (%)
- **Panel 3 — Fed Funds Rate:** The interest rate the Federal Reserve sets (%)""",
        "insights": """- **1970s:** Inflation soars → Fed raises rates massively → causes 1981-82 recession → unemployment hits 10.8%
- **1990s:** All three stabilize — "Goldilocks" era of steady growth
- **2008:** Fed slashes rates to 0% to fight the Financial Crisis
- **2020:** Rates hit 0% again for COVID, then inflation surges
- **2022-23:** Fed raises rates from 0% to **5.25%** in 18 months — fastest tightening in 40 years""",
        "why": """**The Fed's Impossible Balancing Act (Dual Mandate)**

When inflation rises → Fed RAISES rates → slows spending → unemployment tends to rise.
When unemployment rises → Fed LOWERS rates → stimulates spending → inflation may rise.

This tension is the "dual mandate" — the Fed must balance both at the same time.""",
    },
    "🔄 Phillips Curve": {
        "what": """The Phillips Curve is one of the most famous ideas in economics.

In **1958**, economist A.W. Phillips noticed: when unemployment is LOW, inflation tends to be HIGH — and vice versa. The logic: when jobs are plentiful, workers demand higher wages, which pushes up prices.

- **Left chart:** Each dot = one month, colored by year. Black line = overall trend.
- **Right chart:** The relationship split by era to show how it has shifted.""",
        "insights": """| Era | Correlation (r) | Story |
|---|---|---|
| 1960s-70s | +0.32 | Stagflation — BOTH rose together (oil shocks broke the rule) |
| 1980s-90s | +0.31 | Disinflation — both fell together as Volcker's medicine worked |
| 2000s-10s | -0.30 | Classic inverse trade-off returns |
| 2020s | **-0.57** | Strongest inverse link in decades |""",
        "why": """**What r means:**
- r = -1 → perfect inverse trade-off (textbook Phillips Curve)
- r = +1 → both move together (stagflation)
- r ≈ 0 → no clear relationship

The 1970s proved that **supply shocks can completely break the curve** — you can have high inflation AND high unemployment simultaneously.""",
    },
    "📊 Decade Summary": {
        "what": """Average inflation and unemployment rates grouped by decade — a quick way to compare economic eras at a glance.

- 🔴 Red bars = Average Inflation (%)
- 🔵 Blue bars = Average Unemployment (%)""",
        "insights": """| Decade | Story |
|---|---|
| **1960s** | Low unemployment (~4.8%), low inflation (~2.3%) — Kennedy/LBJ boom |
| **1970s** | Inflation explodes to ~7.1% from oil shocks and loose money policy |
| **1980s** | Volcker crushes inflation but unemployment averages 7.3% — highest ever |
| **1990s** | "Goldilocks" — internet boom, inflation ~3%, unemployment ~5.8% |
| **2000s** | Tech bust + Financial Crisis; decade ends in worst crisis since 1929 |
| **2010s** | Slow recovery; inflation stays BELOW 2% target — the opposite problem |
| **2020s** | Most volatile decade since 1970s: COVID shock, 9% inflation, rapid tightening |""",
        "why": """**The Big Picture**

Looking across decades reveals long economic cycles that are invisible in month-to-month data. The 1980s stand out as the decade of painful medicine — deliberately engineering high unemployment to permanently break inflation expectations.""",
    },
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 US Economic\nDashboard")
    st.caption("Data: Federal Reserve (FRED)")
    st.divider()

    view = st.radio(
        "Select View",
        options=list(EXPLANATIONS.keys()),
        index=0,
    )

    st.divider()
    st.markdown("**Date Range Filter**")
    st.caption("Applies to all charts except Decade Summary")
    year_range = st.slider(
        "Years",
        min_value=1961,
        max_value=2025,
        value=(1961, 2025),
        step=1,
    )

    st.divider()
    st.markdown(
        "<div style='font-size:11px;color:#7f8c9a'>"
        "Gray bands = NBER recessions<br>"
        "Source: FRED (St. Louis Fed)<br>"
        "CPI, UNRATE, FEDFUNDS series"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main Content ──────────────────────────────────────────────────────────────
# Load data with spinner
with st.spinner("Downloading data from FRED — this only happens once..."):
    df = load_data()

start_yr, end_yr = year_range

# ── Key Metrics Row ───────────────────────────────────────────────────────────
sub = df[str(start_yr):str(end_yr)]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Avg Inflation",    f"{sub['inflation'].mean():.1f}%")
col2.metric("Peak Inflation",   f"{sub['inflation'].max():.1f}%",
            sub['inflation'].idxmax().strftime("%b %Y"))
col3.metric("Avg Unemployment", f"{sub['UNRATE'].mean():.1f}%")
col4.metric("Peak Unemployment",f"{sub['UNRATE'].max():.1f}%",
            sub['UNRATE'].idxmax().strftime("%b %Y"))
r_val = sub["inflation"].corr(sub["UNRATE"])
col5.metric("Inflation-Unemployment\nCorrelation", f"{r_val:+.2f}",
            "inverse trade-off" if r_val < 0 else "move together")

st.divider()

# ── Chart ─────────────────────────────────────────────────────────────────────
if view == "📈 Inflation Trends":
    fig = chart_inflation(df, start_yr, end_yr)
elif view == "📉 Unemployment":
    fig = chart_unemployment(df, start_yr, end_yr)
elif view == "🖥️ Macro Dashboard":
    fig = chart_dashboard(df, start_yr, end_yr)
elif view == "🔄 Phillips Curve":
    fig = chart_phillips(df, start_yr, end_yr)
else:
    fig = chart_decade(df)

st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── Explanation ───────────────────────────────────────────────────────────────
expl = EXPLANATIONS[view]

st.divider()
exp_col1, exp_col2 = st.columns([1, 1])

with exp_col1:
    st.markdown("#### What you're looking at")
    st.markdown(expl["what"])

with exp_col2:
    st.markdown("#### Key moments & insights")
    st.markdown(expl["insights"])

st.markdown("#### Why it matters")
st.info(expl["why"])
