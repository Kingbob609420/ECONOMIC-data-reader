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
import requests
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from scipy import stats

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Economic Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
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

PRESETS = {
    "Last 10 Years":  (2015, 2025),
    "Last 20 Years":  (2005, 2025),
    "Last 30 Years":  (1995, 2025),
    "Post-2000":      (2000, 2025),
    "Full History":   (1961, 2025),
}

VIEWS = [
    "📈 Inflation Trends",
    "📉 Unemployment",
    "🖥️ Macro Dashboard",
    "🔄 Phillips Curve",
    "📊 Decade Summary",
    "📋 Full Summary",
]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1e2a3a; }
    [data-testid="stSidebar"] * { color: #c8d6e5 !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 15px; padding: 6px 0; }
    .preset-active {
        background-color: #2e86de !important;
        color: white !important;
        border-radius: 4px;
    }
    .summary-card {
        background: #f8f9fa;
        border-left: 4px solid #2e86de;
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state (year range) ────────────────────────────────────────────────
if "yr_start" not in st.session_state:
    st.session_state.yr_start = 1961
if "yr_end" not in st.session_state:
    st.session_state.yr_end = 2025


def apply_preset(s, e):
    st.session_state.yr_start = s
    st.session_state.yr_end = e


# ── Data ──────────────────────────────────────────────────────────────────────
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

@st.cache_data(show_spinner=False)
def load_data():
    raw = {}
    for name, symbol in SERIES.items():
        resp = requests.get(FRED_URL.format(symbol), timeout=30)
        resp.raise_for_status()
        s = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
        s.columns = [name]
        raw[name] = s
    df = pd.concat(raw.values(), axis=1).resample("MS").last()
    df["inflation"] = df["CPI"].pct_change(12) * 100
    df["core_infl"] = df["CORE_CPI"].pct_change(12) * 100
    df = df[df.index >= START]
    df.dropna(subset=["inflation", "UNRATE"], inplace=True)
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────
def shade_recessions(ax, sub):
    for rs, re in RECESSIONS:
        rs_dt, re_dt = pd.to_datetime(rs), pd.to_datetime(re)
        if rs_dt >= sub.index[0] and re_dt <= sub.index[-1]:
            ax.axvspan(rs_dt, re_dt, color=STYLE["recession"], alpha=0.5, lw=0)


def make_fig(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    return fig, ax


def period_stats(sub):
    r, p, *_ = stats.linregress(sub["UNRATE"], sub["inflation"])
    return {
        "avg_infl":    sub["inflation"].mean(),
        "max_infl":    sub["inflation"].max(),
        "max_infl_dt": sub["inflation"].idxmax(),
        "min_infl":    sub["inflation"].min(),
        "min_infl_dt": sub["inflation"].idxmin(),
        "avg_unemp":   sub["UNRATE"].mean(),
        "max_unemp":   sub["UNRATE"].max(),
        "max_unemp_dt":sub["UNRATE"].idxmax(),
        "min_unemp":   sub["UNRATE"].min(),
        "min_unemp_dt":sub["UNRATE"].idxmin(),
        "avg_ff":      sub["FEDFUNDS"].mean(),
        "corr":        sub["inflation"].corr(sub["UNRATE"]),
        "slope":       r,
        "months":      len(sub),
    }


# ── Chart Builders ────────────────────────────────────────────────────────────
def chart_inflation(df, s, e, compact=False):
    sub = df[str(s):str(e)]
    fig, ax = make_fig(h=4 if compact else 5)
    shade_recessions(ax, sub)
    ax.plot(sub.index, sub["inflation"], color=STYLE["inflation"],
            lw=1.8, label="Headline Inflation")
    ax.plot(sub.index, sub["core_infl"], color=STYLE["core_infl"],
            lw=1.6, ls="--", label="Core Inflation")
    ax.axhline(2, color="#555", lw=0.9, ls=":", label="2% Fed Target")
    ax.axhline(0, color="#bbb", lw=0.5)
    ax.set_title("Inflation Over Time", fontsize=11 if compact else 13,
                 fontweight="bold", pad=8)
    ax.set_ylabel("YoY Change (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    if not compact:
        extra = [Patch(facecolor=STYLE["recession"], label="NBER Recession")]
        h_, l_ = ax.get_legend_handles_labels()
        ax.legend(h_ + extra, l_ + ["NBER Recession"], fontsize=8, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    return fig


def chart_unemployment(df, s, e, compact=False):
    sub = df[str(s):str(e)]
    fig, ax = make_fig(h=4 if compact else 5)
    shade_recessions(ax, sub)
    ax.fill_between(sub.index, sub["UNRATE"], alpha=0.15, color=STYLE["unemployment"])
    ax.plot(sub.index, sub["UNRATE"], color=STYLE["unemployment"],
            lw=1.8, label="Unemployment Rate")
    roll = sub["UNRATE"].rolling(24).mean()
    ax.plot(sub.index, roll, color="navy", lw=1.3, ls="--",
            label="24-month Avg")
    ax.set_title("Unemployment Rate Over Time", fontsize=11 if compact else 13,
                 fontweight="bold", pad=8)
    ax.set_ylabel("Unemployment (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    if not compact:
        extra = [Patch(facecolor=STYLE["recession"], label="NBER Recession")]
        h_, l_ = ax.get_legend_handles_labels()
        ax.legend(h_ + extra, l_ + ["NBER Recession"], fontsize=8, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    return fig


def chart_dashboard(df, s, e):
    sub = df[str(s):str(e)]
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


def chart_phillips(df, s, e, compact=False):
    sub = df[str(s):str(e)]
    eras = {
        "1960s-70s": (sub["1960":"1979"], "#e41a1c"),
        "1980s-90s": (sub["1980":"1999"], "#377eb8"),
        "2000s-10s": (sub["2000":"2019"], "#4daf4a"),
        "2020s":     (sub["2020":],       "#ff7f00"),
    }
    ncols = 1 if compact else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 if compact else 13, 5))
    if compact:
        axes = [axes]
    fig.patch.set_facecolor("white")
    fig.suptitle("Phillips Curve", fontsize=11 if compact else 13, fontweight="bold")

    ax = axes[0]
    ax.set_facecolor("#fafafa")
    sc = ax.scatter(sub["UNRATE"], sub["inflation"],
                    c=sub.index.year, cmap="plasma", alpha=0.5, s=12, zorder=3)
    plt.colorbar(sc, ax=ax).set_label("Year", fontsize=7)
    sl, ic, r, p, _ = stats.linregress(sub["UNRATE"], sub["inflation"])
    xr = np.linspace(sub["UNRATE"].min(), sub["UNRATE"].max(), 100)
    ax.plot(xr, sl * xr + ic, color="black", lw=2, label=f"r = {r:.2f}")
    ax.axhline(0, color="#bbb", lw=0.5)
    ax.set_xlabel("Unemployment Rate (%)")
    ax.set_ylabel("Inflation (YoY %)")
    ax.set_title(f"r = {r:.2f},  p = {p:.4f}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, ls="--")

    if not compact:
        ax2 = axes[1]
        ax2.set_facecolor("#fafafa")
        for era_label, (era_df, color) in eras.items():
            if len(era_df) < 5:
                continue
            ax2.scatter(era_df["UNRATE"], era_df["inflation"],
                        color=color, alpha=0.65, s=14, label=era_label, zorder=3)
            sl2, ic2, *_ = stats.linregress(era_df["UNRATE"], era_df["inflation"])
            xr2 = np.linspace(era_df["UNRATE"].min(), era_df["UNRATE"].max(), 50)
            ax2.plot(xr2, sl2 * xr2 + ic2, color=color, lw=1.8)
        ax2.axhline(0, color="#bbb", lw=0.5)
        ax2.set_xlabel("Unemployment Rate (%)")
        ax2.set_ylabel("Inflation (YoY %)")
        ax2.set_title("By Era")
        ax2.legend(fontsize=8, framealpha=0.95)
        ax2.grid(alpha=0.25, ls="--")

    fig.tight_layout()
    return fig


def chart_decade(df, compact=False):
    df2 = df.copy()
    df2["decade"] = (df2.index.year // 10 * 10).astype(str) + "s"
    summary = df2.groupby("decade")[["inflation", "UNRATE"]].mean().reset_index()
    fig, ax = make_fig(h=4 if compact else 5)
    x = np.arange(len(summary))
    w = 0.35
    b1 = ax.bar(x - w/2, summary["inflation"], w,
                label="Avg Inflation (%)", color=STYLE["inflation"], alpha=0.85)
    b2 = ax.bar(x + w/2, summary["UNRATE"], w,
                label="Avg Unemployment (%)", color=STYLE["unemployment"], alpha=0.85)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=7 if compact else 9,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["decade"], fontsize=8 if compact else 10)
    ax.set_ylabel("Average Rate (%)")
    ax.set_title("Avg Rates by Decade", fontsize=11 if compact else 13,
                 fontweight="bold", pad=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.set_ylim(0, max(summary["inflation"].max(), summary["UNRATE"].max()) + 2)
    fig.tight_layout()
    return fig


# ── Auto-generated Summary Text ───────────────────────────────────────────────
def generate_summary(df, s, e):
    sub = df[str(s):str(e)]
    p   = period_stats(sub)
    full = period_stats(df)

    infl_trend  = "falling" if sub["inflation"].iloc[-12:].mean() < sub["inflation"].iloc[:12].mean() else "rising"
    unemp_trend = "falling" if sub["UNRATE"].iloc[-12:].mean() < sub["UNRATE"].iloc[:12].mean() else "rising"
    corr_desc   = "a strong inverse trade-off (textbook Phillips Curve)" if p["corr"] < -0.3 \
        else ("no clear relationship" if abs(p["corr"]) < 0.1 \
        else "both moving in the same direction (stagflation dynamics)")

    vs_hist_infl  = "above" if p["avg_infl"]  > full["avg_infl"]  else "below"
    vs_hist_unemp = "above" if p["avg_unemp"] > full["avg_unemp"] else "below"

    current_infl  = sub["inflation"].iloc[-1]
    current_unemp = sub["UNRATE"].iloc[-1]
    current_ff    = sub["FEDFUNDS"].iloc[-1]

    near_target = abs(current_infl - 2) < 1.0

    return {
        "period_label": f"{s}–{e}",
        "months": p["months"],
        "stats": p,
        "full_stats": full,
        "narrative": f"""
Over the **{s}–{e}** period ({p['months']} months of data), the US economy showed the following
macroeconomic profile:

**Inflation** averaged **{p['avg_infl']:.1f}%** per year — {vs_hist_infl} the long-run historical
average of {full['avg_infl']:.1f}%. Inflation peaked at **{p['max_infl']:.1f}%**
({p['max_infl_dt'].strftime('%B %Y')}) and fell as low as **{p['min_infl']:.1f}%**
({p['min_infl_dt'].strftime('%B %Y')}). The most recent reading was **{current_infl:.1f}%**,
which is {'near' if near_target else 'away from'} the Fed's 2% target. The overall
trend in inflation over this period was **{infl_trend}**.

**Unemployment** averaged **{p['avg_unemp']:.1f}%** — {vs_hist_unemp} the historical average of
{full['avg_unemp']:.1f}%. The labor market was tightest at **{p['min_unemp']:.1f}%**
({p['min_unemp_dt'].strftime('%B %Y')}) and most stressed at **{p['max_unemp']:.1f}%**
({p['max_unemp_dt'].strftime('%B %Y')}). The most recent unemployment rate was
**{current_unemp:.1f}%**, and the overall trend was **{unemp_trend}**.

**The Fed Funds Rate** averaged **{p['avg_ff']:.1f}%** over this period, with the most
recent level at **{current_ff:.1f}%**.

**Inflation vs. Unemployment (Phillips Curve):** The correlation between the two was
**r = {p['corr']:+.2f}**, indicating {corr_desc}. A negative correlation suggests the
classic trade-off is working; a positive correlation suggests supply-side shocks are
dominating.
""",
        "current": {
            "inflation":    current_infl,
            "unemployment": current_unemp,
            "fed_funds":    current_ff,
        },
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 US Economic\nDashboard")
    st.caption("Data: Federal Reserve (FRED)")
    st.divider()

    view = st.radio("Select View", options=VIEWS, index=0)

    st.divider()
    st.markdown("**Time Period**")

    # Preset toggle buttons
    for label, (s, e) in PRESETS.items():
        is_active = (st.session_state.yr_start == s and st.session_state.yr_end == e)
        st.button(
            f"{'✅ ' if is_active else ''}{label}",
            on_click=apply_preset,
            args=(s, e),
            use_container_width=True,
            key=f"preset_{label}",
        )

    st.caption("Or drag to set a custom range:")
    year_range = st.slider(
        "Custom range",
        min_value=1961, max_value=2025,
        value=(st.session_state.yr_start, st.session_state.yr_end),
        label_visibility="collapsed",
    )
    # Keep session state in sync with slider
    st.session_state.yr_start = year_range[0]
    st.session_state.yr_end   = year_range[1]

    st.divider()
    st.markdown(
        "<div style='font-size:11px;color:#7f8c9a'>"
        "Gray bands = NBER recessions<br>"
        "Source: FRED (St. Louis Fed)<br>"
        "CPI · UNRATE · FEDFUNDS series"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Downloading data from FRED — this only happens once..."):
    df = load_data()

s = st.session_state.yr_start
e = st.session_state.yr_end
sub = df[str(s):str(e)]

# ── Metrics Row ───────────────────────────────────────────────────────────────
st.markdown(f"### {view}  `{s} – {e}`")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Avg Inflation",      f"{sub['inflation'].mean():.1f}%")
m2.metric("Peak Inflation",     f"{sub['inflation'].max():.1f}%",
          sub['inflation'].idxmax().strftime("%b %Y"))
m3.metric("Avg Unemployment",   f"{sub['UNRATE'].mean():.1f}%")
m4.metric("Peak Unemployment",  f"{sub['UNRATE'].max():.1f}%",
          sub['UNRATE'].idxmax().strftime("%b %Y"))
m5.metric("Avg Fed Funds Rate", f"{sub['FEDFUNDS'].mean():.1f}%")
r_val = sub["inflation"].corr(sub["UNRATE"])
m6.metric("Infl/Unemp Corr.",   f"{r_val:+.2f}",
          "inverse" if r_val < 0 else "moves together")

st.divider()

# ── View Router ───────────────────────────────────────────────────────────────

# ── Individual chart views ────────────────────────────────────────────────────
if view == "📈 Inflation Trends":
    st.pyplot(chart_inflation(df, s, e), use_container_width=True)
    plt.close("all")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### What you're looking at")
        st.markdown("""
This chart shows US inflation measured as the **year-over-year % change in the Consumer Price Index (CPI)**.

- 🔴 **Headline Inflation** — all goods including food & energy
- 🟠 **Core Inflation** (dashed) — strips out volatile food & energy prices
- ⬛ **2% dotted line** — the Federal Reserve's official inflation target

Gray bands are **NBER-defined US recessions**.
""")
    with c2:
        st.markdown("#### Key moments")
        st.markdown("""
- **1973–74:** Arab oil embargo — inflation surges past 10%
- **1980:** Inflation peaks at **14.6%** — highest in modern US history
- **1983–2019:** "The Great Moderation" — inflation stays low & stable
- **2021–22:** Post-pandemic supply shocks push inflation to **9%**, a 40-year high
- **2023–24:** Fed rate hikes bring inflation back toward target
""")
    st.info("""**Why Core Inflation matters:** Core CPI strips out food and energy because they're extremely volatile. A hurricane spiking gas prices tells a different story than broad price increases. The Fed watches core inflation to understand the *true* underlying trend.""")

elif view == "📉 Unemployment":
    st.pyplot(chart_unemployment(df, s, e), use_container_width=True)
    plt.close("all")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### What you're looking at")
        st.markdown("""
The **civilian unemployment rate** — the % of people actively looking for work but unable to find it.

- 🔵 **Solid line** — monthly unemployment rate
- 🟦 **Dashed line** — 24-month rolling average (smooths out noise)
- Gray bands — NBER recessions

Notice how unemployment **spikes sharply INTO recessions** but **recovers slowly** afterwards.
""")
    with c2:
        st.markdown("#### Key moments")
        st.markdown("""
- **1968:** Historic low of **3.4%** during the Vietnam-era boom
- **1982:** Peaks at **10.8%** — deliberate, to kill 1970s inflation
- **2009:** Financial Crisis drives unemployment to **10%**
- **2020:** COVID-19 — 3.5% → **14.8%** in just **2 months** (fastest spike ever)
- **2022:** Fastest recovery ever — back below 4% within 2 years
""")
    st.info("""**The Asymmetric Pattern:** Jobs are easy to destroy, hard to rebuild. Recessions wipe out years of job gains in months. Recoveries take years. This is why preventing recessions matters more than ending them.""")

elif view == "🖥️ Macro Dashboard":
    st.pyplot(chart_dashboard(df, s, e), use_container_width=True)
    plt.close("all")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### What you're looking at")
        st.markdown("""
Three indicators stacked to show how they **interact and respond to each other**:

- 🔴 **Inflation** — year-over-year CPI change
- 🔵 **Unemployment** — civilian unemployment rate
- 🟢 **Fed Funds Rate** — the interest rate the Federal Reserve controls

The Fed Funds Rate is the **main lever** the Fed uses to manage the economy.
""")
    with c2:
        st.markdown("#### The policy chain reaction")
        st.markdown("""
**When inflation rises:**
Fed RAISES rates → borrowing costs more → spending slows → unemployment rises

**When unemployment rises:**
Fed LOWERS rates → borrowing is cheaper → spending rises → inflation may rise

**Key eras:**
- **1979-83:** Volcker raised rates to 19% to kill inflation — caused deep recession
- **2008-15:** Fed slashed rates to 0% to fight Financial Crisis
- **2022-23:** Fed raised rates from 0% → 5.25% in just 18 months
""")
    st.info("""**The Dual Mandate:** The Fed is legally required to balance BOTH low inflation AND low unemployment at the same time. These goals often conflict — managing that tension is the central challenge of monetary policy.""")

elif view == "🔄 Phillips Curve":
    st.pyplot(chart_phillips(df, s, e), use_container_width=True)
    plt.close("all")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### What you're looking at")
        st.markdown("""
The **Phillips Curve** — one of the most famous ideas in economics.

In **1958**, A.W. Phillips observed: when unemployment is LOW, inflation tends to be HIGH (and vice versa). When jobs are plentiful, workers demand higher wages → prices rise.

- **Left chart:** Every month as a dot, colored by year. Black line = OLS trend.
- **Right chart:** Same data split by era — shows the curve **shifting over time**.
""")
    with c2:
        st.markdown("#### Era breakdown")
        st.markdown("""
| Era | r value | What happened |
|---|---|---|
| **1960s-70s** | +0.32 | Oil shocks caused **stagflation** — broke the rule |
| **1980s-90s** | +0.31 | Both fell together during Volcker disinflation |
| **2000s-10s** | -0.30 | Classic inverse trade-off returns |
| **2020s** | **-0.57** | Strongest inverse link in decades |

**r = -1** → perfect inverse (textbook) · **r = +1** → stagflation · **r ≈ 0** → no link
""")
    st.info("""**The 1970s lesson:** Supply shocks (like oil embargos) can completely break the Phillips Curve. You can have BOTH high inflation AND high unemployment simultaneously. This is called stagflation, and it's the worst-case scenario for policymakers.""")

elif view == "📊 Decade Summary":
    st.pyplot(chart_decade(df), use_container_width=True)
    plt.close("all")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### What you're looking at")
        st.markdown("""
Average **inflation** and **unemployment** grouped by decade — a 10,000-foot view of US economic history.

- 🔴 Red bars = Average annual inflation
- 🔵 Blue bars = Average unemployment rate

This view reveals **long economic cycles** invisible in month-to-month data.
""")
    with c2:
        st.markdown("#### Decade by decade")
        st.markdown("""
| Decade | Inflation | Unemployment | Story |
|---|---|---|---|
| **1960s** | ~2.3% | ~4.8% | Kennedy/LBJ boom |
| **1970s** | ~7.1% | ~6.2% | Oil shocks, stagflation |
| **1980s** | ~5.6% | ~7.3% | Volcker medicine — painful but effective |
| **1990s** | ~3.0% | ~5.8% | Internet boom, Goldilocks era |
| **2000s** | ~2.6% | ~5.5% | Tech bust + Financial Crisis |
| **2010s** | ~1.8% | ~6.2% | Slow recovery, below-target inflation |
| **2020s** | ~4.5% | ~4.4% | COVID, 9% inflation, rapid tightening |
""")
    st.info("""**The standout decade:** The 1980s had the highest average unemployment of any peacetime decade — a deliberate choice. The Volcker Fed decided that permanently breaking inflation expectations was worth years of economic pain. It worked — inflation stayed low for 40 years.""")


# ── FULL SUMMARY VIEW ─────────────────────────────────────────────────────────
elif view == "📋 Full Summary":
    summary = generate_summary(df, s, e)
    p = summary["stats"]
    full = summary["full_stats"]

    # ── Narrative ─────────────────────────────────────────────────────────────
    st.markdown("### Auto-Generated Analysis Report")
    st.markdown(f"*Based on {p['months']} months of FRED data ({s}–{e})*")
    st.info(summary["narrative"])

    st.divider()

    # ── Stats comparison table ─────────────────────────────────────────────────
    st.markdown("### Period Comparison")
    st.caption("How the selected period compares to the full historical record (1961–2025)")

    last10 = period_stats(df["2015":"2025"])

    comparison = pd.DataFrame({
        "Metric": [
            "Avg Inflation (%)", "Peak Inflation (%)", "Trough Inflation (%)",
            "Avg Unemployment (%)", "Peak Unemployment (%)", "Min Unemployment (%)",
            "Avg Fed Funds Rate (%)", "Infl/Unemp Correlation (r)",
        ],
        f"Selected ({s}–{e})": [
            f"{p['avg_infl']:.2f}",  f"{p['max_infl']:.2f}",  f"{p['min_infl']:.2f}",
            f"{p['avg_unemp']:.2f}", f"{p['max_unemp']:.2f}", f"{p['min_unemp']:.2f}",
            f"{p['avg_ff']:.2f}",   f"{p['corr']:+.3f}",
        ],
        "Last 10 Years (2015–2025)": [
            f"{last10['avg_infl']:.2f}",  f"{last10['max_infl']:.2f}",  f"{last10['min_infl']:.2f}",
            f"{last10['avg_unemp']:.2f}", f"{last10['max_unemp']:.2f}", f"{last10['min_unemp']:.2f}",
            f"{last10['avg_ff']:.2f}",   f"{last10['corr']:+.3f}",
        ],
        "Full History (1961–2025)": [
            f"{full['avg_infl']:.2f}",  f"{full['max_infl']:.2f}",  f"{full['min_infl']:.2f}",
            f"{full['avg_unemp']:.2f}", f"{full['max_unemp']:.2f}", f"{full['min_unemp']:.2f}",
            f"{full['avg_ff']:.2f}",   f"{full['corr']:+.3f}",
        ],
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.divider()

    # ── Current state ─────────────────────────────────────────────────────────
    st.markdown("### Current State (Most Recent Reading)")
    cur = summary["current"]
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Inflation",      f"{cur['inflation']:.1f}%",
               "near target" if abs(cur['inflation'] - 2) < 1 else
               "above target" if cur['inflation'] > 2 else "below target")
    cc2.metric("Unemployment",   f"{cur['unemployment']:.1f}%",
               "historically low" if cur['unemployment'] < 4.5 else
               "elevated" if cur['unemployment'] > 6 else "moderate")
    cc3.metric("Fed Funds Rate", f"{cur['fed_funds']:.1f}%")
    cc4.metric("Infl/Unemp r",   f"{p['corr']:+.2f}",
               "inverse trade-off" if p["corr"] < -0.2 else
               "no clear link" if abs(p["corr"]) < 0.1 else "stagflation dynamics")

    st.divider()

    # ── All 5 mini charts ──────────────────────────────────────────────────────
    st.markdown("### All Charts at a Glance")

    gc1, gc2 = st.columns(2)
    with gc1:
        st.markdown("**Inflation**")
        fig = chart_inflation(df, s, e, compact=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with gc2:
        st.markdown("**Unemployment**")
        fig = chart_unemployment(df, s, e, compact=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    gc3, gc4 = st.columns(2)
    with gc3:
        st.markdown("**Phillips Curve**")
        fig = chart_phillips(df, s, e, compact=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with gc4:
        st.markdown("**Decade Summary**")
        fig = chart_decade(df, compact=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("**Macro Dashboard (all 3 indicators)**")
    fig = chart_dashboard(df, s, e)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()

    # ── Key findings ───────────────────────────────────────────────────────────
    st.markdown("### Key Findings")
    kf1, kf2 = st.columns(2)
    with kf1:
        st.markdown("""
**Inflation**
- The long-run average inflation rate is **3.8%** (1961–2025)
- The Fed's 2% target was only officially adopted in 2012
- The highest inflation was **14.6%** in March 1980
- The 1980s Volcker shock is the only successful deliberate disinflation in US history
- Core and headline inflation diverge most during **supply shocks** (1970s, 2021-22)

**Unemployment**
- The long-run average unemployment rate is **5.9%** (1961–2025)
- Unemployment peaked at **14.8%** in April 2020 (COVID-19)
- Historic low was **3.4%** in September 1968
- Every recession caused unemployment to rise by at least 1.5 percentage points
""")
    with kf2:
        st.markdown("""
**The Phillips Curve relationship**
- The classic inverse trade-off (low unemployment → high inflation) **does not always hold**
- It held best in the 2000s–2010s and the 2020s
- It broke down in the 1970s–80s due to oil supply shocks (**stagflation**)
- The 2022-23 period is notable: inflation surged without unusually high unemployment

**The Fed's role**
- The Fed Funds Rate is the primary tool to balance both mandates
- Rate hikes slow inflation but risk raising unemployment
- Rate cuts boost employment but risk stoking inflation
- The 2022-23 rate hikes (0% → 5.25%) were the fastest in 40 years
- Achieving low inflation AND low unemployment simultaneously is called a **"soft landing"** — rare but happened in 2023-24
""")

    st.success("""
**Bottom Line:** US economic history shows that inflation and unemployment are connected but not rigidly so. Supply shocks can break the normal relationship. The Fed's job is to navigate this constantly changing landscape. The data shows that while inflation has been tamed multiple times, it always requires either time, pain (higher unemployment), or both.
""")
