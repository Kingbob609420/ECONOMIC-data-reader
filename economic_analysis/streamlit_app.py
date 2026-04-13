"""
Global Economic Data Explorer
==============================
Run with:  streamlit run economic_analysis/streamlit_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
from scipy import stats

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Economic Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── US Indicators (FRED) ──────────────────────────────────────────────────────
# transform: "yoy" = year-over-year % change, None = use raw value
US_INDICATORS = {
    "Inflation (CPI YoY %)":           {"symbol": "CPIAUCSL",      "transform": "yoy", "color": "#d62728", "unit": "%"},
    "Core Inflation (YoY %)":          {"symbol": "CPILFESL",      "transform": "yoy", "color": "#ff7f0e", "unit": "%"},
    "Unemployment Rate (%)":           {"symbol": "UNRATE",         "transform": None,  "color": "#1f77b4", "unit": "%"},
    "Fed Funds Rate (%)":              {"symbol": "FEDFUNDS",       "transform": None,  "color": "#2ca02c", "unit": "%"},
    "Real GDP Growth (%)":             {"symbol": "GDPC1",          "transform": "yoy", "color": "#9467bd", "unit": "%"},
    "Industrial Production (YoY %)":   {"symbol": "INDPRO",         "transform": "yoy", "color": "#8c564b", "unit": "%"},
    "Labor Force Participation (%)":   {"symbol": "CIVPART",        "transform": None,  "color": "#e377c2", "unit": "%"},
    "10-Year Treasury Rate (%)":       {"symbol": "DGS10",          "transform": None,  "color": "#7f7f7f", "unit": "%"},
    "Consumer Sentiment (Index)":      {"symbol": "UMCSENT",        "transform": None,  "color": "#bcbd22", "unit": "idx"},
    "Housing Starts (thousands)":      {"symbol": "HOUST",          "transform": None,  "color": "#17becf", "unit": "K"},
    "Avg Hourly Earnings (YoY %)":     {"symbol": "CES0500000003",  "transform": "yoy", "color": "#aec7e8", "unit": "%"},
}

# ── Global Indicators (World Bank) ────────────────────────────────────────────
GLOBAL_INDICATORS = {
    "Inflation (annual %)":         "FP.CPI.TOTL.ZG",
    "Unemployment (% labor force)": "SL.UEM.TOTL.ZS",
    "GDP Growth (annual %)":        "NY.GDP.MKTP.KD.ZG",
    "GDP per Capita (USD)":         "NY.GDP.PCAP.CD",
}

COUNTRIES = {
    "United States": "US", "United Kingdom": "GB", "Germany": "DE",
    "Japan": "JP",         "China": "CN",          "Canada": "CA",
    "France": "FR",        "Italy": "IT",           "Brazil": "BR",
    "India": "IN",         "Australia": "AU",       "South Korea": "KR",
    "Mexico": "MX",        "Spain": "ES",           "Netherlands": "NL",
    "Sweden": "SE",        "Switzerland": "CH",     "Argentina": "AR",
    "Indonesia": "ID",     "Turkey": "TR",
}

# World Bank returns different names for some countries — map them back
WB_NAME_FIX = {
    "Korea, Rep.": "South Korea",
    "Turkiye":     "Turkey",
    "Iran, Islamic Rep.": "Iran",
    "Egypt, Arab Rep.": "Egypt",
    "Russian Federation": "Russia",
    "Venezuela, RB": "Venezuela",
}

COUNTRY_COLORS = [
    "#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
]

RECESSIONS = [
    ("1960-04","1961-02"),("1969-12","1970-11"),("1973-11","1975-03"),
    ("1980-01","1980-07"),("1981-07","1982-11"),("1990-07","1991-03"),
    ("2001-03","2001-11"),("2007-12","2009-06"),("2020-02","2020-04"),
]

PRESETS = {
    "Last 10 Years": (2015, 2025),
    "Last 20 Years": (2005, 2025),
    "Last 30 Years": (1995, 2025),
    "Post-2000":     (2000, 2025),
    "Full History":  (1961, 2025),
}

# ── Session state ─────────────────────────────────────────────────────────────
if "yr_start" not in st.session_state:
    st.session_state.yr_start = 2000
if "yr_end" not in st.session_state:
    st.session_state.yr_end = 2025

def apply_preset(s, e):
    st.session_state.yr_start = s
    st.session_state.yr_end = e

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1e2a3a; }
    [data-testid="stSidebar"] * { color: #c8d6e5 !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 14px; }
    .mode-label {
        font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
        color: #7f8c9a; text-transform: uppercase; margin: 10px 0 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

@st.cache_data(show_spinner=False)
def fetch_fred(symbol: str) -> pd.Series:
    resp = requests.get(FRED_URL.format(symbol), timeout=30)
    resp.raise_for_status()
    s = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True).squeeze()
    s = pd.to_numeric(s, errors="coerce")
    return s.resample("MS").last()


@st.cache_data(show_spinner=False)
def fetch_worldbank(indicator_code: str, country_codes: tuple) -> pd.DataFrame:
    codes = ";".join(country_codes)
    rows = []
    page = 1
    while True:
        url = (
            f"https://api.worldbank.org/v2/country/{codes}/indicator/{indicator_code}"
            f"?format=json&date=1960:2025&per_page=1000&page={page}"
        )
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=90)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"World Bank API failed after 3 attempts: {exc}"
                    ) from exc
        if len(data) < 2 or not data[1]:
            break
        for item in data[1]:
            if item.get("value") is not None:
                raw_name = item["country"]["value"]
                country  = WB_NAME_FIX.get(raw_name, raw_name)
                rows.append({
                    "country": country,
                    "year":    int(item["date"]),
                    "value":   float(item["value"]),
                })
        total_pages = data[0].get("pages", 1)
        if page >= total_pages:
            break
        page += 1

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.pivot(index="year", columns="country", values="value").sort_index()


def load_us_indicators(selected: list[str]) -> pd.DataFrame:
    frames = {}
    for name in selected:
        cfg = US_INDICATORS[name]
        raw = fetch_fred(cfg["symbol"])
        if cfg["transform"] == "yoy":
            raw = raw.pct_change(12) * 100
        frames[name] = raw
    return pd.DataFrame(frames).dropna(how="all")


# ── Helpers ───────────────────────────────────────────────────────────────────
def shade_recessions(ax, index):
    for rs, re in RECESSIONS:
        rs_dt, re_dt = pd.to_datetime(rs), pd.to_datetime(re)
        if not index.empty and rs_dt >= index[0] and re_dt <= index[-1]:
            ax.axvspan(rs_dt, re_dt, color="#d0d0d0", alpha=0.45, lw=0)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


# ── Chart builders ─────────────────────────────────────────────────────────────
def chart_us_grid(df: pd.DataFrame, selected: list[str], chart_type: str, show_recession: bool):
    n = len(selected)
    if n == 0:
        return None
    ncols = min(2, n)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")

    for i, name in enumerate(selected):
        ax = axes[i // ncols][i % ncols]
        ax.set_facecolor("#fafafa")
        cfg = US_INDICATORS[name]
        col_data = df[name].dropna()
        if col_data.empty:
            ax.set_title(name, fontsize=9)
            continue

        color = cfg["color"]
        if chart_type == "Area":
            ax.fill_between(col_data.index, col_data, alpha=0.25, color=color)
        ax.plot(col_data.index, col_data, color=color, lw=1.8)
        if show_recession:
            shade_recessions(ax, col_data.index)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.tick_params(labelsize=7)
        # Annotate min/max
        if len(col_data) > 1:
            ax.axhline(col_data.mean(), color=color, lw=0.8, ls=":", alpha=0.7)

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.tight_layout(pad=2)
    return fig


def chart_us_single(df: pd.DataFrame, selected: list[str], chart_type: str, show_recession: bool):
    """All selected indicators on one chart (normalised to z-score for comparability)."""
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    for i, name in enumerate(selected):
        col_data = df[name].dropna()
        if col_data.empty:
            continue
        # Normalise so different-unit series are comparable
        z = (col_data - col_data.mean()) / col_data.std()
        color = US_INDICATORS[name]["color"]
        if chart_type == "Area":
            ax.fill_between(z.index, z, alpha=0.12, color=color)
        ax.plot(z.index, z, color=color, lw=1.8, label=name)

    if show_recession:
        shade_recessions(ax, df.index)
    ax.axhline(0, color="#aaa", lw=0.6)
    ax.set_title("All Selected Indicators (Z-Score Normalised)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Standard deviations from mean")
    ax.legend(fontsize=8, framealpha=0.95, loc="upper left")
    ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    return fig


def chart_global(wb_df: pd.DataFrame, indicator: str, chart_type: str, s: int, e: int):
    sub = wb_df.loc[s:e]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    for i, country in enumerate(sub.columns):
        col = sub[country].dropna()
        if col.empty:
            continue
        color = COUNTRY_COLORS[i % len(COUNTRY_COLORS)]
        if chart_type == "Area":
            ax.fill_between(col.index, col, alpha=0.1, color=color)
        ax.plot(col.index, col, color=color, lw=2, marker="o", ms=3, label=country)

    ax.set_title(f"{indicator} — Country Comparison", fontsize=12, fontweight="bold")
    ax.set_ylabel(indicator)
    ax.legend(fontsize=8, framealpha=0.95, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    return fig


def chart_global_bar(wb_df: pd.DataFrame, indicator: str, year: int):
    if year not in wb_df.index:
        # Find nearest year
        year = wb_df.index[np.argmin(np.abs(wb_df.index - year))]
    row = wb_df.loc[year].dropna().sort_values(ascending=True)
    if row.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, len(row) * 0.45)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    colors = [COUNTRY_COLORS[i % len(COUNTRY_COLORS)] for i in range(len(row))]
    bars = ax.barh(row.index, row.values, color=colors, alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + abs(row.values.max()) * 0.01, bar.get_y() + bar.get_height()/2,
                f"{w:.1f}", va="center", fontsize=8)
    ax.set_title(f"{indicator} — {year}", fontsize=11, fontweight="bold")
    ax.axvline(0, color="#aaa", lw=0.7)
    ax.grid(axis="x", alpha=0.3, ls="--")
    fig.tight_layout()
    return fig


def chart_correlation(df: pd.DataFrame, x_name: str, y_name: str):
    combined = df[[x_name, y_name]].dropna()
    if len(combined) < 10:
        return None, None
    x, y = combined[x_name], combined[y_name]
    slope, intercept, r, p, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    sc = ax.scatter(x, y, c=combined.index.year if hasattr(combined.index, "year") else range(len(combined)),
                    cmap="plasma", alpha=0.55, s=18, zorder=3)
    plt.colorbar(sc, ax=ax, label="Year")
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, slope * xr + intercept, color="black", lw=2,
            label=f"r = {r:.3f},  p = {p:.4f}")
    ax.set_xlabel(x_name, fontsize=9)
    ax.set_ylabel(y_name, fontsize=9)
    ax.set_title(f"Correlation: {x_name.split('(')[0].strip()} vs\n{y_name.split('(')[0].strip()}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, ls="--")
    fig.tight_layout()

    stats_dict = {
        "Correlation (r)": f"{r:.4f}",
        "R-squared":        f"{r**2:.4f}",
        "Slope":            f"{slope:.4f}",
        "p-value":          f"{p:.6f}",
        "Significant?":     "Yes (p < 0.05)" if p < 0.05 else "No",
        "N (months)":       str(len(combined)),
        "Interpretation":   (
            "Strong inverse relationship" if r < -0.5 else
            "Moderate inverse relationship" if r < -0.2 else
            "Weak or no relationship" if abs(r) < 0.2 else
            "Moderate positive relationship" if r < 0.5 else
            "Strong positive relationship"
        ),
    }
    return fig, stats_dict


# ── In-depth US Summary ───────────────────────────────────────────────────────
def render_us_summary(df: pd.DataFrame, selected: list[str], s: int, e: int):
    """Renders a full in-depth summary directly into Streamlit."""
    full_df = df  # already sliced to [s:e] before calling

    st.markdown(f"## In-Depth US Economic Summary: {s}–{e}")
    st.caption(f"{len(df)} monthly observations · {len(selected)} indicator(s) selected · Source: FRED")
    st.divider()

    # ── Per-indicator deep dive ───────────────────────────────────────────────
    for name in selected:
        if name not in df.columns:
            continue
        col = df[name].dropna()
        if len(col) < 2:
            continue
        cfg = US_INDICATORS[name]
        unit = cfg["unit"]

        # Trend calculations
        latest       = col.iloc[-1]
        latest_dt    = col.index[-1].strftime("%B %Y")
        prev_yr      = col.iloc[-13] if len(col) > 13 else col.iloc[0]
        yr_change    = latest - prev_yr
        trend_12     = col.iloc[-12:].mean()
        trend_prev12 = col.iloc[-24:-12].mean() if len(col) >= 24 else col.iloc[:12].mean()
        direction    = "rising" if trend_12 > trend_prev12 else "falling"
        volatility   = col.std()
        hist_mean    = col.mean()
        pct_above    = (col > hist_mean).mean() * 100

        # Z-score of latest reading vs period
        z = (latest - hist_mean) / volatility if volatility > 0 else 0
        z_label = (
            "significantly above average" if z > 1.5 else
            "above average" if z > 0.5 else
            "near the period average" if abs(z) <= 0.5 else
            "below average" if z > -1.5 else
            "significantly below average"
        )

        # Momentum: last 3 months vs prior 3
        mom = col.iloc[-3:].mean() - col.iloc[-6:-3].mean() if len(col) >= 6 else 0
        mom_label = "accelerating" if mom > 0.1 else "decelerating" if mom < -0.1 else "stable"

        with st.expander(f"**{name}**  —  Latest: {latest:.2f}{unit}  |  Trend: {direction.upper()}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latest",   f"{latest:.2f}{unit}",   f"{yr_change:+.2f}{unit} vs yr ago")
            c2.metric("Average",  f"{hist_mean:.2f}{unit}")
            c3.metric("Peak",     f"{col.max():.2f}{unit}", col.idxmax().strftime("%b %Y"))
            c4.metric("Trough",   f"{col.min():.2f}{unit}", col.idxmin().strftime("%b %Y"))

            st.markdown(f"""
**What the numbers say:**

The **{name}** averaged **{hist_mean:.2f}{unit}** over the {s}–{e} period,
with a standard deviation of **{volatility:.2f}{unit}** — indicating
{"high" if volatility > hist_mean * 0.3 else "moderate" if volatility > hist_mean * 0.1 else "low"} volatility.
The most recent reading of **{latest:.2f}{unit}** ({latest_dt}) is {z_label}
(z-score: {z:+.2f}).

**Trend:** The indicator is currently **{direction}** — the 12-month average
({trend_12:.2f}{unit}) is {"above" if trend_12 > trend_prev12 else "below"} the
prior 12-month average ({trend_prev12:.2f}{unit}).
Short-term momentum is **{mom_label}** (3-month change: {mom:+.2f}{unit}).

**Historical context:** {pct_above:.0f}% of monthly readings in this period were
above the period average. The range from trough to peak spans
**{col.max() - col.min():.2f}{unit}**, suggesting
{"a very volatile period" if col.max() - col.min() > hist_mean * 1.5 else "a relatively stable period"}.
""")

            # Indicator-specific context
            if name == "Inflation (CPI YoY %)":
                gap = latest - 2.0
                st.info(f"**Fed target gap:** Inflation is currently **{abs(gap):.1f}pp {'above' if gap > 0 else 'below'}** the 2% target. "
                        f"{'The Fed is likely in tightening mode.' if gap > 1 else 'The Fed may be considering cuts.' if gap < -0.5 else 'Inflation is near target — neutral stance likely.'}")
            elif name == "Unemployment Rate (%)":
                nairu = 4.5  # rough NAIRU estimate
                st.info(f"**Labor market:** Unemployment is {'below' if latest < nairu else 'above'} the estimated natural rate (~4.5%). "
                        f"{'A tight labor market — workers have bargaining power, wages tend to rise.' if latest < nairu else 'Slack in the labor market — workers have less leverage, wage growth subdued.'}")
            elif name == "Fed Funds Rate (%)":
                st.info(f"**Policy stance:** A rate of {latest:.2f}% is considered "
                        f"{'highly restrictive — designed to slow the economy and cool inflation.' if latest > 4 else 'moderately restrictive.' if latest > 2 else 'accommodative — designed to stimulate growth.' if latest < 1 else 'roughly neutral.'}")
            elif name == "Real GDP Growth (%)":
                st.info(f"**Growth assessment:** {latest:.1f}% growth is "
                        f"{'contraction — the economy is shrinking.' if latest < 0 else 'below trend — slow recovery or stagnation.' if latest < 1.5 else 'near potential — steady expansion.' if latest < 3 else 'above trend — strong growth, may raise inflation concerns.'}")
            elif name == "10-Year Treasury Rate (%)":
                st.info(f"**Yield signal:** The 10-year at {latest:.2f}% reflects market expectations for long-run growth and inflation. "
                        f"{'High by recent standards — markets expect persistent inflation or tighter policy.' if latest > 4 else 'Low — markets expect slow growth or loose policy ahead.'}")

    st.divider()

    # ── Cross-indicator analysis ──────────────────────────────────────────────
    st.markdown("### Cross-Indicator Analysis")

    has_infl  = "Inflation (CPI YoY %)"    in selected and "Inflation (CPI YoY %)"    in df.columns
    has_unemp = "Unemployment Rate (%)"     in selected and "Unemployment Rate (%)"     in df.columns
    has_fed   = "Fed Funds Rate (%)"        in selected and "Fed Funds Rate (%)"        in df.columns
    has_gdp   = "Real GDP Growth (%)"       in selected and "Real GDP Growth (%)"       in df.columns
    has_wages = "Avg Hourly Earnings (YoY %)" in selected and "Avg Hourly Earnings (YoY %)" in df.columns

    if has_infl and has_unemp:
        r_pc = df["Inflation (CPI YoY %)"].corr(df["Unemployment Rate (%)"])
        pc_label = (
            "strongly holds — low unemployment is clearly associated with higher inflation"   if r_pc < -0.5 else
            "moderately holds — some inverse relationship present"                            if r_pc < -0.2 else
            "weak or absent — supply-side forces may be dominating"                           if abs(r_pc) < 0.2 else
            "reversed — both rising or falling together (stagflation or disinflation dynamics)"
        )
        st.markdown(f"""
**Phillips Curve (Inflation vs. Unemployment):** r = **{r_pc:+.3f}**

The trade-off {pc_label}. {"A negative correlation is the textbook expectation — tight labor markets give workers wage bargaining power, pushing up prices." if r_pc < 0 else "A positive correlation is unusual and typically signals an external shock (e.g. oil embargo, pandemic supply disruption) is driving both indicators simultaneously."}
""")

    if has_infl and has_fed:
        r_fi = df["Fed Funds Rate (%)"].corr(df["Inflation (CPI YoY %)"])
        st.markdown(f"""
**Fed Policy vs. Inflation:** r = **{r_fi:+.3f}**

{"The Fed has been raising rates alongside rising inflation — reactive but expected policy." if r_fi > 0.3 else "Weak or negative correlation — the Fed may have been cutting rates despite inflation, or the two moved independently."}
The Fed typically targets a 'real rate' (Fed Funds Rate minus Inflation) above zero to be restrictive. Currently: **{df["Fed Funds Rate (%)"].iloc[-1] - df["Inflation (CPI YoY %)"].iloc[-1]:.2f}pp** real rate.
""")

    if has_gdp and has_unemp:
        r_gu = df["Real GDP Growth (%)"].corr(df["Unemployment Rate (%)"])
        st.markdown(f"""
**Okun's Law (GDP Growth vs. Unemployment):** r = **{r_gu:+.3f}**

{"Strong inverse relationship — GDP growth is reducing unemployment as expected (Okun's Law holding)." if r_gu < -0.3 else "Weaker than expected — growth may be happening without proportionate job creation ('jobless recovery' pattern)."}
""")

    if has_wages and has_infl:
        r_wi = df["Avg Hourly Earnings (YoY %)"].corr(df["Inflation (CPI YoY %)"])
        st.markdown(f"""
**Wage-Price Spiral Risk:** r = **{r_wi:+.3f}**

{"Wages and inflation are moving together — a wage-price spiral risk where higher wages fuel higher prices which demand higher wages." if r_wi > 0.4 else "Wages and inflation are not strongly correlated in this period — workers may be losing real purchasing power if inflation is outpacing wages."}
""")

    if not any([has_infl and has_unemp, has_infl and has_fed, has_gdp and has_unemp]):
        st.info("Select more indicators (e.g. Inflation + Unemployment + Fed Funds Rate) to unlock cross-indicator analysis.")

    st.divider()

    # ── Period verdict ────────────────────────────────────────────────────────
    st.markdown("### Overall Period Assessment")
    verdicts = []
    if has_infl:
        avg_i = df["Inflation (CPI YoY %)"].mean()
        verdicts.append(f"**Inflation:** {'High-inflation period' if avg_i > 5 else 'Moderate-inflation period' if avg_i > 3 else 'Low-inflation period' if avg_i > 1 else 'Deflationary risk period'} (avg {avg_i:.1f}%)")
    if has_unemp:
        avg_u = df["Unemployment Rate (%)"].mean()
        verdicts.append(f"**Labor Market:** {'Tight — strong employment' if avg_u < 5 else 'Moderate — near full employment' if avg_u < 6.5 else 'Weak — elevated unemployment'} (avg {avg_u:.1f}%)")
    if has_gdp:
        avg_g = df["Real GDP Growth (%)"].mean()
        verdicts.append(f"**Growth:** {'Expansion — above trend growth' if avg_g > 2.5 else 'Moderate growth' if avg_g > 1 else 'Near stagnation or contraction'} (avg {avg_g:.1f}%)")
    if has_fed:
        avg_f = df["Fed Funds Rate (%)"].mean()
        verdicts.append(f"**Monetary Policy:** {'Restrictive — high rates' if avg_f > 4 else 'Neutral — moderate rates' if avg_f > 1.5 else 'Accommodative — low rates'} (avg {avg_f:.1f}%)")

    for v in verdicts:
        st.markdown(f"- {v}")

    if has_infl and has_unemp:
        avg_i = df["Inflation (CPI YoY %)"].mean()
        avg_u = df["Unemployment Rate (%)"].mean()
        if avg_i < 4 and avg_u < 5.5:
            st.success(f"**Period Verdict:** This was a **Goldilocks period** — both inflation and unemployment were relatively low and stable. Ideal macroeconomic conditions.")
        elif avg_i > 5 and avg_u > 6:
            st.error(f"**Period Verdict:** This period showed **stagflation characteristics** — high inflation combined with high unemployment. The worst-case scenario for policymakers.")
        elif avg_i > 5:
            st.warning(f"**Period Verdict:** This was a **high-inflation period**. Price stability was the dominant policy challenge.")
        elif avg_u > 7:
            st.warning(f"**Period Verdict:** This was a **high-unemployment period**. Labor market weakness was the dominant concern.")
        else:
            st.info(f"**Period Verdict:** Mixed conditions — some indicators elevated, others stable. A period of transition or policy adjustment.")


# ── In-depth Global Summary ───────────────────────────────────────────────────
def render_global_summary(wb_df: pd.DataFrame, indicator: str, s: int, e: int, selected_countries: list[str]):
    sub = wb_df.loc[s:e].dropna(how="all")
    if sub.empty:
        st.warning("No data available for this selection.")
        return

    latest_year  = sub.dropna(how="all").index[-1]
    earliest_year = sub.dropna(how="all").index[0]
    latest_row   = sub.loc[latest_year].dropna()
    available    = [c for c in selected_countries if c in sub.columns]

    st.markdown(f"## In-Depth Global Summary: {indicator}")
    st.caption(f"Countries: {', '.join(available)} · Period: {earliest_year}–{latest_year} · Source: World Bank")
    st.divider()

    # ── Global snapshot ───────────────────────────────────────────────────────
    st.markdown(f"### Snapshot: {latest_year}")
    if not latest_row.empty:
        ranked     = latest_row.sort_values(ascending=False)
        global_avg = latest_row.mean()
        spread     = latest_row.max() - latest_row.min()

        snap_c1, snap_c2, snap_c3, snap_c4 = st.columns(4)
        snap_c1.metric("Group Average",  f"{global_avg:.2f}")
        snap_c2.metric("Highest",  f"{ranked.index[0]}", f"{ranked.iloc[0]:.2f}")
        snap_c3.metric("Lowest",   f"{ranked.index[-1]}", f"{ranked.iloc[-1]:.2f}")
        snap_c4.metric("Spread (High–Low)", f"{spread:.2f}")

        st.markdown(f"""
In **{latest_year}**, the average **{indicator}** across selected countries was **{global_avg:.2f}**.
The gap between the highest (**{ranked.index[0]}** at {ranked.iloc[0]:.2f}) and lowest
(**{ranked.index[-1]}** at {ranked.iloc[-1]:.2f}) was **{spread:.2f}** — indicating
{"extreme divergence between economies" if spread > global_avg * 2 else "significant but manageable variation" if spread > global_avg else "relatively convergent performance across countries"}.
""")

    st.divider()

    # ── Per-country deep dive ─────────────────────────────────────────────────
    st.markdown("### Country-by-Country Analysis")
    for country in available:
        col_data = sub[country].dropna()
        if len(col_data) < 2:
            continue

        latest_val   = col_data.iloc[-1]
        latest_yr    = col_data.index[-1]
        earliest_val = col_data.iloc[0]
        earliest_yr  = col_data.index[0]
        avg_val      = col_data.mean()
        peak_val     = col_data.max()
        peak_yr      = col_data.idxmax()
        trough_val   = col_data.min()
        trough_yr    = col_data.idxmin()
        total_change = latest_val - earliest_val
        volatility   = col_data.std()

        # Trend over last 5 years
        recent = col_data.iloc[-5:] if len(col_data) >= 5 else col_data
        prior  = col_data.iloc[-10:-5] if len(col_data) >= 10 else col_data
        trend  = "improving" if recent.mean() < prior.mean() else "worsening"
        # For GDP Growth / GDP per capita, higher = better; invert for inflation/unemployment
        if "GDP" in indicator:
            trend = "improving" if recent.mean() > prior.mean() else "worsening"

        # Rank in latest year
        if not latest_row.empty and country in latest_row.index:
            rank_asc  = int((latest_row < latest_val).sum()) + 1
            rank_desc = int((latest_row > latest_val).sum()) + 1
            n_countries = len(latest_row)
            rank_label = f"#{rank_desc} highest of {n_countries}" if "GDP" in indicator else f"#{rank_asc} lowest of {n_countries}"
        else:
            rank_label = "N/A"

        with st.expander(f"**{country}** — {latest_yr}: {latest_val:.2f}  |  Trend: {trend.upper()}", expanded=False):
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Latest",  f"{latest_val:.2f}",  f"({latest_yr})")
            cc2.metric("Average", f"{avg_val:.2f}")
            cc3.metric("Peak",    f"{peak_val:.2f}",    f"({peak_yr})")
            cc4.metric("Trough",  f"{trough_val:.2f}",  f"({trough_yr})")

            st.markdown(f"""
**{country}** recorded a **{indicator}** of **{latest_val:.2f}** in {latest_yr},
ranking **{rank_label}** among selected countries.

Over the {earliest_yr}–{latest_yr} period, the indicator averaged **{avg_val:.2f}**
with a standard deviation of **{volatility:.2f}** —
{"highly volatile" if volatility > avg_val * 0.5 else "moderately volatile" if volatility > avg_val * 0.2 else "relatively stable"}.
The total change from {earliest_yr} to {latest_yr} was **{total_change:+.2f}**
({"improving" if (total_change < 0 and "Inflation" in indicator) or (total_change < 0 and "Unemployment" in indicator) or (total_change > 0 and "GDP" in indicator) else "worsening"} direction).

The peak of **{peak_val:.2f}** occurred in **{peak_yr}**
and the trough of **{trough_val:.2f}** in **{trough_yr}**.
Recent 5-year trend is **{trend}** compared to the prior 5-year period.
""")

            # Country-specific context callouts
            if indicator == "Inflation (annual %)":
                if latest_val > 20:
                    st.error(f"**Crisis-level inflation.** At {latest_val:.1f}%, {country} is experiencing hyperinflationary or near-hyperinflationary conditions. This destroys purchasing power rapidly and typically signals deep fiscal or monetary problems.")
                elif latest_val > 8:
                    st.warning(f"**High inflation.** {latest_val:.1f}% is well above the typical 2% target for developed economies. Central bank is likely under significant pressure to tighten.")
                elif latest_val < 0:
                    st.warning(f"**Deflation.** Negative inflation sounds good but is dangerous — it encourages people to delay purchases expecting lower prices, which slows the economy.")
                else:
                    st.success(f"**Stable inflation.** {latest_val:.1f}% is within a broadly healthy range.")

            elif indicator == "Unemployment (% labor force)":
                if latest_val > 15:
                    st.error(f"**Severe unemployment.** At {latest_val:.1f}%, {country} faces a serious structural labor market challenge with major social and economic consequences.")
                elif latest_val > 8:
                    st.warning(f"**Elevated unemployment.** {latest_val:.1f}% suggests significant slack in the labor market. Growth is not translating into enough jobs.")
                elif latest_val < 4:
                    st.success(f"**Near full employment.** {latest_val:.1f}% is historically very low. Workers have strong bargaining power; wage growth likely.")
                else:
                    st.info(f"**Moderate unemployment.** {latest_val:.1f}% is near the typical range for developed economies.")

            elif indicator == "GDP Growth (annual %)":
                if latest_val < 0:
                    st.error(f"**Recession.** Negative GDP growth ({latest_val:.1f}%) means the economy is contracting — output, incomes, and employment are all declining.")
                elif latest_val < 1:
                    st.warning(f"**Near stagnation.** {latest_val:.1f}% is barely growing — not enough to absorb new workers or raise living standards.")
                elif latest_val > 6:
                    st.success(f"**Rapid expansion.** {latest_val:.1f}% growth is very strong — typically seen in emerging markets catching up, or post-recession rebounds.")
                else:
                    st.info(f"**Steady growth.** {latest_val:.1f}% is within the normal healthy range for an established economy.")

            elif indicator == "GDP per Capita (USD)":
                if latest_val > 50000:
                    st.success(f"**High-income economy.** ${latest_val:,.0f} per capita places {country} among the wealthiest nations.")
                elif latest_val > 15000:
                    st.info(f"**Upper-middle-income economy.** ${latest_val:,.0f} per capita — significant development achieved but gap remains vs. top economies.")
                elif latest_val > 4000:
                    st.warning(f"**Lower-middle-income economy.** ${latest_val:,.0f} per capita — large share of population still in poverty by global standards.")
                else:
                    st.error(f"**Low-income economy.** ${latest_val:,.0f} per capita — significant development challenges.")

    st.divider()

    # ── Comparative trends ────────────────────────────────────────────────────
    st.markdown("### Comparative Trends Across Countries")

    if len(available) >= 2:
        # Fastest improver / deteriorator
        changes = {}
        for country in available:
            col_data = sub[country].dropna()
            if len(col_data) >= 2:
                changes[country] = col_data.iloc[-1] - col_data.iloc[0]

        if changes:
            best_country = min(changes, key=changes.get) if "Inflation" in indicator or "Unemployment" in indicator else max(changes, key=changes.get)
            worst_country = max(changes, key=changes.get) if "Inflation" in indicator or "Unemployment" in indicator else min(changes, key=changes.get)
            st.markdown(f"""
**Biggest improvement** over {earliest_year}–{latest_year}: **{best_country}**
(change: {changes[best_country]:+.2f})

**Biggest deterioration** over {earliest_year}–{latest_year}: **{worst_country}**
(change: {changes[worst_country]:+.2f})
""")

        # Convergence / divergence
        early_std = sub.loc[earliest_year:earliest_year+4].std(axis=1).mean() if len(sub) > 5 else None
        recent_std = sub.iloc[-5:].std(axis=1).mean()
        if early_std is not None:
            if recent_std < early_std * 0.8:
                st.info(f"**Convergence:** Countries have become MORE similar over time (spread narrowed from ~{early_std:.2f} to ~{recent_std:.2f}). This suggests shared global forces or policy coordination.")
            elif recent_std > early_std * 1.2:
                st.warning(f"**Divergence:** Countries have become MORE different over time (spread widened from ~{early_std:.2f} to ~{recent_std:.2f}). Different policy choices or structural factors are pulling economies apart.")
            else:
                st.info(f"**Stable spread:** The variation between countries has remained broadly consistent (~{recent_std:.2f}).")

    st.divider()

    # ── Overall verdict ───────────────────────────────────────────────────────
    st.markdown("### Overall Period Assessment")
    if not latest_row.empty:
        avg = latest_row.mean()
        high_performers = [c for c in latest_row.index if (
            (latest_row[c] < avg and ("Inflation" in indicator or "Unemployment" in indicator)) or
            (latest_row[c] > avg and "GDP" in indicator)
        )]
        low_performers = [c for c in latest_row.index if c not in high_performers]

        if high_performers:
            st.success(f"**Outperforming in {latest_year}:** {', '.join(high_performers)}")
        if low_performers:
            st.warning(f"**Underperforming in {latest_year}:** {', '.join(low_performers)}")

        st.markdown(f"""
The **{indicator}** data for {s}–{e} reveals {"significant" if latest_row.std() > latest_row.mean() * 0.5 else "moderate"} variation
across selected countries. With a group average of **{avg:.2f}** in {latest_year}, the data
shows that economic outcomes are {"highly divergent — country-specific factors dominate" if latest_row.std() > latest_row.mean() * 0.5 else "broadly aligned — global forces are the dominant driver"}.
""")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Economic\nData Explorer")
    st.divider()

    mode = st.radio("Mode", ["🇺🇸 US Economy", "🌍 Country Comparison"], label_visibility="collapsed")
    st.divider()

    if mode == "🇺🇸 US Economy":
        st.markdown("**Select Indicators**")
        selected_indicators = []
        for name in US_INDICATORS:
            default = name in ("Inflation (CPI YoY %)", "Unemployment Rate (%)")
            if st.checkbox(name, value=default, key=f"chk_{name}"):
                selected_indicators.append(name)

        st.divider()
        chart_layout = st.radio("Chart layout", ["Grid (one each)", "Overlay (normalised)"])
        chart_type   = st.radio("Chart style",  ["Line", "Area"])
        show_rec     = st.checkbox("Show recessions", value=True)

    else:  # Global mode
        st.markdown("**Select Indicator**")
        global_indicator = st.selectbox(
            "Indicator", list(GLOBAL_INDICATORS.keys()), label_visibility="collapsed")

        st.markdown("**Select Countries**")
        default_countries = ["United States", "United Kingdom", "Germany", "Japan", "China"]
        selected_countries = st.multiselect(
            "Countries", list(COUNTRIES.keys()),
            default=default_countries, label_visibility="collapsed")

        chart_type = st.radio("Chart style", ["Line", "Area"])

    st.divider()
    st.markdown("**Time Period**")
    for label, (ps, pe) in PRESETS.items():
        active = st.session_state.yr_start == ps and st.session_state.yr_end == pe
        st.button(f"{'✅ ' if active else ''}{label}", on_click=apply_preset,
                  args=(ps, pe), use_container_width=True, key=f"p_{label}")

    st.caption("Custom range:")
    year_range = st.slider("", min_value=1961, max_value=2025,
                           value=(st.session_state.yr_start, st.session_state.yr_end),
                           label_visibility="collapsed")
    st.session_state.yr_start = year_range[0]
    st.session_state.yr_end   = year_range[1]

    st.divider()
    st.caption("Source: FRED (US) · World Bank (Global)")


# ── Main ──────────────────────────────────────────────────────────────────────
s, e = st.session_state.yr_start, st.session_state.yr_end

# ════════════════════════════════════════
# US ECONOMY MODE
# ════════════════════════════════════════
if mode == "🇺🇸 US Economy":

    st.markdown(f"## 🇺🇸 US Economy Explorer  `{s}–{e}`")

    if not selected_indicators:
        st.warning("Select at least one indicator in the sidebar.")
        st.stop()

    with st.spinner("Loading FRED data..."):
        df_all = load_us_indicators(selected_indicators)

    df = df_all[str(s):str(e)]

    # ── Metrics row ───────────────────────────────────────────────────────────
    metric_cols = st.columns(min(len(selected_indicators), 4))
    for i, name in enumerate(selected_indicators[:4]):
        col_data = df[name].dropna()
        if col_data.empty:
            continue
        latest = col_data.iloc[-1]
        prev   = col_data.iloc[-13] if len(col_data) > 13 else col_data.iloc[0]
        cfg    = US_INDICATORS[name]
        metric_cols[i].metric(
            label=name.split("(")[0].strip(),
            value=f"{latest:.1f}{cfg['unit']}",
            delta=f"{latest - prev:+.1f}{cfg['unit']} vs yr ago",
        )

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_charts, tab_table, tab_corr, tab_summary = st.tabs(
        ["📈 Charts", "📊 Data Table", "🔗 Correlations", "📋 Summary"])

    with tab_charts:
        if chart_layout == "Grid (one each)":
            fig = chart_us_grid(df, selected_indicators, chart_type, show_rec)
        else:
            fig = chart_us_single(df, selected_indicators, chart_type, show_rec)
        if fig:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if show_rec:
            st.caption("Gray bands = NBER-defined US recessions")

    with tab_table:
        st.markdown("### Data Table")
        display_df = df[selected_indicators].copy()
        display_df.index = display_df.index.strftime("%Y-%m")
        display_df = display_df.round(2)

        # Summary stats above table
        st.markdown("**Summary Statistics**")
        st.dataframe(
            df[selected_indicators].describe().round(2),
            use_container_width=True,
        )
        st.markdown("**Monthly Data**")
        st.dataframe(display_df.sort_index(ascending=False), use_container_width=True)

        st.download_button(
            label="⬇️ Download as CSV",
            data=to_csv_bytes(df[selected_indicators]),
            file_name=f"us_economic_data_{s}_{e}.csv",
            mime="text/csv",
        )

    with tab_corr:
        st.markdown("### Correlation Analysis")
        if len(selected_indicators) < 2:
            st.info("Select at least 2 indicators in the sidebar to run correlation analysis.")
        else:
            cc1, cc2 = st.columns(2)
            x_ind = cc1.selectbox("X axis (independent)", selected_indicators, index=0)
            y_ind = cc2.selectbox("Y axis (dependent)",   selected_indicators,
                                  index=min(1, len(selected_indicators)-1))

            if x_ind == y_ind:
                st.warning("Please select two different indicators.")
            else:
                fig, stat_dict = chart_correlation(df, x_ind, y_ind)
                if fig:
                    col_fig, col_stats = st.columns([3, 2])
                    with col_fig:
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    with col_stats:
                        st.markdown("**Regression Statistics**")
                        stats_df = pd.DataFrame(
                            stat_dict.items(), columns=["Metric", "Value"])
                        st.dataframe(stats_df, hide_index=True, use_container_width=True)

                        interp = stat_dict["Interpretation"]
                        r_val  = float(stat_dict["Correlation (r)"])
                        if r_val < -0.3:
                            st.success(f"**{interp}** — as one rises, the other tends to fall.")
                        elif r_val > 0.3:
                            st.warning(f"**{interp}** — both tend to move in the same direction.")
                        else:
                            st.info(f"**{interp}** — no consistent directional pattern.")

                # Correlation matrix if 3+ indicators
                if len(selected_indicators) >= 3:
                    st.markdown("### Full Correlation Matrix")
                    corr_matrix = df[selected_indicators].corr().round(3)
                    st.dataframe(corr_matrix.style.background_gradient(
                        cmap="RdYlGn", vmin=-1, vmax=1), use_container_width=True)

    with tab_summary:
        render_us_summary(df, selected_indicators, s, e)


# ════════════════════════════════════════
# GLOBAL COMPARISON MODE
# ════════════════════════════════════════
else:
    st.markdown(f"## 🌍 Global Comparison: {global_indicator}  `{s}–{e}`")

    if not selected_countries:
        st.warning("Select at least one country in the sidebar.")
        st.stop()

    country_codes = tuple(COUNTRIES[c] for c in selected_countries if c in COUNTRIES)

    with st.spinner("Loading World Bank data... (may take a few seconds)"):
        try:
            wb_df = fetch_worldbank(GLOBAL_INDICATORS[global_indicator], country_codes)
        except RuntimeError as exc:
            st.error(f"**Could not reach World Bank API.** Check your internet connection and try again.\n\n`{exc}`")
            st.stop()

    if wb_df.empty:
        st.warning("World Bank returned no data for this combination. Try selecting fewer countries or a different indicator.")
        st.stop()

    # Rename columns to full country names
    code_to_name = {v: k for k, v in COUNTRIES.items()}

    # ── Metrics ───────────────────────────────────────────────────────────────
    sub_wb = wb_df.loc[s:e] if s in wb_df.index or e in wb_df.index else wb_df
    latest_year = sub_wb.dropna(how="all").index[-1]
    latest_row  = sub_wb.loc[latest_year].dropna()

    metric_cols = st.columns(min(len(selected_countries), 5))
    for i, country in enumerate(selected_countries[:5]):
        if country in latest_row.index:
            metric_cols[i].metric(
                label=country,
                value=f"{latest_row[country]:.1f}",
                delta=f"({latest_year})",
            )

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_line, tab_bar, tab_table, tab_summary = st.tabs(
        ["📈 Trend Chart", "📊 Country Comparison Bar", "🗂️ Data Table", "📋 Summary"])

    with tab_line:
        fig = chart_global(wb_df, global_indicator, chart_type, s, e)
        if fig:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.caption("Source: World Bank. Annual data.")

    with tab_bar:
        st.markdown("**Compare countries at a specific year**")
        available_years = sorted(wb_df.loc[s:e].dropna(how="all").index.tolist(), reverse=True)
        if available_years:
            bar_year = st.select_slider("Select year", options=available_years,
                                        value=available_years[0])
            fig = chart_global_bar(wb_df.loc[s:e], global_indicator, bar_year)
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    with tab_table:
        st.markdown("### Data Table")
        display_wb = wb_df.loc[s:e].round(2).sort_index(ascending=False)

        st.markdown("**Summary Statistics**")
        st.dataframe(display_wb.describe().round(2), use_container_width=True)

        st.markdown("**Annual Data**")
        st.dataframe(display_wb, use_container_width=True)

        st.download_button(
            label="⬇️ Download as CSV",
            data=to_csv_bytes(wb_df.loc[s:e]),
            file_name=f"global_{global_indicator.replace(' ', '_')}_{s}_{e}.csv",
            mime="text/csv",
        )

    with tab_summary:
        render_global_summary(wb_df, global_indicator, s, e, selected_countries)
