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
    url = (
        f"https://api.worldbank.org/v2/country/{codes}/indicator/{indicator_code}"
        f"?format=json&date=1960:2025&per_page=5000"
    )
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return pd.DataFrame()
    if len(data) < 2 or not data[1]:
        return pd.DataFrame()
    rows = []
    for item in data[1]:
        if item.get("value") is not None:
            rows.append({
                "country": item["country"]["value"],
                "year":    int(item["date"]),
                "value":   float(item["value"]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
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


# ── Auto summary ──────────────────────────────────────────────────────────────
def build_us_summary(df: pd.DataFrame, selected: list[str], s: int, e: int) -> str:
    lines = [f"## Economic Summary: {s}–{e}\n"]
    lines.append(f"*{len(df)} monthly observations across {len(selected)} indicator(s)*\n")
    for name in selected:
        if name not in df.columns:
            continue
        col = df[name].dropna()
        if col.empty:
            continue
        cfg = US_INDICATORS[name]
        trend = "rising" if col.iloc[-12:].mean() > col.iloc[-24:-12].mean() else "falling"
        lines.append(f"### {name}")
        lines.append(f"- **Average:** {col.mean():.2f}{cfg['unit']}")
        lines.append(f"- **Peak:** {col.max():.2f}{cfg['unit']} ({col.idxmax().strftime('%b %Y')})")
        lines.append(f"- **Trough:** {col.min():.2f}{cfg['unit']} ({col.idxmin().strftime('%b %Y')})")
        lines.append(f"- **Recent trend:** {trend}")
        lines.append(f"- **Latest reading:** {col.iloc[-1]:.2f}{cfg['unit']} ({col.index[-1].strftime('%b %Y')})\n")
    return "\n".join(lines)


def build_global_summary(wb_df: pd.DataFrame, indicator: str, s: int, e: int) -> str:
    sub = wb_df.loc[s:e].dropna(how="all")
    if sub.empty:
        return "No data available."
    latest_year = sub.dropna(how="all").index[-1]
    latest = sub.loc[latest_year].dropna().sort_values()
    lines = [f"## Global Summary: {indicator} ({s}–{e})\n"]
    lines.append(f"**Latest available year: {latest_year}**\n")
    if not latest.empty:
        lines.append(f"- Lowest:  **{latest.index[0]}** at {latest.iloc[0]:.2f}")
        lines.append(f"- Highest: **{latest.index[-1]}** at {latest.iloc[-1]:.2f}")
        lines.append(f"- Average across selected countries: {latest.mean():.2f}\n")
    lines.append("### Country-by-country (latest year)\n")
    for country, val in latest.sort_values(ascending=False).items():
        lines.append(f"- **{country}:** {val:.2f}")
    return "\n".join(lines)


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
        st.markdown(build_us_summary(df, selected_indicators, s, e))

        # Key insights box
        st.divider()
        st.markdown("### How to interpret this data")
        if "Inflation (CPI YoY %)" in selected_indicators and "Unemployment Rate (%)" in selected_indicators:
            r_val = df["Inflation (CPI YoY %)"].corr(df["Unemployment Rate (%)"])
            direction = "inverse (Phillips Curve holds)" if r_val < -0.1 \
                else "positive (stagflation dynamics)" if r_val > 0.1 \
                else "no clear relationship"
            st.info(f"""**Phillips Curve check for {s}–{e}:**
Inflation vs. Unemployment correlation = **{r_val:+.3f}** → {direction}

The classic Phillips Curve predicts a negative correlation (when unemployment is low, inflation rises).
A positive correlation suggests supply shocks are overriding normal demand-side dynamics.""")

        if "Fed Funds Rate (%)" in selected_indicators and "Inflation (CPI YoY %)" in selected_indicators:
            r_val2 = df["Fed Funds Rate (%)"].corr(df["Inflation (CPI YoY %)"])
            st.info(f"""**Fed policy check for {s}–{e}:**
Fed Funds Rate vs. Inflation correlation = **{r_val2:+.3f}**

A positive correlation means the Fed raised rates when inflation rose (reactive policy).
A negative correlation would suggest the Fed was ahead of the curve.""")


# ════════════════════════════════════════
# GLOBAL COMPARISON MODE
# ════════════════════════════════════════
else:
    st.markdown(f"## 🌍 Global Comparison: {global_indicator}  `{s}–{e}`")

    if not selected_countries:
        st.warning("Select at least one country in the sidebar.")
        st.stop()

    country_codes = tuple(COUNTRIES[c] for c in selected_countries if c in COUNTRIES)

    with st.spinner("Loading World Bank data..."):
        wb_df = fetch_worldbank(GLOBAL_INDICATORS[global_indicator], country_codes)

    if wb_df.empty:
        st.error("No data returned from World Bank API. Try a different indicator or country selection.")
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
        st.markdown(build_global_summary(wb_df, global_indicator, s, e))

        # Rankings table
        if not latest_row.empty:
            st.markdown(f"### Country Rankings ({latest_year})")
            rank_df = latest_row.sort_values(ascending=False).reset_index()
            rank_df.columns = ["Country", global_indicator]
            rank_df.insert(0, "Rank", range(1, len(rank_df)+1))
            rank_df[global_indicator] = rank_df[global_indicator].round(2)
            st.dataframe(rank_df, hide_index=True, use_container_width=True)
