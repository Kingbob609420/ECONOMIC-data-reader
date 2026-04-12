"""
US Economic Data Dashboard
==========================
Interactive GUI for exploring inflation, unemployment, and their relationship.
Run with:  python economic_analysis/dashboard.py
"""

import sys
import threading
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pandas_datareader.data as web
from scipy import stats

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

# UI colours
SIDEBAR_BG   = "#1e2a3a"
SIDEBAR_FG   = "#c8d6e5"
BTN_ACTIVE   = "#2e86de"
BTN_HOVER    = "#2c3e56"
CONTENT_BG   = "#f4f6f9"
EXPL_BG      = "#ffffff"
TITLE_FG     = "#ffffff"

# ── Explanations ──────────────────────────────────────────────────────────────
EXPLANATIONS = {
    "Inflation Trends": """\
WHAT YOU'RE LOOKING AT
This chart shows US inflation from 1960 to 2025, measured as the year-over-year
percentage change in the Consumer Price Index (CPI).

Two lines are shown:
  - Headline Inflation  (red)  - all goods including food & energy
  - Core Inflation      (orange, dashed) - excludes volatile food & energy prices
  - 2% Target           (dotted line) - the Federal Reserve's goal since the 1990s

Gray bands = NBER-defined US recessions.

KEY MOMENTS TO NOTICE
  - 1973-74: Oil embargo causes inflation to surge past 10%
  - 1980:    Inflation peaks at 14.6% - the highest in modern history
  - 1983-2019: "The Great Moderation" - inflation stays relatively low & stable
  - 2021-22: Post-pandemic supply shocks push inflation to 9%, a 40-year high
  - 2023-24: Fed rate hikes bring inflation back toward the 2% target

WHY CORE INFLATION MATTERS
Core CPI strips out food and energy, which are very volatile. Economists and the
Fed focus on core inflation to understand the underlying trend - if only food
prices spike (e.g. a bad harvest), that tells a different story than broad
price increases across the economy.""",

    "Unemployment": """\
WHAT YOU'RE LOOKING AT
This chart shows the US civilian unemployment rate from 1960 to 2025 - the
percentage of people actively looking for work but unable to find it.

The dashed line is a 24-month rolling average, which smooths out noise and
shows the longer-term trend more clearly.

Gray bands = NBER-defined US recessions.

KEY MOMENTS TO NOTICE
  - 1968:  Unemployment hits a historic low of 3.4% during the Vietnam-era boom
  - 1982:  Unemployment peaks at 10.8% - the worst since the Great Depression -
           due to the Volcker Fed deliberately raising interest rates to kill
           the 1970s inflation
  - 2009:  Financial Crisis drives unemployment to 10%
  - 2020:  COVID-19 causes the fastest spike ever: 3.5% to 14.8% in 2 months,
           then the fastest recovery ever back below 4% by 2022

THE PATTERN
Unemployment rises sharply during recessions (gray bands) and falls slowly
during recoveries. Economists call this "asymmetric" - it takes years to build
jobs back after a recession, but only months to lose them.""",

    "Macro Dashboard": """\
WHAT YOU'RE LOOKING AT
Three indicators stacked together to show how they interact:

  Panel 1 - Inflation:       Year-over-year CPI change (%)
  Panel 2 - Unemployment:    Civilian unemployment rate (%)
  Panel 3 - Fed Funds Rate:  The interest rate the Federal Reserve sets (%)

The Fed Funds Rate is the main policy tool the Fed uses to manage the economy.

HOW THEY CONNECT
  When inflation rises -> The Fed RAISES rates to cool spending -> This slows
  growth -> Unemployment tends to RISE (people lose jobs)

  When unemployment rises -> The Fed LOWERS rates to stimulate spending ->
  This encourages hiring -> Inflation may RISE

This tension is called the "dual mandate" - the Fed must balance low inflation
AND low unemployment at the same time.

THE 1970s-80s STORY
Look at the 1970s: inflation soared, and the Fed raised rates massively (to
19%!). This caused the 1981-82 recession and spiked unemployment to 10.8%.
But it worked - inflation was broken and stayed low for 40 years.

THE 2022-23 STORY
Same playbook: inflation hit 9% in 2022, the Fed raised rates from 0% to 5.25%
in 18 months. Unemployment stayed low this time - a historically rare "soft
landing".""",

    "Phillips Curve": """\
WHAT YOU'RE LOOKING AT
The Phillips Curve is one of the most famous ideas in economics.

In 1958, economist A.W. Phillips noticed: when unemployment is LOW, inflation
tends to be HIGH, and vice versa. The logic: when jobs are plentiful, workers
can demand higher wages, which pushes up prices.

LEFT CHART - Full Period (1960-2025)
Each dot is one month. Color shows the year (dark = older, bright = recent).
The black line is the overall trend.

RIGHT CHART - By Era
The relationship is NOT stable over time. Each era has its own curve:
  - Red   (1960s-70s): Stagflation broke the rule - BOTH rose together
  - Blue  (1980s-90s): Disinflation - both fell together as Volcker's medicine worked
  - Green (2000s-10s): Classic inverse relationship returns
  - Orange (2020s):   Strongest inverse link in decades

WHAT THE NUMBERS MEAN
  r = correlation coefficient (-1 to +1)
  r = -1 means perfect inverse trade-off (textbook Phillips Curve)
  r = +1 means both move in the same direction (stagflation)
  r near 0 means no clear relationship

The 1970s stagflation (oil shocks) showed that supply disruptions can break
the Phillips Curve entirely - you can have both high inflation AND high
unemployment at the same time.""",

    "Decade Summary": """\
WHAT YOU'RE LOOKING AT
Average inflation and unemployment rates grouped by decade.

This gives you a bird's-eye view of how each era compares.

  Red bars   = Average Inflation (%)
  Blue bars  = Average Unemployment (%)

DECADE BY DECADE

  1960s: Low unemployment (~4.8%), moderate inflation (~2.3%) - economic boom,
         Kennedy/LBJ Great Society spending, Vietnam War buildup

  1970s: Inflation explodes (~7.1%) from oil shocks and loose monetary policy.
         Unemployment also rises - this "stagflation" was a policy nightmare.

  1980s: Volcker's aggressive rate hikes crush inflation but cause deep recession.
         Unemployment averages 7.3%, highest of any full decade.

  1990s: The "Goldilocks" era - inflation comes down to ~3%, unemployment falls
         to ~5.8%. The internet boom drives growth without overheating.

  2000s: Tech bust + Financial Crisis keep unemployment elevated (~5.5%).
         Inflation stays moderate but the decade ends in crisis.

  2010s: Slow recovery from 2008. Unemployment gradually falls. Inflation
         stubbornly stays BELOW the 2% target - the opposite problem.

  2020s: COVID shock, then the fastest inflation in 40 years, then rapid
         Fed tightening. The most volatile decade since the 1970s.""",
}

# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data():
    raw = {}
    for name, symbol in SERIES.items():
        raw[name] = web.DataReader(symbol, "fred", START, END)[symbol]
    df = pd.DataFrame(raw).resample("MS").last()
    df["inflation"]  = df["CPI"].pct_change(12) * 100
    df["core_infl"]  = df["CORE_CPI"].pct_change(12) * 100
    df.dropna(subset=["inflation", "UNRATE"], inplace=True)
    return df


# ── Figure Builders ───────────────────────────────────────────────────────────
def shade_recessions(ax):
    for rs, re in RECESSIONS:
        rs_dt, re_dt = pd.to_datetime(rs), pd.to_datetime(re)
        ax.axvspan(rs_dt, re_dt, color=STYLE["recession"], alpha=0.5, lw=0)


def build_inflation_fig(df):
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(CONTENT_BG)
    ax.set_facecolor("#fafafa")
    shade_recessions(ax)

    ax.plot(df.index, df["inflation"],  color=STYLE["inflation"],  lw=1.8,
            label="Headline Inflation (YoY %)")
    ax.plot(df.index, df["core_infl"], color=STYLE["core_infl"], lw=1.8,
            ls="--", label="Core Inflation (excl. food & energy)")
    ax.axhline(2, color="#555", lw=1, ls=":", label="2% Fed Target")
    ax.axhline(0, color="#aaa", lw=0.6)

    ax.set_title("US Inflation Over Time (1960-2025)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_ylabel("Year-over-Year Change (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.legend(framealpha=0.95, fontsize=9)
    ax.grid(axis="y", alpha=0.3, ls="--")

    annotations = {
        "1973-11": ("Oil Crisis", 8),
        "1979-06": ("Volcker Era", 13),
        "2022-06": ("Post-COVID", 7),
    }
    for ds, (lbl, yo) in annotations.items():
        dt = pd.to_datetime(ds)
        if dt in df.index:
            ax.annotate(lbl, xy=(dt, df.loc[dt, "inflation"]),
                        xytext=(dt, df.loc[dt, "inflation"] + yo),
                        fontsize=7.5, ha="center", color="#555",
                        arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

    from matplotlib.patches import Patch
    extra = [Patch(facecolor=STYLE["recession"], label="NBER Recession")]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + extra, labels + ["NBER Recession"],
              framealpha=0.95, fontsize=9)

    fig.tight_layout()
    return fig


def build_unemployment_fig(df):
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(CONTENT_BG)
    ax.set_facecolor("#fafafa")
    shade_recessions(ax)

    ax.fill_between(df.index, df["UNRATE"], alpha=0.18,
                    color=STYLE["unemployment"])
    ax.plot(df.index, df["UNRATE"], color=STYLE["unemployment"],
            lw=1.8, label="Unemployment Rate")
    roll = df["UNRATE"].rolling(24).mean()
    ax.plot(df.index, roll, color="navy", lw=1.2, ls="--",
            label="24-month Moving Average")

    ax.set_title("US Unemployment Rate Over Time (1960-2025)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Unemployment Rate (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    from matplotlib.patches import Patch
    extra = [Patch(facecolor=STYLE["recession"], label="NBER Recession")]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + extra, labels + ["NBER Recession"],
              framealpha=0.95, fontsize=9)
    ax.grid(axis="y", alpha=0.3, ls="--")

    fig.tight_layout()
    return fig


def build_dashboard_fig(df):
    fig = plt.figure(figsize=(11, 8))
    fig.patch.set_facecolor(CONTENT_BG)
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

    panels = [
        (gs[0], "inflation",  STYLE["inflation"],    "Inflation (YoY %)"),
        (gs[1], "UNRATE",     STYLE["unemployment"], "Unemployment (%)"),
        (gs[2], "FEDFUNDS",   STYLE["fedfunds"],     "Fed Funds Rate (%)"),
    ]

    axes = []
    for i, (slot, col, color, ylabel) in enumerate(panels):
        ax = fig.add_subplot(slot)
        ax.set_facecolor("#fafafa")
        shade_recessions(ax)
        ax.plot(df.index, df[col], color=color, lw=1.6, label=ylabel)
        if col == "inflation":
            ax.axhline(2, color="#555", lw=0.8, ls=":", label="2% target")
            ax.set_title("US Macroeconomic Dashboard (1960-2025)",
                         fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(axis="y", alpha=0.25, ls="--")
        axes.append(ax)

    axes[-1].set_xlabel("Year")
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=STYLE["recession"])], ["NBER Recession"],
               loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def build_phillips_fig(df):
    eras = {
        "1960s-70s": (df["1960":"1979"], "#e41a1c"),
        "1980s-90s": (df["1980":"1999"], "#377eb8"),
        "2000s-10s": (df["2000":"2019"], "#4daf4a"),
        "2020s":     (df["2020":],       "#ff7f00"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor(CONTENT_BG)
    fig.suptitle("Phillips Curve: Inflation vs. Unemployment",
                 fontsize=13, fontweight="bold")

    # Left: full period
    ax = axes[0]
    ax.set_facecolor("#fafafa")
    sc = ax.scatter(df["UNRATE"], df["inflation"],
                    c=df.index.year, cmap="plasma", alpha=0.5, s=14, zorder=3)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Year", fontsize=8)

    sl, ic, r, p, _ = stats.linregress(df["UNRATE"], df["inflation"])
    xr = np.linspace(df["UNRATE"].min(), df["UNRATE"].max(), 100)
    ax.plot(xr, sl * xr + ic, color="black", lw=1.8,
            label=f"Trend  r = {r:.2f}")
    ax.axhline(0, color="#aaa", lw=0.6)
    ax.axhline(2, color="#aaa", lw=0.6, ls=":")
    ax.set_xlabel("Unemployment Rate (%)")
    ax.set_ylabel("Inflation Rate (YoY %)")
    ax.set_title(f"Full Period\nr = {r:.2f},  p = {p:.4f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, ls="--")

    # Right: by era
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

    ax2.axhline(0, color="#aaa", lw=0.6)
    ax2.axhline(2, color="#aaa", lw=0.6, ls=":")
    ax2.set_xlabel("Unemployment Rate (%)")
    ax2.set_ylabel("Inflation Rate (YoY %)")
    ax2.set_title("By Era  (the curve shifts over time)")
    ax2.legend(fontsize=8, framealpha=0.95)
    ax2.grid(alpha=0.25, ls="--")

    fig.tight_layout()
    return fig


def build_decade_fig(df):
    df2 = df.copy()
    df2["decade"] = (df2.index.year // 10 * 10).astype(str) + "s"
    summary = df2.groupby("decade")[["inflation", "UNRATE"]].mean().reset_index()

    x = np.arange(len(summary))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(CONTENT_BG)
    ax.set_facecolor("#fafafa")

    b1 = ax.bar(x - w/2, summary["inflation"], w,
                label="Avg Inflation (%)", color=STYLE["inflation"], alpha=0.85)
    b2 = ax.bar(x + w/2, summary["UNRATE"], w,
                label="Avg Unemployment (%)", color=STYLE["unemployment"], alpha=0.85)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(summary["decade"], fontsize=10)
    ax.set_ylabel("Average Rate (%)")
    ax.set_title("Average Inflation & Unemployment by Decade",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.set_ylim(0, max(summary["inflation"].max(), summary["UNRATE"].max()) + 2)

    fig.tight_layout()
    return fig


# ── Dashboard UI ──────────────────────────────────────────────────────────────
VIEWS = [
    ("Inflation Trends",  build_inflation_fig),
    ("Unemployment",      build_unemployment_fig),
    ("Macro Dashboard",   build_dashboard_fig),
    ("Phillips Curve",    build_phillips_fig),
    ("Decade Summary",    build_decade_fig),
]


class EconomicDashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("US Economic Data Dashboard")
        self.root.geometry("1280x820")
        self.root.configure(bg=SIDEBAR_BG)
        self.root.minsize(900, 600)

        self.df       = None
        self.figures  = {}
        self.active   = None       # current view name
        self._canvas  = None
        self._toolbar = None
        self._btn_map = {}

        self._build_ui()
        # Load data in background so UI stays responsive
        threading.Thread(target=self._load, daemon=True).start()

    # ── UI Layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Left sidebar ──────────────────────────────────────────────────────
        self.sidebar = tk.Frame(self.root, bg=SIDEBAR_BG, width=200)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo / title
        tk.Label(
            self.sidebar,
            text="US Economic\nDashboard",
            bg=SIDEBAR_BG, fg=TITLE_FG,
            font=("Helvetica", 13, "bold"),
            pady=20,
        ).pack(fill="x")

        tk.Frame(self.sidebar, bg="#2c3e56", height=1).pack(fill="x", padx=16)

        # Nav buttons (disabled until data loads)
        tk.Label(
            self.sidebar, text="VIEWS",
            bg=SIDEBAR_BG, fg="#7f8c9a",
            font=("Helvetica", 8, "bold"),
            pady=10,
        ).pack(fill="x", padx=16, anchor="w")

        for label, _ in VIEWS:
            btn = tk.Button(
                self.sidebar,
                text=label,
                bg=SIDEBAR_BG, fg=SIDEBAR_FG,
                activebackground=BTN_ACTIVE, activeforeground="white",
                relief="flat", anchor="w",
                font=("Helvetica", 10),
                padx=20, pady=10,
                cursor="hand2",
                state="disabled",
                command=lambda l=label: self.show_view(l),
            )
            btn.pack(fill="x")
            btn.bind("<Enter>", lambda e, b=btn: self._btn_hover(b, True))
            btn.bind("<Leave>", lambda e, b=btn: self._btn_hover(b, False))
            self._btn_map[label] = btn

        # Status label at bottom of sidebar
        self.status_var = tk.StringVar(value="Downloading data...")
        tk.Label(
            self.sidebar,
            textvariable=self.status_var,
            bg=SIDEBAR_BG, fg="#7f8c9a",
            font=("Helvetica", 8),
            wraplength=170,
        ).pack(side="bottom", padx=10, pady=16)

        tk.Frame(self.sidebar, bg="#2c3e56", height=1).pack(
            side="bottom", fill="x", padx=16)

        # ── Right content area ────────────────────────────────────────────────
        self.content = tk.Frame(self.root, bg=CONTENT_BG)
        self.content.pack(side="left", fill="both", expand=True)

        # Chart frame (top ~65%)
        self.chart_frame = tk.Frame(self.content, bg=CONTENT_BG)
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        # Loading label shown while data downloads
        self.loading_label = tk.Label(
            self.chart_frame,
            text="Downloading data from FRED...\nPlease wait.",
            bg=CONTENT_BG, fg="#555",
            font=("Helvetica", 14),
        )
        self.loading_label.place(relx=0.5, rely=0.45, anchor="center")

        # Divider
        tk.Frame(self.content, bg="#dde3ea", height=1).pack(
            fill="x", padx=10, pady=6)

        # Explanation panel (bottom ~35%)
        expl_container = tk.Frame(self.content, bg=CONTENT_BG)
        expl_container.pack(fill="both", padx=10, pady=(0, 10))

        tk.Label(
            expl_container,
            text="HOW TO READ THIS CHART",
            bg=CONTENT_BG, fg="#7f8c9a",
            font=("Helvetica", 8, "bold"),
            anchor="w",
        ).pack(fill="x")

        self.expl_text = tk.Text(
            expl_container,
            bg=EXPL_BG, fg="#2c3e50",
            font=("Courier", 9),
            relief="flat",
            wrap="word",
            height=10,
            padx=12, pady=8,
            state="disabled",
            cursor="arrow",
        )
        scrollbar = ttk.Scrollbar(expl_container, command=self.expl_text.yview)
        self.expl_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.expl_text.pack(fill="both", expand=True)

    def _btn_hover(self, btn: tk.Button, entering: bool):
        label = btn["text"]
        if label == self.active:
            return
        btn.configure(bg=BTN_HOVER if entering else SIDEBAR_BG)

    # ── Data Loading ──────────────────────────────────────────────────────────
    def _load(self):
        try:
            self.root.after(0, lambda: self.status_var.set(
                "Downloading 4 series\nfrom FRED..."))
            df = load_data()
            self.root.after(0, lambda: self.status_var.set(
                "Building charts..."))
            figures = {}
            for label, builder in VIEWS:
                figures[label] = builder(df)
            self.df = df
            self.figures = figures
            self.root.after(0, self._on_ready)
        except Exception as exc:
            self.root.after(0, lambda: self._on_error(str(exc)))

    def _on_ready(self):
        self.loading_label.destroy()
        self.status_var.set(f"Loaded: {len(self.df)} months\n({self.df.index[0].year}"
                            f"-{self.df.index[-1].year})")
        for btn in self._btn_map.values():
            btn.configure(state="normal")
        # Show first view automatically
        self.show_view(VIEWS[0][0])

    def _on_error(self, msg: str):
        self.loading_label.configure(
            text=f"Failed to load data.\n\n{msg}\n\nCheck your internet connection.",
            fg="#c0392b",
        )
        self.status_var.set("Error loading data.")

    # ── View Switching ────────────────────────────────────────────────────────
    def show_view(self, label: str):
        if label == self.active:
            return

        # Reset previous button style
        if self.active and self.active in self._btn_map:
            self._btn_map[self.active].configure(bg=SIDEBAR_BG, fg=SIDEBAR_FG)

        self.active = label
        self._btn_map[label].configure(bg=BTN_ACTIVE, fg="white")

        # Remove old canvas + toolbar
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._toolbar:
            self._toolbar.destroy()
            self._toolbar = None

        # Embed new figure
        fig = self.figures[label]
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self.chart_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

        self._canvas  = canvas
        self._toolbar = toolbar

        # Update explanation
        text = EXPLANATIONS.get(label, "")
        self.expl_text.configure(state="normal")
        self.expl_text.delete("1.0", "end")
        self.expl_text.insert("1.0", text)
        self.expl_text.configure(state="disabled")
        self.expl_text.yview_moveto(0)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()

    # App icon colour (Windows taskbar)
    try:
        root.iconbitmap(default="")
    except Exception:
        pass

    app = EconomicDashboard(root)
    root.mainloop()
