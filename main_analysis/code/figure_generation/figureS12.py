"""
FIGURE S12 - the study areas falling furthest below the fitted RCP-OOP trend.

Fits an exponential saturation curve to the RCP-OOP relationship at the
baseline surveillance intensity (N = 10, F = 52), identifies the study areas
with the most negative residuals, and plots them:

    row a: plant density
    row b: infection-visit probability

    python figureS12.py
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import curve_fit
from joblib import load
from matplotlib.ticker import FuncFormatter

# Scientific Reports: commas separate thousands from five digits upward
COMMA = FuncFormatter(lambda v, p: f"{v:,.0f}" if abs(v) >= 10000 else f"{v:g}")


from pathlib import Path

# ==========================================================================
# All paths are resolved relative to the repository root, so the scripts run
# from anywhere without editing.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]   # main_analysis/code/figure_generation/ -> repo root

DIR_OPT   = PROJECT_ROOT / "main_analysis" / "Outputs" / "optimisation"
DIR_DISTR = PROJECT_ROOT / "main_analysis" / "host_distributions"
DIR_ROAD  = PROJECT_ROOT / "main_analysis" / "road_patterns"
DIR_VISIT = PROJECT_ROOT / "main_analysis" / "Outputs" / "infection_visit_counts"
DIR_NIG   = PROJECT_ROOT / "main_analysis" / "code" / "nigeria_comparison" / "Outputs"
DIR_OUT   = PROJECT_ROOT / "main_analysis" / "Outputs" / "figures"

DIR_OPT   = DIR_OPT
DIR_DISTR = DIR_DISTR
DIR_VISIT = DIR_VISIT
DIR_ROAD  = DIR_ROAD
OUT_PNG = os.path.join(DIR_OUT, "FigureS12.png")
OUT_CSV = os.path.join(DIR_OUT, "FigureS12_residuals.csv")

NSITES, SFREQ = 10, 52
N_ANOMALIES = 5
N_SIMS = 2000
# ==========================================================================

DIR_OUT.mkdir(parents=True, exist_ok=True)


def set_font(preferred="CMU Serif"):
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in [preferred, "Latin Modern Roman", "DejaVu Serif"]:
        if c in available:
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.serif'] = [c]
            mpl.rcParams['mathtext.fontset'] = 'cm'
            return
    print(f"'{preferred}' not found; using matplotlib default.")


set_font()
for k in ['font.weight', 'axes.titleweight', 'axes.labelweight']:
    mpl.rcParams[k] = 500
mpl.rcParams['font.size'] = 12

# ---------------------------------------------------------------- load
dir_metric = os.path.join(DIR_OPT, f"output_metric_sfreq{SFREQ}_nsite{NSITES}")
files = [f for f in os.listdir(dir_metric) if f.endswith(".joblib")]

road_m = [load(os.path.join(dir_metric, f)) for f in files if 'NoRoad' not in f]
noroad_m = [load(os.path.join(dir_metric, f)) for f in files if 'NoRoad' in f]

best = {m['area']: m['objective'] for m in noroad_m}

df = pd.DataFrame([{
    'area': m['area'],
    'road': m['road'],
    'RCP': m['road_coverage_plant'],
    'objective': m['objective'],
    'OOP': m['objective'] / best[m['area']],
} for m in road_m]).sort_values(['area', 'road']).reset_index(drop=True)

print(f"{len(df)} study areas at N={NSITES}, F={SFREQ}")


# ------------------------------------------------- fit and residuals
def saturation(x, a, b):
    """Exponential saturation: y = a(1 - exp(-bx))."""
    return a * (1 - np.exp(-b * x))


popt, _ = curve_fit(saturation, df['RCP'], df['OOP'],
                    p0=[1.0, 5.0], maxfev=20000)
a, b = popt
df['fitted'] = saturation(df['RCP'], a, b)
df['residual'] = df['OOP'] - df['fitted']
mse = np.mean(df['residual'] ** 2)

print(f"fit: OOP = {a:.3f}(1 - exp(-{b:.3f} RCP))   MSE = {mse:.4f}")

df_sorted = df.sort_values('residual')
anomalies = df_sorted.head(N_ANOMALIES).copy()

df.round(4).to_csv(OUT_CSV, index=False)

print(f"\n{N_ANOMALIES} most negative residuals:")
print(anomalies[['area', 'road', 'RCP', 'OOP', 'fitted',
                 'residual']].to_string(index=False,
                                        float_format=lambda v: f"{v:.4f}"))

print("\nnext three, for reference:")
print(df_sorted.iloc[N_ANOMALIES:N_ANOMALIES + 3][
    ['area', 'road', 'RCP', 'OOP', 'residual']].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------- figure
distr_cache, visit_all = {}, load(
    os.path.join(DIR_VISIT, "infection_visit_count_8areas.joblib"))


def distr(area):
    if area not in distr_cache:
        distr_cache[area] = load(os.path.join(DIR_DISTR, f"area{area}.joblib"))
    return distr_cache[area]


n = len(anomalies)
fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 8.6), dpi=300)

for j, (_, row) in enumerate(anomalies.iterrows()):
    d = distr(row['area'])
    pos = np.asarray(d['xy'])
    pop = np.asarray(d['host_population'])
    non0 = pop > 0
    visit = np.asarray(visit_all[int(row['area']) - 1])

    shp = gpd.read_file(os.path.join(DIR_ROAD,
                                     f"roadnetwork{row['road']}.shp"))

    # row a: plant density
    ax = axes[0, j]
    im_pop = ax.scatter(pos[non0, 0], pos[non0, 1], marker='s', s=14,
                        linewidths=0, c=pop[non0], cmap='RdYlGn_r', alpha=0.85)
    shp.plot(ax=ax, color='black', linewidth=1.4, alpha=0.7)
    ax.set_title(f"area {row['area']}, road {row['road']}\n"
                 f"RCP = {row['RCP']:.3f},  OOP = {row['OOP']:.3f}",
                 fontsize=11)

    # row b: infection-visit probability
    ax = axes[1, j]
    im_vis = ax.scatter(pos[non0, 0], pos[non0, 1], marker='s', s=14,
                        linewidths=0, c=visit[non0] / N_SIMS,
                        cmap='RdYlGn_r', alpha=0.85)
    shp.plot(ax=ax, color='black', linewidth=1.4, alpha=0.7)

    for ax in (axes[0, j], axes[1, j]):
        ax.set_aspect('equal')
        ax.set_xlim(0, 50000)
        ax.set_ylim(0, 50000)
        ax.xaxis.set_major_formatter(COMMA)
        ax.yaxis.set_major_formatter(COMMA)
        ax.set_xlabel('Easting (m)', fontsize=10)
        if j > 0:
            ax.set_yticklabels([])

axes[0, 0].set_ylabel('Northing (m)', fontsize=10)
axes[1, 0].set_ylabel('Northing (m)', fontsize=10)

for ax, letter in [(axes[0, 0], 'a'), (axes[1, 0], 'b')]:
    ax.text(-0.30, 1.06, letter, transform=ax.transAxes,
            fontsize=18, fontweight=700, va='top', ha='left')

fig.subplots_adjust(left=0.07, right=0.90, top=0.93, bottom=0.07,
                    wspace=0.08, hspace=0.18)

cax1 = fig.add_axes([0.915, 0.55, 0.014, 0.32])
cax2 = fig.add_axes([0.915, 0.11, 0.014, 0.32])
fig.colorbar(im_pop, cax=cax1, label='Plant population')
fig.colorbar(im_vis, cax=cax2, label='Infection probability')

fig.savefig(OUT_PNG)
print(f"\nSaved {OUT_PNG}\n      {OUT_CSV}")
