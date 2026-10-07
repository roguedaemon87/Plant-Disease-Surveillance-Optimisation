"""
FIGURE S13 - controlled test cases for the anomalies below the RCP-OOP trend.

a. The four test-case landscapes: infection probability (from 2,000
   simulations), the road network and the optimal surveillance sites.
       Test case 1: area 0, road 0      Test case 2: area 0, road 1
       Test case 3: area 2, road 2      Test case 4: area 2, road 3
b. The test cases plotted against the RCP-OOP relationship fitted to the
   136 study areas of the main analysis (N = 10, F = 52).

Host positions and populations are taken from the simulation files, as the
optimisation used them. Infection probability = proportion of simulations
in which the cell was ever infected.

    python figureS13.py
"""

import os
import gc
import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import curve_fit
from joblib import load

from pathlib import Path

# ==========================================================================
# All paths are resolved relative to the repository root, so the scripts run
# from anywhere without editing.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]   # main_analysis/code/figure_generation/ -> repo root

OUTLIER  = PROJECT_ROOT / "main_analysis" / "Outputs" / "outlier_cases"
DIR_SIM  = OUTLIER / "simulations"
DIR_MET  = OUTLIER / "optimisation" / "output_metric"
DIR_ROAD = (PROJECT_ROOT / "main_analysis" / "code" / "optimisation" /
            "outlier_cases" / "inputs" / "road_patterns")
MAIN_MET = (PROJECT_ROOT / "main_analysis" / "Outputs" / "optimisation" /
            "output_metric_sfreq52_nsite10")
DIR_OUT  = PROJECT_ROOT / "main_analysis" / "Outputs" / "figures"
OUT_PNG  = os.path.join(DIR_OUT, "FigureS13.png")
# ==========================================================================

SUFFIX = "surveyfreq52_nsites10"
TEST_CASES = [(1, '0', '0'), (2, '0', '1'), (3, '2', '2'), (4, '2', '3')]

DIR_OUT.mkdir(parents=True, exist_ok=True)


def set_font(preferred="CMU Serif"):
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in [preferred, "Latin Modern Roman", "DejaVu Serif"]:
        if c in available:
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.serif'] = [c]
            mpl.rcParams['mathtext.fontset'] = 'cm'
            mpl.rcParams['axes.labelsize'] = 14
            mpl.rcParams['xtick.labelsize'] = 12
            mpl.rcParams['ytick.labelsize'] = 12
            return


set_font()
for k in ['font.weight', 'axes.titleweight', 'axes.labelweight']:
    mpl.rcParams[k] = 500
mpl.rcParams['font.size'] = 11

# ---------------------------------------------------------------- landscapes
land = {}
for area in sorted({a for _, a, _ in TEST_CASES}):
    print(f"area {area}: loading simulations ...", flush=True)
    sims = load(os.path.join(DIR_SIM, f"df_sims{area}.joblib"))
    pos = np.column_stack((sims[0]['x'], sims[0]['y']))
    pop = np.asarray(sims[0]['host_population'])
    visits = np.sum([np.asarray(s['state']) != 'S' for s in sims], axis=0)
    land[area] = {'pos': pos, 'pop': pop, 'prob': visits / len(sims)}
    print(f"  {len(sims)} simulations, {int((pop > 0).sum())} populated cells, "
          f"max infection probability {land[area]['prob'].max():.3f}")
    del sims
    gc.collect()


def metric(area, road):
    tag = f"NoRoad_area{area}" if road is None else f"road{road}_area{area}"
    return load(os.path.join(DIR_MET, f"metric_{tag}_{SUFFIX}.joblib"))


cases = []
for k, area, road in TEST_CASES:
    m = metric(area, road)
    base = metric(area, None)
    cases.append({
        'k': k, 'area': area, 'road': road,
        'RCP': m['road_coverage_plant'],
        'OOP': m['objective'] / base['objective'],
        'objective': m['objective'],
        'config': np.asarray(m['config_opt']),
        'shp': gpd.read_file(os.path.join(DIR_ROAD, f"roadnetwork{road}.shp")),
    })

# ---------------------------------------------------------------- main trend
files = [f for f in os.listdir(MAIN_MET) if f.endswith(".joblib")]
best = {}
pts = []
for f in files:
    m = load(os.path.join(MAIN_MET, f))
    if 'NoRoad' in f:
        best[m['area']] = m['objective']
for f in files:
    if 'NoRoad' in f:
        continue
    m = load(os.path.join(MAIN_MET, f))
    pts.append((m['road_coverage_plant'], m['objective'] / best[m['area']]))
pts = np.array(pts)


def saturation(x, a, b):
    return a * (1 - np.exp(-b * x))


(a_fit, b_fit), _ = curve_fit(saturation, pts[:, 0], pts[:, 1],
                              p0=[1.0, 5.0], maxfev=20000)
print(f"\nmain trend ({len(pts)} study areas): "
      f"OOP = {a_fit:.3f}(1 - exp(-{b_fit:.3f} RCP))")

print(f"\n{'case':>4} {'area':>4} {'road':>4} {'RCP':>7} {'OOP':>7} "
      f"{'fitted':>7} {'resid':>7}")
for c in cases:
    fit = saturation(c['RCP'], a_fit, b_fit)
    c['resid'] = c['OOP'] - fit
    print(f"{c['k']:>4} {c['area']:>4} {c['road']:>4} {c['RCP']:>7.3f} "
          f"{c['OOP']:>7.3f} {fit:>7.3f} {c['resid']:>+7.3f}")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.05],
                      height_ratios=[1, 1, 1.05], hspace=0.38, wspace=0.12)

vmax = max(l['prob'].max() for l in land.values())
map_axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
im = None
for ax, c in zip(map_axes, cases):
    L = land[c['area']]
    non0 = L['pop'] > 0
    im = ax.scatter(L['pos'][non0, 0], L['pos'][non0, 1], marker='s', s=7,
                    linewidths=0, c=L['prob'][non0], cmap='RdYlGn_r',
                    vmin=0, vmax=vmax, alpha=0.85)
    c['shp'].plot(ax=ax, color='black', linewidth=1.6, alpha=0.7)
    ax.scatter(L['pos'][c['config'], 0], L['pos'][c['config'], 1],
               marker='x', s=22, c='black', linewidths=1.1)
    ax.set_xlim(0, 50000)
    ax.set_ylim(0, 50000)
    ax.set_aspect('equal')
    ax.set_title(f"Test case {c['k']}\nRCP = {c['RCP']:.2f},  "
                 f"OOP = {c['OOP']:.2f}", fontsize=11)
    ax.tick_params(labelsize=9)

for i, ax in enumerate(map_axes):
    if i % 2 == 0:
        ax.set_ylabel('Northing (m)')
    else:
        ax.set_yticklabels([])
    if i >= 2:
        ax.set_xlabel('Easting (m)')
    else:
        ax.set_xticklabels([])

cax = fig.add_subplot(gs[0:2, 2])
fig.colorbar(im, cax=cax, label='Infection probability\n'
             '(based on 2000 stochastic spread simulations)')

# ---- panel b
axb = fig.add_subplot(gs[2, 0:2])
axb.scatter(pts[:, 0], pts[:, 1], s=14, color='0.65', alpha=0.8,
            linewidths=0, label='Study areas (main analysis)')
xx = np.linspace(0, max(pts[:, 0].max(), 1.0), 300)
axb.plot(xx, saturation(xx, a_fit, b_fit), color='C0', lw=1.5,
         label='Fitted relationship')
for c in cases:
    axb.scatter(c['RCP'], c['OOP'], s=70, color='C3', edgecolors='black',
                linewidths=0.8, zorder=4)
    axb.annotate(str(c['k']), (c['RCP'], c['OOP']), xytext=(7, 4),
                 textcoords='offset points', fontsize=12, fontweight=700)
axb.scatter([], [], s=70, color='C3', edgecolors='black', linewidths=0.8,
            label='Test cases')
axb.set_xlabel('Road Coverage of Plants (RCP)')
axb.set_ylabel('Optimal Objective Proportion (OOP)')
axb.set_xlim(0, xx.max())
axb.set_ylim(0, 1.05)
axb.spines[['top', 'right']].set_visible(False)
axb.legend(frameon=False, fontsize=10, loc='lower right')

for ax, letter in [(map_axes[0], 'a.'), (axb, 'b.')]:
    ax.text(-0.22 if ax is map_axes[0] else -0.16, 1.12 if ax is map_axes[0]
            else 1.06, letter, transform=ax.transAxes, fontsize=16,
            fontweight=700, va='top', ha='left')

fig.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)
fig.savefig(OUT_PNG)
print(f"\nSaved {OUT_PNG}")
