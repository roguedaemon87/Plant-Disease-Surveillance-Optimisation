"""
FIGURE S11 - the genuine landscape and road network pairings.

Answers Reviewer 3's comment 1 by separating the three cases he distinguishes:

  a. the 17 study areas in which a road network is paired with the
     sub-landscape it was taken from, shown as four progressions of
     increasing road coverage, against the 136 study areas as a whole.
  b. deviation from the fitted RCP-OOP relationship for the genuine
     pairings against the transplanted ones.

    python figureS11.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu
from joblib import load

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
DIR_METRIC = DIR_OPT / "output_metric_sfreq52_nsite10"
OUT_PNG = os.path.join(DIR_OUT, "FigureS11.png")
OUT_CSV = os.path.join(DIR_OUT, "FigureS11_self_paired_data.csv")

COLOURS = {'01': '#1b6ca8', '02': '#d95f02', '03': '#1b9e77', '05': '#7570b3'}
# ==========================================================================

DIR_OUT.mkdir(parents=True, exist_ok=True)


def set_font(preferred="CMU Serif"):
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in [preferred, "Latin Modern Roman", "DejaVu Serif"]:
        if c in available:
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.serif'] = [c]
            mpl.rcParams['mathtext.fontset'] = 'cm'
            if c != preferred:
                print(f"'{preferred}' not found; using '{c}'")
            return
    print(f"'{preferred}' not found; using matplotlib default.")


set_font()
for k in ['font.weight', 'axes.titleweight', 'axes.labelweight']:
    mpl.rcParams[k] = 500
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11

# ---------------------------------------------------------------- data
files = [f for f in os.listdir(DIR_METRIC) if f.endswith('.joblib')]
best = {}
for f in files:
    if 'NoRoad' in f:
        m = load(os.path.join(DIR_METRIC, f))
        best[m['area']] = m['objective']

rows = []
for f in files:
    if 'NoRoad' in f:
        continue
    m = load(os.path.join(DIR_METRIC, f))
    rows.append({'area': m['area'], 'road': m['road'],
                 'series': m['road'].split('_')[0],
                 'RCP': m['road_coverage_plant'],
                 'OOP': m['objective'] / best[m['area']]})

d = pd.DataFrame(rows)
d['self_paired'] = d.series == d.area


def sat(x, a, b):
    return a * (1 - np.exp(-b * x))


(a_fit, b_fit), _ = curve_fit(sat, d.RCP, d.OOP, p0=[1.0, 5.0], maxfev=40000)
d['residual'] = d.OOP - sat(d.RCP, a_fit, b_fit)
d.round(6).to_csv(OUT_CSV, index=False)

sp = d[d.self_paired].sort_values(['series', 'RCP'])
tp = d[~d.self_paired]
u, p = mannwhitneyu(sp.residual, tp.residual)
print(f"{len(d)} study areas, {len(sp)} genuine pairings")
print(f"genuine OOP range {sp.OOP.min():.3f} to {sp.OOP.max():.3f}")
print(f"residual: genuine {sp.residual.mean():+.4f}, "
      f"transplanted {tp.residual.mean():+.4f}, p = {p:.4g}")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=300,
                         gridspec_kw={'width_ratios': [1.75, 1]})

# ----- panel a
ax = axes[0]
ax.scatter(tp.RCP, tp.OOP, s=16, color='0.72', linewidths=0, alpha=0.9,
           label='Transplanted pairings (119)', zorder=1)

xx = np.linspace(0, d.RCP.max() * 1.02, 400)
ax.plot(xx, sat(xx, a_fit, b_fit), color='0.25', lw=1.4, ls='--',
        label='Fitted relationship (all 136)', zorder=2)

for s, g in sp.groupby('series'):
    ax.plot(g.RCP, g.OOP, '-o', color=COLOURS[s], ms=6, lw=1.6,
            markeredgecolor='white', markeredgewidth=0.8, zorder=3,
            label=f'Sub-landscape {s} with its own networks')

ax.set_xlabel('Road coverage over plants (RCP)')
ax.set_ylabel('Optimal objective proportion (OOP)')
ax.set_xlim(0, d.RCP.max() * 1.02)
ax.set_ylim(0, 1.05)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=9.5, loc='lower right')

# ----- panel b
ax = axes[1]
rng = np.random.default_rng(0)
groups = [('Genuine\npairings', sp.residual.values),
          ('Transplanted\npairings', tp.residual.values)]

bp = ax.boxplot([g[1] for g in groups], tick_labels=[g[0] for g in groups],
                patch_artist=True, widths=0.5, showfliers=False,
                medianprops=dict(color='black', lw=1.2))
for patch in bp['boxes']:
    patch.set_facecolor('0.88')
    patch.set_edgecolor('0.35')
for i, (_, v) in enumerate(groups, start=1):
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=11,
               color='0.25', alpha=0.6, linewidths=0, zorder=3)

ax.axhline(0, color='0.25', lw=1.0, ls='--')
ax.set_ylabel('Deviation from the fitted relationship')
ax.spines[['top', 'right']].set_visible(False)

for ax, letter, dx in zip(axes, 'ab', (-0.075, -0.17)):
    ax.text(dx, 1.06, letter, transform=ax.transAxes, fontsize=16,
            fontweight=700, va='top', ha='left')

fig.tight_layout()
fig.savefig(OUT_PNG)
print(f"\nSaved {OUT_PNG}\n      {OUT_CSV}")
