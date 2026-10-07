"""
FIGURE S3 - cassava host abundance across the eight sub-landscapes.

Eight panels on a shared colour scale so that the range of host-density
patterns is comparable between sub-landscapes. The scale is capped at the
99th percentile of non-zero cell populations across all eight, so that a
few very dense cells do not flatten the sparser landscapes, and the
colourbar is marked as extending beyond that value. The colour sequence
runs dark green through yellow and orange to red, matching the host and
infection maps in Figures 4 and S7.

    python figureS3.py
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
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
DIR_DISTR = DIR_DISTR
OUT_PNG = os.path.join(DIR_OUT, "FigureS3.png")

CAP_PERCENTILE = 99      # colour scale cap, percentile of non-zero cells

# dark green - light green - yellow - orange - red, as in Figures 4 and S7
CMAP = 'RdYlGn_r'
# ==========================================================================

DIR_OUT.mkdir(parents=True, exist_ok=True)

# Scientific Reports: commas separate thousands from five digits upward
COMMA = FuncFormatter(lambda v, p: f"{v:,.0f}" if abs(v) >= 10000 else f"{v:g}")


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

# ---------------------------------------------------------------- data
AREAS = [f"{i:02d}" for i in range(1, 9)]
distr = {a: load(os.path.join(DIR_DISTR, f"area{a}.joblib")) for a in AREAS}

allpop = np.concatenate([np.asarray(distr[a]['host_population']) for a in AREAS])
vmax = np.percentile(allpop[allpop > 0], CAP_PERCENTILE)
print(f"non-zero cells: {int((allpop > 0).sum())}, "
      f"max {allpop.max():,.0f}, {CAP_PERCENTILE}th percentile {vmax:,.0f}")
for a in AREAS:
    pop = np.asarray(distr[a]['host_population'])
    print(f"  sub-landscape {a}: {int((pop > 0).sum()):4d} populated cells, "
          f"max {pop.max():,.0f}")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 4, figsize=(13.0, 7.0), dpi=300)

im = None
for ax, a in zip(axes.ravel(), AREAS):
    pos = np.asarray(distr[a]['xy'])
    pop = np.asarray(distr[a]['host_population'])
    non0 = pop > 0

    im = ax.scatter(pos[non0, 0], pos[non0, 1], marker='s', s=5,
                    linewidths=0, c=pop[non0], cmap=CMAP,
                    vmin=0, vmax=vmax)

    ax.set_title(f"Sub-landscape {a}", fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(0, 50000)
    ax.set_ylim(0, 50000)
    ax.set_xticks([0, 25000, 50000])
    ax.set_yticks([0, 25000, 50000])
    ax.xaxis.set_major_formatter(COMMA)
    ax.yaxis.set_major_formatter(COMMA)

for ax in axes[1, :]:
    ax.set_xlabel('Easting (m)')
for ax in axes[:, 0]:
    ax.set_ylabel('Northing (m)')
for ax in axes[0, :]:
    ax.set_xticklabels([])
for ax in axes[:, 1:].ravel():
    ax.set_yticklabels([])

fig.subplots_adjust(left=0.07, right=0.90, top=0.94, bottom=0.09,
                    wspace=0.12, hspace=0.22)

cax = fig.add_axes([0.915, 0.12, 0.014, 0.76])
cb = fig.colorbar(im, cax=cax, extend='max')
cb.set_label('Cassava plants per km$^2$')
cb.ax.yaxis.set_major_formatter(COMMA)

fig.savefig(OUT_PNG)
print(f"\nSaved {OUT_PNG}")
