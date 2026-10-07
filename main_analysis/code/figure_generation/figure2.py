"""
FIGURE 2 - host distributions, road networks and infection probabilities for
two example sub-landscapes.

Rows a and b are sub-landscapes 01 and 02. Within each row, column 1 shows
the cassava host distribution, columns 2 and 3 show the same distribution
overlaid with road networks 01_1 and 01_4, and column 4 shows the
probability that each cell becomes infected before the epidemic reaches 5%
landscape-level prevalence.

Infection probabilities depend only on the host distribution, so they are
identical for both road networks within a row and are drawn once.

    python figure2.py
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
from joblib import load

from pathlib import Path

# ==========================================================================
# All paths are resolved relative to the repository root.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

OUTPUTS   = PROJECT_ROOT / "main_analysis" / "Outputs"
DIR_DISTR = PROJECT_ROOT / "main_analysis" / "host_distributions"
DIR_ROAD  = PROJECT_ROOT / "main_analysis" / "road_patterns"
DIR_VISIT = OUTPUTS / "infection_visit_counts"
DIR_OUT   = OUTPUTS / "figures"

OUT_PNG = os.path.join(DIR_OUT, "Figure2.png")

ROWS = [("01", "a"), ("02", "b")]        # sub-landscapes, in row order
ROADS = ["01_1", "01_4"]                 # columns 2 and 3
POP_SCALE = 1e6                          # colourbar is in millions of plants
N_SIMS = 2000
EXTENT = 50000
# ==========================================================================

DIR_OUT.mkdir(parents=True, exist_ok=True)

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
mpl.rcParams['font.size'] = 20

# ---------------------------------------------------------------- data
distr = {a: load(os.path.join(DIR_DISTR, f"area{a}.joblib")) for a, _ in ROWS}
visits = load(os.path.join(DIR_VISIT, "infection_visit_count_8areas.joblib"))
roads = {r: gpd.read_file(os.path.join(DIR_ROAD, f"roadnetwork{r}.shp"))
         for r in ROADS}

# the stored values are counts of simulations in which the cell was infected
visit_prob = {a: np.asarray(visits[int(a) - 1], dtype=float) / N_SIMS
              for a, _ in ROWS}

vmax_pop = max(np.asarray(distr[a]['host_population']).max() for a, _ in ROWS)
vmax_prob = max(visit_prob[a].max() for a, _ in ROWS)
print(f"max plants per cell {vmax_pop:,.0f}, "
      f"max infection probability {vmax_prob:.3f}")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 4, figsize=(26, 14), dpi=220)

im_pop = im_prob = None
for r, (area, letter) in enumerate(ROWS):
    pos = np.asarray(distr[area]['xy'])
    pop = np.asarray(distr[area]['host_population'])
    non0 = pop > 0

    for c in range(4):
        ax = axes[r, c]
        if c < 3:
            im_pop = ax.scatter(pos[non0, 0], pos[non0, 1], marker='s', s=26,
                                linewidths=0, c=pop[non0] / POP_SCALE,
                                cmap='RdYlGn_r', vmin=0,
                                vmax=vmax_pop / POP_SCALE)
            if c > 0:
                roads[ROADS[c - 1]].plot(ax=ax, color='black', linewidth=3.0)
                # geopandas sets axis labels from the CRS; clear them
                ax.set_xlabel('')
                ax.set_ylabel('')
        else:
            p = visit_prob[area]
            im_prob = ax.scatter(pos[non0, 0], pos[non0, 1], marker='s', s=26,
                                 linewidths=0, c=p[non0], cmap='viridis',
                                 vmin=0, vmax=vmax_prob)

        ax.set_aspect('equal')
        ax.set_xlim(0, EXTENT)
        ax.set_ylim(0, EXTENT)
        ax.set_xticks(range(0, EXTENT + 1, 10000))
        ax.set_yticks(range(0, EXTENT + 1, 10000))
        ax.xaxis.set_major_formatter(COMMA)
        ax.yaxis.set_major_formatter(COMMA)

        if c == 0:
            ax.set_ylabel('Northing (m)')
        else:
            ax.set_yticklabels([])
        if r == len(ROWS) - 1:
            ax.set_xlabel('Easting (m)')
        else:
            ax.set_xticklabels([])

    axes[r, 0].text(-0.30, 1.06, letter, transform=axes[r, 0].transAxes,
                    fontsize=34, fontweight=700, va='top', ha='left')

fig.subplots_adjust(left=0.065, right=0.985, top=0.965, bottom=0.215,
                    wspace=0.06, hspace=0.10)

# rule between the two rows, as in the published figure
y_rule = (axes[0, 0].get_position().y0 + axes[1, 0].get_position().y1) / 2
fig.add_artist(plt.Line2D([0.03, 0.99], [y_rule, y_rule],
                          color='0.35', linewidth=2.5))

# ---------------------------------------------------------------- colourbars
cax_pop = fig.add_axes([0.175, 0.115, 0.26, 0.016])
cb = fig.colorbar(im_pop, cax=cax_pop, orientation='horizontal')
cb.set_label('Cassava plants per km$^2$ ($\\times 10^{6}$)', labelpad=10)

cax_prob = fig.add_axes([0.655, 0.115, 0.26, 0.016])
cb2 = fig.colorbar(im_prob, cax=cax_prob, orientation='horizontal')
cb2.set_label('Infection probability\n'
              '(based on 2,000 stochastic spread simulations)', labelpad=10)

fig.savefig(OUT_PNG)
print(f"\nSaved {OUT_PNG}")
