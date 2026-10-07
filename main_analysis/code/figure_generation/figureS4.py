"""
FIGURE S4 - the 17 road networks, grouped by series.

Each row holds one series, in which every network contains the sections of
the one before it, so that reading left to right along a row shows how
coverage was built up. This arrangement answers Reviewer 3's request that
sub-networks derived from the same network appear on a single row.

    01_0 ... 01_4   (5)
    02_0 ... 02_2   (3)
    03_0 ... 03_5   (6)
    05_0 ... 05_2   (3)

All panels share the same 50 x 50 km frame, so the extent of each network
is directly comparable.

    python figureS4.py
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

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
OUT_PNG = os.path.join(DIR_OUT, "FigureS4.png")

SERIES = [
    ('01', ['01_0', '01_1', '01_2', '01_3', '01_4']),
    ('02', ['02_0', '02_1', '02_2']),
    ('03', ['03_0', '03_1', '03_2', '03_3', '03_4', '03_5']),
    ('05', ['05_0', '05_1', '05_2']),
]
EXTENT = 50000           # sub-landscape side, metres
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

# ---------------------------------------------------------------- figure
ncol = max(len(m) for _, m in SERIES)
nrow = len(SERIES)

# lowest occupied row in each column, so x labels appear only at the foot
LAST_ROW = {c: max(r for r, (_, m) in enumerate(SERIES) if c < len(m))
            for c in range(ncol)}

fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 2.35 * nrow),
                         dpi=300)

for r, (label, members) in enumerate(SERIES):
    for c in range(ncol):
        ax = axes[r, c]
        if c >= len(members):
            ax.axis('off')
            continue

        net = members[c]
        shp = gpd.read_file(os.path.join(DIR_ROAD, f"roadnetwork{net}.shp"))
        shp.plot(ax=ax, color='black', linewidth=1.0)
        print(f"{net}: {shp.geometry.length.sum()/1000:7.1f} km")

        ax.set_title(net, fontsize=11)
        ax.set_aspect('equal')
        ax.set_xlim(0, EXTENT)
        ax.set_ylim(0, EXTENT)
        ax.set_xticks([0, 25000, 50000])
        ax.set_yticks([0, 25000, 50000])
        ax.xaxis.set_major_formatter(COMMA)
        ax.yaxis.set_major_formatter(COMMA)
        ax.tick_params(labelsize=9)

        # tick labels only on the outer edge of the occupied grid
        if c > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Northing (m)', fontsize=10)
        if r == LAST_ROW[c]:
            ax.set_xlabel('Easting (m)', fontsize=10)
        else:
            ax.set_xticklabels([])

fig.subplots_adjust(left=0.075, right=0.985, top=0.955, bottom=0.065,
                    wspace=0.14, hspace=0.30)
fig.savefig(OUT_PNG)
print(f"\nSaved {OUT_PNG}")
