"""
FIGURE S8 - sensitivity of the results to within-cell variation in disease
progression (R1 comment 7).

a. Objective of each optimal configuration with variation vs without.
b. Optimal objective proportion (OOP) with variation vs without.

Reads noise_test_scores.csv from score_noise_test.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pathlib import Path

# ==========================================================================
# All paths are resolved relative to the repository root.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

OUTPUTS   = PROJECT_ROOT / "main_analysis" / "Outputs"
DIR_SIMS  = OUTPUTS / "simulations"
DIR_OPT   = OUTPUTS / "optimisation"
DIR_DISTR = PROJECT_ROOT / "main_analysis" / "host_distributions"
DIR_OUT   = OUTPUTS / "figures"
CSV = OUTPUTS / "noise_test" / "noise_test_scores.csv"
OUT = DIR_OUT / "FigureS8.png"
# ==========================================================================
# ==========================================================================

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif']
rcParams['font.weight'] = 500
rcParams['axes.titleweight'] = 700
rcParams['mathtext.fontset'] = 'cm'

df = pd.read_csv(CSV)
con = df[df.road != 'NoRoad']

levels = [(0.25, 'Lower variation', '#1f77b4', 'o'),
          (0.5, 'Higher variation', '#d62728', '^')]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

panels = [
    (axes[0], df, 'obj', 'Objective, no variation', 'Objective, with variation',
     'a.', 'All 144 scenarios'),
    (axes[1], con, 'oop', 'OOP, no variation', 'OOP, with variation',
     'b.', '136 study areas'),
]

for ax, data, prefix, xlab, ylab, letter, note in panels:
    x = data[f'{prefix}_noise0.0']
    lo = min(x.min(), *(data[f'{prefix}_noise{n}'].min() for n, *_ in levels))
    hi = max(x.max(), *(data[f'{prefix}_noise{n}'].max() for n, *_ in levels))
    pad = 0.03 * (hi - lo)
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, color='0.4', lw=1, ls='--', zorder=1, label='1:1 line')
    for n, lab, col, mk in levels:
        ax.scatter(x, data[f'{prefix}_noise{n}'], s=18, marker=mk,
                   facecolor='none', edgecolor=col, linewidth=0.9,
                   label=lab, zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.set_xlabel(xlab, fontweight=500)
    ax.set_ylabel(ylab, fontweight=500)
    ax.set_title(letter, loc='left', fontweight=700)
    ax.text(0.97, 0.04, note, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='0.3')
    ax.spines[['top', 'right']].set_visible(False)

axes[0].legend(frameon=False, loc='upper left', fontsize=9)

fig.tight_layout()
DIR_OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300)
print(f"Saved {OUT}")
