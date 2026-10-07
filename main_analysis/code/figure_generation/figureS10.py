"""
FIGURE S10 - distribution of the optimal objective proportion (OOP)
across the 136 accessibility-constrained study areas, grouped by sub-landscape
and by road network.

Answers Reviewer 1's requests to show the OOP distribution (Figure 5 comment)
and to distinguish landscape-driven from network-driven variation (Figure 6
comment), without recolouring Figure 6.

Also reports the proportion of variation in OOP, and in deviation from the
fitted RCP-OOP relationship, attributable to each factor.

    python figureS10.py
"""

import os
import numpy as np
import pandas as pd
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

DIR_OPT   = PROJECT_ROOT / "main_analysis" / "Outputs" / "optimisation"
DIR_DISTR = PROJECT_ROOT / "main_analysis" / "host_distributions"
DIR_ROAD  = PROJECT_ROOT / "main_analysis" / "road_patterns"
DIR_VISIT = PROJECT_ROOT / "main_analysis" / "Outputs" / "infection_visit_counts"
DIR_NIG   = PROJECT_ROOT / "main_analysis" / "code" / "nigeria_comparison" / "Outputs"
DIR_OUT   = PROJECT_ROOT / "main_analysis" / "Outputs" / "figures"
DIR_METRIC = DIR_OPT / "output_metric_sfreq52_nsite10"
OUT_PNG = os.path.join(DIR_OUT, "FigureS10.png")
OUT_CSV = os.path.join(DIR_OUT, "FigureS10_variance_decomposition.csv")
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
                 'RCP': m['road_coverage_plant'],
                 'OOP': m['objective'] / best[m['area']]})

d = pd.DataFrame(rows).sort_values(['area', 'road']).reset_index(drop=True)
print(f"{len(d)} study areas, OOP range {d.OOP.min():.3f} to {d.OOP.max():.3f} "
      f"({100*d.OOP.min():.1f}% to {100*d.OOP.max():.1f}%)")


def saturation(x, a, b):
    return a * (1 - np.exp(-b * x))


(a_fit, b_fit), _ = curve_fit(saturation, d.RCP, d.OOP, p0=[1.0, 5.0],
                              maxfev=20000)
d['residual'] = d.OOP - saturation(d.RCP, a_fit, b_fit)


# ------------------------------------------- variance decomposition
def r2_by(df, resp, factor):
    """Proportion of variance in resp explained by group means of factor."""
    y = df[resp].values
    ss_tot = ((y - y.mean()) ** 2).sum()
    pred = df.groupby(factor)[resp].transform('mean').values
    return 1 - ((y - pred) ** 2).sum() / ss_tot


def r2_both(df, resp):
    """Additive two-factor model fitted by least squares on dummy variables."""
    y = df[resp].values
    X = pd.get_dummies(df[['area', 'road']], drop_first=True).astype(float)
    X.insert(0, 'const', 1.0)
    beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
    resid = y - X.values @ beta
    return 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()


out = []
for resp in ['OOP', 'residual']:
    ra = r2_by(d, resp, 'area')
    rr = r2_by(d, resp, 'road')
    rb = r2_both(d, resp)
    out.append({'response': resp, 'sub_landscape_R2': ra,
                'road_network_R2': rr, 'both_R2': rb})
    print(f"{resp:9s}: sub-landscape {100*ra:5.1f}%   "
          f"road network {100*rr:5.1f}%   both {100*rb:5.1f}%")

pd.DataFrame(out).round(4).to_csv(OUT_CSV, index=False)

# ---------------------------------------------------------------- figure
areas = sorted(d.area.unique())
roads = sorted(d.road.unique())

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300,
                         gridspec_kw={'width_ratios': [1, 2.1]}, sharey=True)

for ax, factor, order, xlabel in [
        (axes[0], 'area', areas, 'Sub-landscape'),
        (axes[1], 'road', roads, 'Road network')]:
    data = [d.loc[d[factor] == k, 'OOP'].values for k in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', lw=1.2),
                    flierprops=dict(marker='o', ms=3, mfc='0.4',
                                    mec='none', alpha=0.7))
    for patch in bp['boxes']:
        patch.set_facecolor('0.82')
        patch.set_edgecolor('0.35')
    ax.set_xlabel(xlabel)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, 1.05)

axes[0].set_ylabel('Optimal objective proportion (OOP)')
axes[1].tick_params(axis='x', rotation=45)

for ax, letter in zip(axes, 'ab'):
    ax.text(-0.10 if ax is axes[0] else -0.05, 1.07, letter,
            transform=ax.transAxes, fontsize=16, fontweight=700,
            va='top', ha='left')

fig.tight_layout()
fig.savefig(OUT_PNG)
print(f"\nSaved {OUT_PNG}\n      {OUT_CSV}")
