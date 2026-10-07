"""
FIGURE S14 - simulated annealing traces for the five Nigerian states.

Two traces per state: the optimisation constrained to within 1 km of the
observed road network, and the unconstrained optimisation. Baseline
surveillance intensity (N = 10 sites, F = 52 weeks).

    python figureS14.py
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
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
NIG = DIR_NIG

DIR_TRACE = os.path.join(NIG, "output_trace")
DIR_METRIC = os.path.join(NIG, "output_metric")
OUT_PNG = os.path.join(DIR_OUT, "FigureS14.png")

SUFFIX = "surveyfreq52_nsites10"
STATES = ['anambra', 'plateau', 'nasarawa', 'kebbi', 'ogun']   # by RCP
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

fig, axes = plt.subplots(5, 2, figsize=(8.27, 11.69), dpi=300,
                         sharex=True, sharey=True)

letters = 'abcdefghij'
k = 0
for i, st in enumerate(STATES):
    m = load(os.path.join(DIR_METRIC, f"metric_{st}_{SUFFIX}.joblib"))
    b = load(os.path.join(DIR_METRIC, f"metric_NoRoad_{st}_{SUFFIX}.joblib"))
    oop = m['objective'] / b['objective']
    print(f"{st:10s} constrained {m['objective']:.4f}  "
          f"unconstrained {b['objective']:.4f}  OOP {oop:.4f}")

    for j, (tag, title) in enumerate([
            (st, 'Accessibility constrained'),
            (f"NoRoad_{st}", 'Unconstrained')]):
        tr = np.asarray(load(os.path.join(
            DIR_TRACE, f"trace_objval_{tag}_{SUFFIX}.joblib")))
        ax = axes[i, j]
        ax.plot(np.arange(len(tr)), tr, lw=1.0, color='C0')
        ax.set_ylim(0, 1.0)
        ax.xaxis.set_major_formatter(COMMA)
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(0.97, 0.08, f"final = {tr[-1]:.4f}", transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9)
        ax.text(-0.17 if j == 0 else -0.09, 1.10, letters[k],
                transform=ax.transAxes, fontsize=13, fontweight=700,
                va='top', ha='left')
        if i == 0:
            ax.set_title(title, fontsize=12)
        if j == 0:
            ax.set_ylabel(f"{st.capitalize()}\nObjective value", fontsize=10)
        if i == len(STATES) - 1:
            ax.set_xlabel('Optimisation trial number')
        k += 1

fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.06,
                    hspace=0.30, wspace=0.15)
fig.savefig(OUT_PNG)
print(f"\nSaved {OUT_PNG}")
