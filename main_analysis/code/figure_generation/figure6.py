"""
FIGURE 6 - optimisation objective proportion against four accessibility
metrics, for the 136 study areas, with the five Nigerian states overlaid on
the two metrics computed for them.

  a. RCP          Road Coverage over Plants
  b. CPA          Cell-based Plant Accessibility
  c. road length
  d. TDB          total distance of accessible cells to the landscape boundary

Exponential saturation curves OOP = a(1 - exp(-bx)) are fitted to the 136
study areas in each panel; the fitted equation and its MSE are printed on
each panel.

Road length and TDB were not computed for the Nigerian states, so no crosses
appear on panels c and d.

    python figure6.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
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

DIR_METRIC = DIR_OPT / "output_metric_sfreq52_nsite10"
DIR_DISTR = DIR_DISTR
DIR_NIG = DIR_NIG / "output_metric"
OUT_PNG = os.path.join(DIR_OUT, "Figure6.png")
OUT_CSV = os.path.join(DIR_OUT, "Figure6_data.csv")

SUFFIX = "surveyfreq52_nsites10"
NIG_STATES = ['anambra', 'kebbi', 'nasarawa', 'ogun', 'plateau']
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


def saturation(x, a, b):
    return a * (1 - np.exp(-b * x))


# ---------------------------------------------------------------- inputs
distr = {f"{i:02d}": load(os.path.join(DIR_DISTR, f"area{i:02d}.joblib"))
         for i in range(1, 9)}

# landscape boundary cells (the grid is identical for every sub-landscape)
pos0 = np.asarray(distr['01']['xy'])
xmin, xmax = pos0[:, 0].min(), pos0[:, 0].max()
ymin, ymax = pos0[:, 1].min(), pos0[:, 1].max()
BOUNDARIES = np.vstack([
    pos0[pos0[:, 0] == xmin], pos0[pos0[:, 0] == xmax],
    pos0[pos0[:, 1] == ymin], pos0[pos0[:, 1] == ymax]])


def tdb(area, allowed):
    """Total shortest distance from accessible cells to the landscape edge."""
    p = np.asarray(distr[area]['xy'])[allowed]
    return np.min(cdist(p, BOUNDARIES), axis=1).sum()


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
    allowed = np.asarray(m['sites_allowed'])
    rows.append({
        'area': m['area'], 'road': m['road'],
        'RCP': m['road_coverage_plant'],
        'CPA': len(allowed) / int(m['num_host_pop_non0']),
        'road_length': m['road_length'],
        'TDB': tdb(m['area'], allowed),
        'OOP': m['objective'] / best[m['area']],
    })

df = pd.DataFrame(rows).sort_values(['area', 'road']).reset_index(drop=True)
print(f"{len(df)} study areas")

nrows = []
for st in NIG_STATES:
    m = load(os.path.join(DIR_NIG, f"metric_{st}_{SUFFIX}.joblib"))
    b = load(os.path.join(DIR_NIG, f"metric_NoRoad_{st}_{SUFFIX}.joblib"))
    nrows.append({'state': st.capitalize(), 'RCP': m['RCP'], 'CPA': m['CPA'],
                  'OOP': m['objective'] / b['objective']})
nig = pd.DataFrame(nrows).sort_values('RCP').reset_index(drop=True)
print(nig.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

df.round(6).to_csv(OUT_CSV, index=False)

# ---------------------------------------------------------------- figure
PANELS = [
    ('RCP', 'Road coverage over plants (RCP)', True),
    ('CPA', 'Cell-based plant accessibility (CPA)', True),
    ('road_length', 'Road length (m)', False),
    ('TDB', 'Total distance to boundary (m)', False),
]


def mathtext_coef(v, sig=3):
    """Render a coefficient for mathtext, using a power of ten when small."""
    if v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    if -3 <= exp <= 3:
        return f"{v:.{sig}g}"
    mant = v / 10 ** exp
    return rf"{mant:.{sig}g} \times 10^{{{exp}}}"


fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)

for ax, (col, label, has_nigeria), letter in zip(axes.ravel(), PANELS, 'abcd'):
    x, y = df[col].values, df.OOP.values
    (a, b), _ = curve_fit(saturation, x, y,
                          p0=[1.0, 5.0 / max(x.mean(), 1e-9)], maxfev=40000)
    mse = np.mean((y - saturation(x, a, b)) ** 2)

    ax.scatter(x, y, s=16, color='0.55', linewidths=0, alpha=0.85,
               label='Study areas')
    xx = np.linspace(0, x.max() * 1.02, 400)
    ax.plot(xx, saturation(xx, a, b), color='C0', lw=1.5,
            label='Fitted relationship')

    if has_nigeria:
        ax.scatter(nig[col], nig.OOP, s=70, color='C3', marker='X',
                   edgecolors='black', linewidths=0.8, zorder=4,
                   label='Nigerian states')

    eq = (rf"$\mathrm{{OOP}} = {mathtext_coef(a)}"
          rf"\left(1 - e^{{-{mathtext_coef(b)}\,x}}\right)$")
    ax.text(0.97, 0.06, f"{eq}\nMSE = {mse:.3f}", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10)
    print(f"  {col}: OOP = {a:.4f}(1 - exp(-{b:.4g} x)), MSE {mse:.4f}")

    ax.set_xlabel(label)
    ax.set_xlim(0, x.max() * 1.02)
    ax.xaxis.set_major_formatter(COMMA)
    ax.set_ylim(0, 1.05)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(-0.13, 1.07, letter, transform=ax.transAxes, fontsize=16,
            fontweight=700, va='top', ha='left')

for ax in axes[:, 0]:
    ax.set_ylabel('Optimal objective proportion (OOP)')

axes[0, 0].legend(frameon=False, fontsize=10, loc='upper left',
                  bbox_to_anchor=(0.02, 0.98))

fig.tight_layout()
fig.savefig(OUT_PNG)
print(f"\nSaved {OUT_PNG}\n      {OUT_CSV}")
