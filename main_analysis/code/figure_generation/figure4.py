"""
FIGURE 4 - reproduced from the corrected optimisation outputs.

Six panels: three maps of sub-landscape 01 showing infection probability with
the optimal surveillance sites marked (no constraint, low constraint via
roadnetwork01_4, severe constraint via roadnetwork01_1), and the corresponding
simulated annealing traces beneath each.

    python figure4.py
"""

import os
import numpy as np
import geopandas as gpd
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

DIR_CONFIG = DIR_OPT / "output_config"
DIR_TRACE  = DIR_OPT / "output_trace"
DIR_METRIC = DIR_OPT / "output_metric_sfreq52_nsite10"
DIR_DISTR  = DIR_DISTR
DIR_VISIT  = DIR_VISIT
DIR_ROAD   = DIR_ROAD

OUT_PNG = os.path.join(DIR_OUT, "Figure4.png")

N_SIMS = 2000
SUFFIX = "area01_surveyfreq52_nsites10"
# ==========================================================================

DIR_OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- font
def set_font(preferred="CMU Serif"):
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in [preferred, "CMU Serif Roman", "Computer Modern Roman",
                      "Latin Modern Roman", "DejaVu Serif"]:
        if candidate in available:
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.serif'] = [candidate]
            mpl.rcParams['mathtext.fontset'] = 'cm'
            if candidate != preferred:
                print(f"'{preferred}' not found; using '{candidate}'")
            return candidate
    print(f"'{preferred}' not found and no serif fallback matched; "
          f"using matplotlib default.")
    serif_like = sorted(n for n in available
                        if 'cmu' in n.lower() or 'modern' in n.lower())
    if serif_like:
        print("  serif-like fonts detected:", serif_like)
    return None


set_font("CMU Serif")
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------- inputs
host_distr = load(os.path.join(DIR_DISTR, "area01.joblib"))
host_positions = np.asarray(host_distr['xy'])
host_population = np.asarray(host_distr['host_population'])
non0 = host_population > 0
host_positions_non0 = host_positions[non0]

visit01 = np.asarray(load(os.path.join(
    DIR_VISIT, "infection_visit_count_8areas.joblib"))[0])

shp_severe = gpd.read_file(os.path.join(DIR_ROAD, "roadnetwork01_1.shp"))
shp_low = gpd.read_file(os.path.join(DIR_ROAD, "roadnetwork01_4.shp"))


def best_config(tag):
    c = load(os.path.join(DIR_CONFIG, f"config_{tag}_{SUFFIX}.joblib"))
    return np.asarray(c[-1])


def trace(tag):
    return np.asarray(load(os.path.join(
        DIR_TRACE, f"trace_objval_{tag}_{SUFFIX}.joblib")))


def objective(tag):
    m = load(os.path.join(DIR_METRIC, f"metric_{tag}_{SUFFIX}.joblib"))
    return m['objective']


cfg_nc, cfg_low, cfg_sev = (best_config("NoRoad"),
                            best_config("road01_4"),
                            best_config("road01_1"))
tr_nc, tr_low, tr_sev = (trace("NoRoad"),
                         trace("road01_4"),
                         trace("road01_1"))
obj_nc, obj_low, obj_sev = (objective("NoRoad"),
                            objective("road01_4"),
                            objective("road01_1"))

print(f"no constraint : obj={obj_nc:.4f}  OOP=1.00   iters={len(tr_nc)}")
print(f"low           : obj={obj_low:.4f}  OOP={obj_low/obj_nc:.2f}   "
      f"iters={len(tr_low)}")
print(f"severe        : obj={obj_sev:.4f}  OOP={obj_sev/obj_nc:.2f}   "
      f"iters={len(tr_sev)}")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(18, 11), dpi=300)
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.045],
                      wspace=0.25, hspace=0.22)

ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))
ax3, ax4, ax5 = (fig.add_subplot(gs[1, i]) for i in range(3))
cax = fig.add_subplot(gs[0, 3])

prob = visit01[non0] / N_SIMS
vmin, vmax = prob.min(), prob.max()

panels = [
    (ax0, None, cfg_nc, "a", "None", 1.00),
    (ax1, shp_low, cfg_low, "b", "Low", obj_low / obj_nc),
    (ax2, shp_severe, cfg_sev, "c", "Severe", obj_sev / obj_nc),
]

im0 = None
for ax, shp, cfg, letter, level, oop in panels:
    im = ax.scatter(host_positions_non0[:, 0], host_positions_non0[:, 1],
                    marker='s', s=30, linewidths=0, edgecolors='none',
                    alpha=0.7, c=prob, cmap='RdYlGn_r',
                    vmin=vmin, vmax=vmax)
    if im0 is None:
        im0 = im
    if shp is not None:
        shp.plot(ax=ax, color='black', linewidth=2, alpha=0.5)
    ax.scatter(host_positions[cfg, 0], host_positions[cfg, 1],
               marker='x', s=30, c='black')
    ax.set_title(f"Accessibility constraint: {level.lower()}\n"
                 f"Optimal objective proportion: {oop:.2f}",
                 fontsize=13)
    ax.set_xlabel('Easting (m)')
    ax.xaxis.set_major_formatter(COMMA)
    ax.yaxis.set_major_formatter(COMMA)
    ax.set_aspect('equal')
    ax.text(-0.10, 1.10, letter, transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top', ha='left')

ax0.set_ylabel('Northing (m)')

for ax, tr, label in [
        (ax3, tr_nc, 'Objective value trace (no area constraint)'),
        (ax4, tr_low, 'Objective value trace (low area constraint)'),
        (ax5, tr_sev, 'Objective value trace (severe area constraint)')]:
    ax.plot(np.arange(len(tr)), tr, lw=1.0)
    ax.xaxis.set_major_formatter(COMMA)
    ax.set_xlabel('Optimisation trial number')
    ax.set_ylim(0, 0.8)
    ax.legend([label], frameon=False, fontsize=10, loc='lower right')

ax3.set_ylabel('Objective value')

fig.colorbar(im0, cax=cax,
             label=f'Infection probability\n(based on {N_SIMS} stochastic '
                   f'spread simulations)')

fig.savefig(OUT_PNG, bbox_inches='tight')
print(f"\nSaved {OUT_PNG}")
