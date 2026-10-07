"""
Convergence of the objective estimate with simulation count, for all eight
sub-landscapes, using the corrected objective (detection applies to
symptomatic plants only).

For each sub-landscape, the per-simulation detection probability is computed
for two site configurations - ten randomly selected populated cells, and the
ten most densely populated cells - and the running mean and Monte Carlo
standard error are tracked against the number of simulations M.

No new simulation or optimisation is run: this is a single pass over the
stored output in df_sims<AREA>.joblib.

    python convergence_all_areas.py

Outputs (to OUT_DIR):
    FigureS5.png   combined figure, 2 x 4 sub-landscapes
    FigureS5_convergence.csv   per-area statistics
"""

import os
import gc
import numpy as np
import pandas as pd
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
DIR_SIMS  = OUTPUTS / "simulations"
DIR_OPT   = OUTPUTS / "optimisation"
DIR_DISTR = PROJECT_ROOT / "main_analysis" / "host_distributions"
DIR_OUT   = OUTPUTS / "figures"
# ==========================================================================

SIM_DIR = DIR_SIMS
OUT_DIR = DIR_OUT

OUT_PNG = os.path.join(OUT_DIR, "FigureS5.png")
OUT_CSV = os.path.join(OUT_DIR, "FigureS5_convergence.csv")

AREAS = [f"{i:02d}" for i in range(1, 9)]
CHECK_M = [250, 500, 1000, 1500, 2000]
M_MIN_PLOT = 10
# ==========================================================================

# Baseline settings (N = 10 sites, annual surveys) and CONSTANT.py values
SURVEY_FREQ = 52
NSITES = 10
NTREES_SURVEY = 30
P_DETECT = 0.75
SIGMA0 = 0.01
LOGISTIC_RATE = 0.1693

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- font
def set_font(preferred="CMU Serif"):
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in [preferred, "Latin Modern Roman", "DejaVu Serif"]:
        if c in available:
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.serif'] = [c]
            mpl.rcParams['mathtext.fontset'] = 'cm'
            return
    print(f"'{preferred}' not found; using matplotlib default.")


set_font()
for k in ['font.weight', 'axes.titleweight', 'axes.labelweight']:
    mpl.rcParams[k] = 500
mpl.rcParams['font.size'] = 10


# ---------------------------------------------------------------- objective
def logistic_forOpt(time, time_eventS2C, time_eventC2I, host_population,
                    sigma0, c):
    """Verbatim from optimisation.py."""
    t_diff_S2C = time[:, np.newaxis] - time_eventS2C[np.newaxis, :]
    t_diff_S2C[t_diff_S2C < 0] = -np.inf
    host_prop_inC = 1 / (1 + (1 / sigma0 - 1) * np.exp(-c * t_diff_S2C))
    host_num_inC = host_population * host_prop_inC

    t_diff_C2I = time[:, np.newaxis] - time_eventC2I[np.newaxis, :]
    t_diff_C2I[t_diff_C2I < 0] = -np.inf
    host_prop_inI = 1 / (1 + (1 / sigma0 - 1) * np.exp(-c * t_diff_C2I))
    host_num_inI = host_population * host_prop_inI

    host_prop_inC = host_prop_inC - host_prop_inI
    host_num_inC = host_num_inC - host_num_inI

    return host_num_inC, host_prop_inC, host_num_inI, host_prop_inI


def detection_probabilities(data_sim, configID):
    """
    Per-simulation detection probability 1 - G_i(T_i), with detection
    applying to symptomatic plants only. Mean over simulations = objective.
    """
    num_sim = len(data_sim)
    P_arr = np.zeros(num_sim)

    t_star_simsarr = np.array([
        np.max(np.concatenate([
            sim['time_1st_S2C'][sim['time_1st_S2C'] != np.inf],
            sim['time_1st_C2I'][sim['time_1st_C2I'] != np.inf]]))
        for sim in data_sim])

    survey_times = [np.arange(0, t_star, SURVEY_FREQ)
                    for t_star in t_star_simsarr]

    host_population = data_sim[0]['host_population'][configID]
    time_S2C = [sim['time_1st_S2C'][configID] for sim in data_sim]
    time_C2I = [sim['time_1st_C2I'][configID] for sim in data_sim]
    mtrees = np.clip(host_population, a_min=1, a_max=NTREES_SURVEY)

    for s in range(num_sim):
        ID_infected = np.where(data_sim[s]['state'] != 'S')[0]
        if np.intersect1d(configID, ID_infected).size == 0:
            continue
        _, _, _, prop_inI = logistic_forOpt(
            time=survey_times[s],
            time_eventS2C=time_S2C[s],
            time_eventC2I=time_C2I[s],
            host_population=host_population,
            sigma0=SIGMA0, c=LOGISTIC_RATE)
        P_arr[s] = 1 - np.prod((1 - P_DETECT * prop_inI) ** mtrees)

    return P_arr


def running_stats(P):
    """Running mean and Monte Carlo standard error, vectorised."""
    M = np.arange(1, len(P) + 1)
    csum = np.cumsum(P)
    csum2 = np.cumsum(P ** 2)
    mean = csum / M
    var = np.zeros_like(mean)
    var[1:] = (csum2[1:] - M[1:] * mean[1:] ** 2) / (M[1:] - 1)
    var = np.clip(var, 0, None)
    mcse = np.sqrt(var / M)
    return M, mean, mcse


# ---------------------------------------------------------------- compute
results = {}
rows = []

for area in AREAS:
    path = os.path.join(SIM_DIR, f"df_sims{area}.joblib")
    print(f"\n=== area {area} ===  loading ...", flush=True)
    sim_data = load(path)

    pop = np.asarray(sim_data[0]['host_population'])
    populated = np.where(pop > 0)[0]

    rng = np.random.default_rng(0)
    configs = {
        "random": rng.choice(populated, size=NSITES, replace=False),
        "most populated": populated[np.argsort(pop[populated])[-NSITES:]],
    }

    results[area] = {}
    for label, cfg in configs.items():
        P = detection_probabilities(sim_data, cfg)
        M, mean, mcse = running_stats(P)
        final = mean[-1]
        results[area][label] = (M, mean, mcse, final)

        row = {'area': area, 'config': label,
               'objective': final,
               'n_zero': int(np.sum(P == 0)),
               'mcse_2000': mcse[-1],
               'rel_mcse_pct': 100 * mcse[-1] / final if final > 0 else np.nan}
        for m in CHECK_M:
            row[f'dev_M{m}'] = abs(mean[m - 1] - final)
        rows.append(row)

        print(f"  {label:15s} obj={final:.4f}  zeros={row['n_zero']:4d}  "
              f"MCSE={mcse[-1]:.5f} ({row['rel_mcse_pct']:.1f}%)  "
              + "  ".join(f"dev@{m}={row[f'dev_M{m}']:.4f}"
                          for m in CHECK_M[:-1]), flush=True)

    del sim_data
    gc.collect()

stats = pd.DataFrame(rows)
stats.round(6).to_csv(OUT_CSV, index=False)

# ---------------------------------------------------------------- summary
print("\n" + "=" * 70)
print("SUMMARY ACROSS ALL 16 CASES (8 sub-landscapes x 2 configurations)")
print("=" * 70)
for m in CHECK_M[:-1]:
    col = f'dev_M{m}'
    worst = stats.loc[stats[col].idxmax()]
    print(f"  max |deviation from final| at M={m:4d}: {stats[col].max():.4f}  "
          f"(area {worst['area']}, {worst['config']})")
print(f"  MCSE at M=2000: {stats['mcse_2000'].min():.4f} - "
      f"{stats['mcse_2000'].max():.4f}")
print(f"  relative MCSE:  {stats['rel_mcse_pct'].min():.1f}% - "
      f"{stats['rel_mcse_pct'].max():.1f}%")
print(f"  objective range: {stats['objective'].min():.4f} - "
      f"{stats['objective'].max():.4f}")

# ---------------------------------------------------------------- figure
# type sizes, as used for the published figure
FS_TICK, FS_LABEL, FS_TITLE, FS_LETTER, FS_LEGEND = 14, 16, 18, 20, 15

# commas separate thousands on the simulation-count axis
COMMA = FuncFormatter(lambda v, p: f"{v:,.0f}")

# 2 x 4 sub-landscapes; each occupies a running-mean panel above an MCSE panel,
# with a spacer row separating the two blocks
fig = plt.figure(figsize=(19, 14), dpi=300)
gs = fig.add_gridspec(5, 4, height_ratios=[1.4, 1, 0.32, 1.4, 1],
                      hspace=0.12, wspace=0.28)
BLOCK_ROWS = [(0, 1), (3, 4)]

colours = {"random": "C0", "most populated": "C1"}
letters = "abcdefgh"

mcse_axes = []
for k, area in enumerate(AREAS):
    block, col = divmod(k, 4)
    r_mean, r_mcse = BLOCK_ROWS[block]
    ax_m = fig.add_subplot(gs[r_mean, col])
    ax_s = fig.add_subplot(gs[r_mcse, col], sharex=ax_m)
    mcse_axes.append(ax_s)

    top = 0
    for label, (M, mean, mcse, final) in results[area].items():
        keep = M >= M_MIN_PLOT
        c = colours[label]
        lower = np.clip(mean[keep] - 1.96 * mcse[keep], 0, None)
        upper = mean[keep] + 1.96 * mcse[keep]
        ax_m.plot(M[keep], mean[keep], lw=0.9, color=c, label=label)
        ax_m.fill_between(M[keep], lower, upper, color=c, alpha=0.2, lw=0)
        ax_m.axhline(final, color=c, ls='--', lw=0.6)
        top = max(top, upper.max())

        # log axis: skip any M where the standard error is still exactly 0
        pos = keep & (mcse > 0)
        ax_s.plot(M[pos], mcse[pos], lw=0.9, color=c)

    ax_m.set_ylim(0, top * 1.05)
    ax_m.set_title(f"Sub-landscape {area}", fontsize=FS_TITLE)
    ax_m.text(-0.26, 1.14, f"{letters[k]}.", transform=ax_m.transAxes,
              fontsize=FS_LETTER, fontweight=700, va='top', ha='left')
    ax_m.tick_params(labelbottom=False)
    ax_s.set_yscale('log')

    for ax in (ax_m, ax_s):
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=FS_TICK)
        ax.xaxis.set_major_formatter(COMMA)

    if col == 0:
        ax_m.set_ylabel('Objective estimate', fontsize=FS_LABEL)
        ax_s.set_ylabel('MCSE', fontsize=FS_LABEL)
    if block == 0:
        ax_s.tick_params(labelbottom=False)
    else:
        ax_s.set_xlabel('Number of simulations ($M$)', fontsize=FS_LABEL)

# common y-range for the MCSE panels so they are comparable
lo = min(ax.get_ylim()[0] for ax in mcse_axes)
hi = max(ax.get_ylim()[1] for ax in mcse_axes)
for ax in mcse_axes:
    ax.set_ylim(lo, hi)

handles = [plt.Line2D([], [], color=colours[l], lw=1.5) for l in colours]
fig.legend(handles, ["Ten randomly selected populated cells",
                     "Ten most densely populated cells"],
           loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=2,
           frameon=False, fontsize=FS_LEGEND)

fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.05)
fig.savefig(OUT_PNG)
print(f"\nSaved {OUT_PNG}\n      {OUT_CSV}")