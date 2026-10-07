"""
FIGURE S9 - pairwise Pearson correlations among the twelve road-network
accessibility metrics, across the 136 accessibility-constrained study areas
at the baseline surveillance intensity (N = 10 sites, F = 52 weeks).

Follows cells 28-29 of Biqing's plots_for_paper.ipynb.

    python figureS9.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
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


OUT_PNG = os.path.join(DIR_OUT, "FigureS9.png")
OUT_CSV = os.path.join(DIR_OUT, "FigureS9_pearson.csv")

NSITES, SFREQ = 10, 52          # baseline
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
mpl.rcParams['font.size'] = 12

# ---------------------------------------------------------------- inputs
distr_list = [load(os.path.join(DIR_DISTR, f"area{i:02d}.joblib"))
              for i in range(1, 9)]

host_pos0 = np.asarray(distr_list[0]['xy'])
xmin, xmax = host_pos0[:, 0].min(), host_pos0[:, 0].max()
ymin, ymax = host_pos0[:, 1].min(), host_pos0[:, 1].max()
BOUNDARIES = np.vstack([
    host_pos0[host_pos0[:, 0] == xmin], host_pos0[host_pos0[:, 0] == xmax],
    host_pos0[host_pos0[:, 1] == ymin], host_pos0[host_pos0[:, 1] == ymax]])


def flatten_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.item() if v.ndim == 0 else v.tolist()
        else:
            out[k] = v
    return out


def dist_to_boundary(host_distr, allowed, weighted, total):
    pos = np.asarray(host_distr['xy'])[allowed]
    shortest = np.min(cdist(pos, BOUNDARIES), axis=1)
    if weighted:
        pop = np.asarray(host_distr['host_population'])
        vals = shortest * (pop[allowed] / pop.sum())
    else:
        vals = shortest
    return vals.sum() if total else vals.mean()


def rms_p2ap(host_distr, allowed, weighted):
    pos = np.asarray(host_distr['xy'])
    pop = np.asarray(host_distr['host_population'])
    non0 = np.where(pop > 0)[0]
    shortest = np.min(cdist(pos[non0], pos[allowed]), axis=1)
    if weighted:
        w = pop[non0] / pop.sum()
        return (np.mean((shortest ** 2) * w)) ** 0.5
    return (np.mean(shortest ** 2)) ** 0.5


def build_metrics(nsites, sfreq):
    dir_metric = os.path.join(DIR_OPT,
                              f"output_metric_sfreq{sfreq}_nsite{nsites}")
    files = [f for f in os.listdir(dir_metric) if f.endswith(".joblib")]
    road = [load(os.path.join(dir_metric, f)) for f in files
            if 'NoRoad' not in f]
    noroad = [load(os.path.join(dir_metric, f)) for f in files
              if 'NoRoad' in f]

    df = pd.DataFrame([flatten_dict(d) for d in road])
    df_nr = pd.DataFrame([flatten_dict(d) for d in noroad])[
        ['area', 'objective']].rename(columns={'objective': 'best_objective'})

    df = df.merge(df_nr, on='area', how='left')
    df['objective_prop'] = df['objective'] / df['best_objective']
    df = df.sort_values(['area', 'road']).reset_index(drop=True)

    df['sites_allowed_len'] = df['sites_allowed'].apply(len)
    df['CPA'] = df['sites_allowed_len'] / df['num_host_pop_non0']
    df['RCA'] = df['road_coverage_area']
    df['RCP'] = df['road_coverage_plant']
    df['RMS_All2AP'] = df['RMS_toAllowed']
    df['RMS_All2AP_wtd'] = df['RMS_toAllowed_wtd']

    idx = [int(a) - 1 for a in df['area']]
    allowed = df['sites_allowed'].tolist()

    df['MDB'] = [dist_to_boundary(distr_list[idx[i]], allowed[i], False, False)
                 for i in range(len(df))]
    df['MDB_wtd'] = [dist_to_boundary(distr_list[idx[i]], allowed[i], True, False)
                     for i in range(len(df))]
    df['TDB'] = [dist_to_boundary(distr_list[idx[i]], allowed[i], False, True)
                 for i in range(len(df))]
    df['TDB_wtd'] = [dist_to_boundary(distr_list[idx[i]], allowed[i], True, True)
                     for i in range(len(df))]
    df['RMS_P2AP'] = [rms_p2ap(distr_list[idx[i]], allowed[i], False)
                      for i in range(len(df))]
    df['RMS_P2AP_wtd'] = [rms_p2ap(distr_list[idx[i]], allowed[i], True)
                          for i in range(len(df))]
    return df


METRICS = ['road_length', 'RCA', 'CPA', 'RCP',
           'MDB', 'MDB_wtd', 'TDB', 'TDB_wtd',
           'RMS_All2AP', 'RMS_All2AP_wtd', 'RMS_P2AP', 'RMS_P2AP_wtd']

print(f"building metrics for N={NSITES}, F={SFREQ} ...", flush=True)
metric_df = build_metrics(NSITES, SFREQ)
print(f"  {len(metric_df)} study areas")

corr = metric_df[METRICS].corr(method='pearson')
corr.round(4).to_csv(OUT_CSV)

print("\nStrongest off-diagonal pairs:")
off = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack().sort_values(
    ascending=False)
seen = set()
for (a, b), v in off.items():
    if (b, a) in seen:
        continue
    seen.add((a, b))
    print(f"  {a:<16} {b:<16} {corr.loc[a, b]:+.3f}")
    if len(seen) >= 6:
        break

# ---------------------------------------------------------------- figure
# red at both extremes, green at zero: strong association in either
# direction reads as red
cmap = LinearSegmentedColormap.from_list(
    "symmetric_red", [(1, 0, 0), (0, 1, 0), (1, 0, 0)], N=256)

fig, ax = plt.subplots(figsize=(10.5, 9), dpi=300)

sns.heatmap(corr, annot=True, fmt='.2f', annot_kws={'size': 9},
            cmap=cmap, vmin=-1, vmax=1, square=True, ax=ax,
            linewidths=0.4, linecolor='white',
            cbar_kws={"shrink": 0.75})

cbar = ax.collections[0].colorbar
cbar.set_label("Pearson correlation coefficient", fontsize=13)
cbar.ax.tick_params(labelsize=11)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

fig.tight_layout()
fig.savefig(OUT_PNG)
print(f"\nSaved {OUT_PNG}\n      {OUT_CSV}")
