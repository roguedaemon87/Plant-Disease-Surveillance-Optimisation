"""
FIGURE 5 - absolute correlation strength between road network metrics and
optimal objective proportion, across the nine surveillance intensities.

  a. Absolute Kendall's tau.  b. Absolute distance correlation.

Twelve metrics. Each weighted metric is drawn dashed in the same colour as
its unweighted counterpart, so eight colours cover all twelve lines. The
four shortlisted metrics (RCP, CPA, TDB, road_length) are drawn bold; all
four are unweighted and therefore solid, so the two cues do not conflict.

Computation follows metric_kendal_dcor() in Biqing's plots_for_paper.ipynb.

    python figure5.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.spatial.distance import cdist
from joblib import load
import dcor

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

OUT_PNG = os.path.join(DIR_OUT, "Figure5.png")
OUT_CSV_K = os.path.join(DIR_OUT, "Figure5_kendall.csv")
OUT_CSV_D = os.path.join(DIR_OUT, "Figure5_dcor.csv")

NSITES_LIST = [5, 10, 15]
SFREQ_LIST = [26, 52, 104]          # survey interval, weeks
SHORTLIST = ['RCP', 'CPA', 'TDB', 'road_length']

# one colour per unweighted metric; the weighted variant reuses it, dashed
COLOURS = {
    'road_length': '#0072B2',   # blue
    'RCA':         '#8B4513',   # brown
    'CPA':         '#D55E00',   # vermillion
    'RCP':         '#009E73',   # green
    'MDB':         '#CC79A7',   # pink
    'TDB':         '#E69F00',   # orange
    'RMS_All2AP':  '#56B4E9',   # sky blue
    'RMS_P2AP':    '#444444',   # dark grey
}
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

# landscape boundary cells (same grid for every sub-landscape)
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
    """Mean or total shortest distance from accessible cells to landscape edge."""
    pos = np.asarray(host_distr['xy'])[allowed]
    shortest = np.min(cdist(pos, BOUNDARIES), axis=1)
    if weighted:
        pop = np.asarray(host_distr['host_population'])
        w = pop[allowed] / pop.sum()
        vals = shortest * w
    else:
        vals = shortest
    return vals.sum() if total else vals.mean()


def rms_p2ap(host_distr, allowed, weighted):
    """RMS distance from each populated cell to the nearest accessible cell."""
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


def style(metric):
    """Colour from the unweighted base metric; dashed if weighted."""
    weighted = metric.endswith('_wtd')
    base = metric[:-4] if weighted else metric
    return COLOURS[base], ('--' if weighted else '-')


kendall, dcorr, labels = {}, {}, []

for w in NSITES_LIST:
    for s in SFREQ_LIST:
        col = f"({w}, {s})"
        labels.append(col)
        print(f"building {col} ...", flush=True)
        df = build_metrics(w, s)
        print(f"  {len(df)} study areas")

        sub = df[METRICS + ['objective_prop']]
        k = sub.corr(method='kendall')['objective_prop'].drop('objective_prop')
        kendall[col] = k.abs()
        dcorr[col] = pd.Series(
            {m: dcor.distance_correlation(df[m].values.astype(float),
                                          df['objective_prop'].values)
             for m in METRICS})

kendall_df = pd.DataFrame(kendall).loc[METRICS]
dcorr_df = pd.DataFrame(dcorr).loc[METRICS]

kendall_df.round(4).to_csv(OUT_CSV_K)
dcorr_df.round(4).to_csv(OUT_CSV_D)

print("\nAbsolute Kendall's tau:\n", kendall_df.round(3))
print("\nDistance correlation:\n", dcorr_df.round(3))
print("\nHighest per column (Kendall):", kendall_df.idxmax().to_dict())
print("Highest per column (dcor):   ", dcorr_df.idxmax().to_dict())

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69), dpi=300, sharex=True)
x = np.arange(len(labels))

for ax, data, letter, ylab in [
        (axes[0], kendall_df, 'a',
         "Absolute Kendall's $\\tau$ correlation coefficient"),
        (axes[1], dcorr_df, 'b',
         "Absolute distance correlation")]:

    for m in METRICS:
        bold = m in SHORTLIST
        colour, dash = style(m)
        ax.plot(x, data.loc[m].values, marker='o', ms=3.5,
                lw=2.4 if bold else 1.1, ls=dash,
                alpha=1.0 if bold else 0.75,
                color=colour, label=m, zorder=3 if bold else 2)

    ax.set_xticks(x)
    ax.set_xlim(-0.4, len(labels) - 0.6)
    ax.set_ylabel(ylab, fontsize=13)
    ax.set_ylim(0, 1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes,
            fontsize=16, fontweight=700, va='top', ha='left')

axes[1].set_xticklabels(labels, rotation=45, ha='right')
axes[1].set_xlabel('Surveillance intensity (number of sites, survey interval)',
                   fontsize=13)

handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc='center left', bbox_to_anchor=(0.78, 0.58),
           frameon=False, fontsize=9.5)

fig.subplots_adjust(left=0.13, right=0.74, top=0.97, bottom=0.12, hspace=0.10)
fig.savefig(OUT_PNG)
print(f"\nSaved {OUT_PNG}\n      {OUT_CSV_K}\n      {OUT_CSV_D}")
