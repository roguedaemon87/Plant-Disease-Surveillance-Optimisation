"""
Step 2 of the within-cell variation test (Reviewer 1, comment 7).
Writes noise_test_scores.csv, which figureS8.py plots.

For every baseline scenario (N = 10, F = 52: 136 study areas + 8 unconstrained),
takes the optimal configuration from the main analysis and scores it against:
    - the stored original simulations (check: should reproduce the stored
      objective exactly)
    - the noise-0 simulations (same seeds as the noisy runs; the fair baseline)
    - the noisy simulations at each noise level

For each noisy set it also scores 20 configurations that differ from the
optimum by one site, and counts how many beat it (as in Fig. S8).

No optimisation is run. The objective is the same calculation as
objective_more_optimised() in optimisation.py, except that it uses each
cell's own growth rate when the simulations provide one.

Output: noise_test_scores.csv (one row per scenario) and a printed summary.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import glob
import time
import numpy as np
import pandas as pd
from joblib import load
from multiprocessing import Pool
from scipy.stats import kendalltau

from pathlib import Path

# ==========================================================================
# All paths are resolved relative to the repository root.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS    = PROJECT_ROOT / "main_analysis" / "Outputs"
DIR_SIM    = OUTPUTS / "simulations"
DIR_OUT    = OUTPUTS / "noise_test"
DIR_NOISY  = DIR_OUT / "simulations"      # not committed, see the README there
DIR_METRIC = OUTPUTS / "optimisation" / "output_metric_sfreq52_nsite10"

NOISE_LEVELS = [0.25, 0.5]   # noisy sets to test (the 0.0 set is the baseline)
SURVEY_FREQ = 52
N_NEIGHBOURS = 20
N_WORKERS = 8                # each worker holds one area's simulations in memory
# ==========================================================================

from CONSTANT import SIGMA0, LOGISTIC_RATE, P_DETECT, NTREES_SURVEY


def prepare(data_sim):
    """Keep only what the objective needs, computed once rather than per call."""
    out = []
    for sim in data_sim:
        t_all = np.concatenate([sim['time_1st_S2C'][sim['time_1st_S2C'] != np.inf],
                                sim['time_1st_C2I'][sim['time_1st_C2I'] != np.inf]])
        t_star = np.max(t_all)
        out.append({
            'survey_times': np.arange(0, t_star, SURVEY_FREQ),
            'C2I': np.asarray(sim['time_1st_C2I'], dtype=float),
            'infected': np.asarray(sim['state']) != 'S',
            'rate': (np.asarray(sim['logistic_rate'], dtype=float)
                     if 'logistic_rate' in sim else None),
        })
    host_population = np.asarray(data_sim[0]['host_population'])
    return out, host_population


def objective(prep, host_population, config):
    """Same calculation as objective_more_optimised(), with per-cell rates."""
    config = np.asarray(config)
    mtrees = np.clip(host_population[config], a_min=1, a_max=NTREES_SURVEY)
    P = np.zeros(len(prep))
    for s, d in enumerate(prep):
        if not d['infected'][config].any():
            continue
        c = LOGISTIC_RATE if d['rate'] is None else d['rate'][config]
        t_diff = d['survey_times'][:, np.newaxis] - d['C2I'][config][np.newaxis, :]
        t_diff[t_diff < 0] = -np.inf
        with np.errstate(over='ignore'):
            prop_inI = 1 / (1 + (1/SIGMA0 - 1) * np.exp(-c * t_diff))
        f2 = (1 - P_DETECT * prop_inI) ** mtrees
        P[s] = 1 - np.prod(f2)
    return np.mean(P)


def neighbours(config, allowed, rng, n):
    """n configurations differing from config by one site."""
    config = np.asarray(config)
    outside = np.setdiff1d(allowed, config)
    if len(outside) == 0:
        return []
    result = []
    for _ in range(n):
        new = config.copy()
        new[rng.integers(len(config))] = rng.choice(outside)
        result.append(new)
    return result


def process_area(args):
    area, scenarios = args
    t0 = time.perf_counter()

    datasets = {}
    datasets['stored'] = prepare(load(os.path.join(DIR_SIM, f"df_sims{area}.joblib")))
    for noise in [0.0] + NOISE_LEVELS:
        path = os.path.join(DIR_NOISY, f"df_sims{area}_noise{noise}.joblib")
        if os.path.exists(path):
            datasets[noise] = prepare(load(path))

    rows = []
    for k, m in enumerate(scenarios):
        config = np.asarray(m['config_opt'])
        allowed = np.asarray(m['sites_allowed'])
        row = {'area': area, 'road': m['road'], 'n_allowed': len(allowed),
               'obj_stored': m['objective']}

        prep, pop = datasets['stored']
        row['obj_check'] = objective(prep, pop, config)

        rng = np.random.default_rng(1000 * int(area) + k)
        nbrs = neighbours(config, allowed, rng, N_NEIGHBOURS)
        row['n_neighbours'] = len(nbrs)

        for key in [0.0] + NOISE_LEVELS:
            if key not in datasets:
                continue
            prep, pop = datasets[key]
            obj = objective(prep, pop, config)
            row[f'obj_noise{key}'] = obj
            if nbrs:
                nb = np.array([objective(prep, pop, c) for c in nbrs])
                row[f'n_better_noise{key}'] = int(np.sum(nb > obj))
                row[f'max_gain_noise{key}'] = float(np.max(nb - obj))
        rows.append(row)

    mins = (time.perf_counter() - t0) / 60
    return rows, f"area {area}: {len(scenarios)} scenarios in {mins:.1f} min"


if __name__ == "__main__":
    # The noisy simulation sets are not committed (see the README in
    # Outputs/noise_test/). Without them this script would otherwise run to
    # completion and overwrite noise_test_scores.csv with only the baseline
    # columns, which is what Figure S8 is built from. Check first and stop.
    expected = [f"df_sims{a:02d}_noise{n}.joblib"
                for a in range(1, 9) for n in [0.0] + NOISE_LEVELS]
    missing = [f for f in expected if not os.path.exists(os.path.join(DIR_NOISY, f))]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(expected)} noisy simulation files are missing "
            f"from {DIR_NOISY}.\n"
            "These are not distributed with the repository. Generate them with step 1 "
            "of this test, or see the README in that directory.\n"
            "Stopping rather than writing a partial noise_test_scores.csv, which "
            "Figure S8 is built from.")

    files = sorted(glob.glob(os.path.join(DIR_METRIC, "metric_*.joblib")))
    metrics = [load(f) for f in files]
    by_area = {}
    for m in metrics:
        by_area.setdefault(m['area'], []).append(m)
    print(f"{len(metrics)} scenarios across {len(by_area)} areas", flush=True)

    t0 = time.perf_counter()
    all_rows = []
    with Pool(processes=N_WORKERS) as pool:
        for rows, msg in pool.imap_unordered(process_area, sorted(by_area.items())):
            print("  " + msg, flush=True)
            all_rows.extend(rows)
    print(f"Done in {(time.perf_counter() - t0)/60:.1f} min\n")

    df = pd.DataFrame(all_rows).sort_values(['area', 'road']).reset_index(drop=True)

    # OOP = constrained objective / that area's unconstrained objective
    obj_cols = ['obj_stored'] + [c for c in df.columns if c.startswith('obj_noise')]
    for col in obj_cols:
        uncon = df[df.road == 'NoRoad'].set_index('area')[col]
        df['oop' + col[3:]] = df[col] / df['area'].map(uncon)

    df.to_csv(os.path.join(DIR_OUT, "noise_test_scores.csv"), index=False)

    # ---------------- summary ----------------
    print("=== Check: recomputed vs stored objective (should be ~0) ===")
    print(f"  largest difference: {np.max(np.abs(df.obj_check - df.obj_stored)):.2e}\n")

    has_base = 'obj_noise0.0' in df.columns
    base = 'noise0.0' if has_base else 'stored'
    if not has_base:
        print("NOTE: no noise-0 simulations found; comparing against the stored\n"
              "originals, which mixes the noise effect with resampling.\n")

    con = df[df.road != 'NoRoad']

    if has_base:
        d = df['obj_noise0.0'] - df['obj_stored']
        print("=== Reference: noise-0 vs stored (different seeds, no noise) ===")
        print(f"  objective difference: median {np.median(d):+.4f}, "
              f"range {d.min():+.4f} to {d.max():+.4f}")
        print("  (this is the size of difference from resampling alone)\n")

    for noise in [0.0] + NOISE_LEVELS:
        col = f'obj_noise{noise}'
        if col not in df.columns:
            continue
        d = df[col] - df[f'obj_{base}']
        ok = df[f'obj_{base}'] > 0
        rel = 100 * d[ok] / df.loc[ok, f'obj_{base}']
        d_oop = con[f'oop_noise{noise}'] - con[f'oop_{base}']
        tau = kendalltau(con[f'oop_{base}'], con[f'oop_noise{noise}']).statistic

        print(f"=== Noise {noise} vs {base} ===")
        print(f"  objective change:      median {np.median(d):+.4f}, "
              f"range {d.min():+.4f} to {d.max():+.4f}")
        print(f"  relative change:       median {np.median(rel):+.1f}%, "
              f"range {rel.min():+.1f}% to {rel.max():+.1f}%")
        print(f"  OOP change ({len(con)} areas): median {np.median(d_oop):+.4f}, "
              f"largest {np.max(np.abs(d_oop)):.4f}")
        print(f"  Kendall's tau, OOP before vs after: {tau:.3f}")

        nb = df.dropna(subset=[f'n_better_noise{noise}'])
        n_better = int(nb[f'n_better_noise{noise}'].sum())
        n_tested = int(nb['n_neighbours'].sum())
        print(f"  single-swap alternatives beating the optimum: "
              f"{n_better} of {n_tested} "
              f"(in {int((nb[f'n_better_noise{noise}'] > 0).sum())} of {len(nb)} scenarios)")
        if n_better:
            gains = nb[f'max_gain_noise{noise}']
            print(f"  largest gain by an alternative: {gains.max():.4f}")
        print()

    print(f"Full results: {os.path.join(DIR_OUT, 'noise_test_scores.csv')}")
