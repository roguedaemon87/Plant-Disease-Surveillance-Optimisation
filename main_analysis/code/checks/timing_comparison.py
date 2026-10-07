"""
Timing comparison: cost of imposing the accessibility constraint.

Answers Reviewer 1's question on the runtime cost of restricting the
candidate site set, by timing:

  (a) the one-off accessibility preprocessing (buffer road, test centroids)
  (b) a single objective evaluation under the accessibility constraint
  (c) a single objective evaluation with no constraint

(b) and (c) should be indistinguishable: the constraint restricts which
cells may be proposed, not the cost of evaluating a proposal.

USAGE
-----
python main_analysis/code/checks/timing_comparison.py
"""

import time
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point
from joblib import load

# ==========================================================================
# All paths are resolved relative to the repository root.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

AREA = "01"
ROAD = "01_0"

SIM_FILE  = PROJECT_ROOT / "main_analysis" / "Outputs" / "simulations" / f"df_sims{AREA}.joblib"
ROAD_FILE = PROJECT_ROOT / "main_analysis" / "road_patterns" / f"roadnetwork{ROAD}.shp"
# ==========================================================================

NSITES = 10
SURVEY_FREQ = 52
NTREES_SURVEY = 30
P_DETECT = 0.75
SIGMA0 = 0.01
LOGISTIC_RATE = 0.1693
ACCESSIBLE_DIST = 1000

N_REPEATS = 20      # objective evaluations to time


def logistic_forOpt(time_arr, time_eventS2C, time_eventC2I, host_population,
                    sigma0, c):
    t_diff_S2C = time_arr[:, np.newaxis] - time_eventS2C[np.newaxis, :]
    t_diff_S2C[t_diff_S2C < 0] = -np.inf
    host_prop_inC = 1 / (1 + (1 / sigma0 - 1) * np.exp(-c * t_diff_S2C))
    host_num_inC = host_population * host_prop_inC

    t_diff_C2I = time_arr[:, np.newaxis] - time_eventC2I[np.newaxis, :]
    t_diff_C2I[t_diff_C2I < 0] = -np.inf
    host_prop_inI = 1 / (1 + (1 / sigma0 - 1) * np.exp(-c * t_diff_C2I))
    host_num_inI = host_population * host_prop_inI

    host_prop_inC = host_prop_inC - host_prop_inI
    host_num_inC = host_num_inC - host_num_inI
    return host_num_inC, host_prop_inC, host_num_inI, host_prop_inI


def objective(data_sim, configID, t_star_arr, survey_times):
    """objective_more_optimised(), returning the mean only."""
    P_arr = np.zeros(len(data_sim))
    host_population = data_sim[0]['host_population'][configID]
    for s, sim in enumerate(data_sim):
        ID_infected = np.where(sim['state'] != 'S')[0]
        if np.intersect1d(configID, ID_infected).size == 0:
            continue
        _, _, _, prop_inI = logistic_forOpt(
            survey_times[s], sim['time_1st_S2C'][configID],
            sim['time_1st_C2I'][configID], host_population,
            SIGMA0, LOGISTIC_RATE)
        prop_detectable = prop_inI
        mtrees = np.clip(host_population, a_min=1, a_max=NTREES_SURVEY)
        P_arr[s] = 1 - np.prod((1 - P_DETECT * prop_detectable) ** mtrees)
    return P_arr.mean()


if __name__ == "__main__":

    print("Loading simulations ...", flush=True)
    t0 = time.perf_counter()
    sim_data = load(SIM_FILE)
    print(f"  load time: {time.perf_counter() - t0:.1f} s\n")

    host_pos = np.column_stack((sim_data[0]['x'], sim_data[0]['y']))
    host_pop = sim_data[0]['host_population']

    # ---- (a) accessibility preprocessing -------------------------------
    print("Timing accessibility preprocessing ...", flush=True)
    t0 = time.perf_counter()
    road = gpd.read_file(ROAD_FILE)
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    road_shapely = unary_union(road.geometry.tolist())
    points = [Point(x, y) for x, y in zip(host_pos[:, 0], host_pos[:, 1])]
    distances = road_shapely.distance(points)
    allowed = np.where((host_pop > 0) & (distances <= ACCESSIBLE_DIST))[0]
    t_buffer = time.perf_counter() - t0

    unconstrained = np.where(host_pop > 0)[0]

    print(f"  shapefile read:            {t_read*1000:8.1f} ms")
    print(f"  buffer + centroid test:    {t_buffer*1000:8.1f} ms")
    print(f"  candidate cells, constrained:   {len(allowed)}")
    print(f"  candidate cells, unconstrained: {len(unconstrained)}\n")

    # precompute survey schedules (identical for both scenarios)
    t_star_arr = np.array([
        np.max(np.concatenate([
            s['time_1st_S2C'][s['time_1st_S2C'] != np.inf],
            s['time_1st_C2I'][s['time_1st_C2I'] != np.inf]]))
        for s in sim_data])
    survey_times = [np.arange(0, t, SURVEY_FREQ) for t in t_star_arr]

    # ---- (b) and (c) objective evaluation ------------------------------
    rng = np.random.default_rng(0)

    for label, pool in [("constrained", allowed),
                        ("unconstrained", unconstrained)]:
        times = []
        for _ in range(N_REPEATS):
            cfg = rng.choice(pool, size=NSITES, replace=False)
            t0 = time.perf_counter()
            objective(sim_data, cfg, t_star_arr, survey_times)
            times.append(time.perf_counter() - t0)
        times = np.array(times) * 1000
        print(f"{label:>14} objective evaluation: "
              f"{times.mean():7.1f} +/- {times.std():.1f} ms  "
              f"(n={N_REPEATS})")

    print(f"\nAt 50,000 SA iterations, preprocessing is "
          f"{100 * (t_read + t_buffer) / (50000 * times.mean() / 1000):.4f}% "
          f"of total optimisation time.")