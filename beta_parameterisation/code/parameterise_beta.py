"""
BETA PARAMETERISATION (CALIBRATION) SCRIPT
-----------------------------------------

Purpose
-------
This script runs a sweep over candidate transmission parameter values (beta)
and measures epidemic timing metrics in order to choose a plausible beta for
the pathogen spread simulation.

This workflow is OPTIONAL and is NOT required to reproduce the main manuscript
results. The main manuscript uses precomputed host distributions provided in:
    main_analysis/host_distributions/

Inputs
------
Raw GIS shapefiles used to construct a host distribution for this calibration:
- Road shapefile (e.g., northern DRC roads)
- Production/host shapefile (e.g., northern DRC production)

Place required shapefiles in:
    beta_parameterisation/shapefiles_raw/

Outputs
-------
Writes two joblib files to:
    beta_parameterisation/Outputs/

- logbeta_*.joblib    : candidate log(beta) values tested
- time_diff_*.joblib  : for each beta, time difference between reaching two
                        prevalence thresholds (PREV_FINAL0 and PREV_FINAL1)

Index pairing:
    logbeta[i] corresponds to time_diff[i].

Notes
-----
After inspecting outputs, a beta value is selected manually and set in
CONSTANT.py (or your model configuration file). This script does not update
the main analysis automatically.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # avoid oversubscription in linear algebra libs
import sys

from multiprocessing import Pool
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString, box
from shapely.ops import unary_union
import datetime
from scipy.spatial.distance import pdist, squareform
from joblib import dump, load
from simulation import host_distribution_casava, simulate_ibm, logistic
from CONSTANT import ALPHA, BETA, SIGMA0, LOGISTIC_RATE, PREVALENCE
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]   # repo root (beta_parameterisation/code/script.py)
sys.path.insert(0, str(PROJECT_ROOT))

# USER: change this manually if needed
# PROJECT_ROOT = Path(r"/path/to/your/repository")

SHAPEFILE_DIR = PROJECT_ROOT / "beta_parameterisation" / "shapefiles_raw"
OUTPUT_DIR = PROJECT_ROOT / "beta_parameterisation" / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    road_path = SHAPEFILE_DIR / ROAD_SHP_NAME
    area_path = SHAPEFILE_DIR / AREA_SHP_NAME
    cwd = OUTPUT_DIR
    filename_logbeta = f'logbeta_x10y3e5.joblib'
    filename_timediff = f'time_diff_x10y3e5.joblib'    # simulation data file name
    shp_road = gpd.read_file(road_path)
    shp_area = gpd.read_file(area_path)

    host_distr, road_sub = host_distribution_casava(road=shp_road, area=shp_area, 
                                        x_bound=[1e6, 1e6+269000], y_bound=[3e5, 3e5+269000], 
                                        grid_level = True, plot_poly=False, plot_point=False,
                                        plot_save_path=str(OUTPUT_DIR)

    RATE=0.003120016864814313
    PREV_FINAL0=0.1067
    PREV_FINAL1=0.1418
    
    

    np.random.seed(0)
    num_simulations=10
    
    logbeta_arr = np.random.uniform(low=10, high=15, size=num_simulations)
    
    ### ORDER IS: host_distr,alpha, beta, logistic_rate, prev_threshold,entry_method='risk',num_init_S2C=1,init_point_ID=1700,seed=0
    params_list0 = [(host_distr, ALPHA, np.exp(logb), LOGISTIC_RATE, PREV_FINAL0, 'risk', 1, 1700, 0) for logb in logbeta_arr]
    params_list1 = [(host_distr, ALPHA, np.exp(logb), LOGISTIC_RATE, PREV_FINAL1, 'risk', 1, 1700, 0) for logb in logbeta_arr]
    
    
    
    ### parallel processing with timing
    start_time = datetime.datetime.now()
    print("Start time:", start_time, flush=True)
    
    
    ### parallel processing
    with Pool(48) as p:  # use a pool of 48 worker processors
        result0=p.starmap(simulate_ibm, params_list0)
        result1=p.starmap(simulate_ibm, params_list1)
    
    time_at_prev_arr0 = np.zeros_like(logbeta_arr)
    time_at_prev_arr1 = np.zeros_like(logbeta_arr)
    
    for i, sim0 in enumerate(result0):
        all_event_times = np.append(sim0['time_1st_S2C'], sim0['time_1st_C2I'])
        all_event_times=all_event_times[all_event_times!=np.inf]
        time_at_prev_arr0[i]= all_event_times.max()
    
    for j, sim1 in enumerate(result1):
        all_event_times = np.append(sim1['time_1st_S2C'], sim1['time_1st_C2I'])
        all_event_times=all_event_times[all_event_times!=np.inf]
        time_at_prev_arr1[j]= all_event_times.max()
    
    time_diff = time_at_prev_arr1-time_at_prev_arr0
    
    end_time = datetime.datetime.now()
    print("End time:", end_time, flush=True)
    print("Time taken for simulation:", end_time - start_time, flush=True)
    
    ### write data
    dump(logbeta_arr, OUTPUT_DIR / filename_logbeta)
    dump(time_diff, OUTPUT_DIR / filename_timediff)
    
    
    
        
