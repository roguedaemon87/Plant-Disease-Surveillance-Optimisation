"""
##################################
### This file is for parameterising beta such that it takes 2 years to reach prevalence from x% to y%.
### There are 2 sets: Nord-Kivu (16.11% to 24.02%) and Haut-Katanga (1.67% to 15.93%)
### I parameterise beta for both sets to find a reasonable range.
##################################
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # avoid oversubscription in linear algebra libs
import sys
from pathlib import Path
# --- PATH SETUP: make local modules importable ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # repo root (beta_parameterisation/code/script.py)
sys.path.insert(0, str(PROJECT_ROOT))

# USER: if auto-detection fails, uncomment and set manually:
# PROJECT_ROOT = Path(r"/path/to/your/repository")

SHAPEFILE_DIR = PROJECT_ROOT / "beta_parameterisation" / "shapefiles_raw"
OUTPUT_DIR = PROJECT_ROOT / "beta_parameterisation" / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
from CONSTANT import ALPHA, BETA, SIGMA0, LOGISTIC_RATE, PREVALENCE, PREV_FINAL


if __name__ == '__main__':
    area='NordKivu'
    PREV_FINAL0=0.1611
    PREV_FINAL1=0.2402
    
    area='HautKatanga'
    PREV_FINAL0=0.0167
    PREV_FINAL1=0.1593
    road_path = SHAPEFILE_DIR / ROAD_SHP_NAME
    area_path = SHAPEFILE_DIR / AREA_SHP_NAME
    cwd = OUTPUT_DIR
    os.makedirs(cwd, exist_ok=True)
    filename_logbeta = f'logbeta_2500km2_x9.5y3e5_{area}.joblib'
    # filename_time = f'time_2500km2_x9.5y3e5.joblib'    # simulation data file name
    filename_timediff = f'timediff_2500km2_x9.5y3e5_{area}.joblib'    # simulation data file name
    shp_road = gpd.read_file(road_path)
    shp_area = gpd.read_file(area_path)

    host_distr, road_sub = host_distribution_casava(road=shp_road, area=shp_area, 
                                        x_bound=[0.95e6, 0.95e6+50000], y_bound=[3e5, 3e5+50000], 
                                        grid_level = True, plot_poly=False, plot_point=False,
                                        plot_save_path=f'{cwd}')
                                        
    np.random.seed(0)
    num_simulations=10000
    seeds = np.arange(num_simulations)
    logbeta_arr = np.random.uniform(low=2, high=11, size=num_simulations)  # for NordKivu
    logbeta_arr = np.random.uniform(low=6, high=13, size=num_simulations)  # for HautKatanga
    
    
    ### ORDER IS: host_distr,alpha, beta, logistic_rate, prev_threshold,entry_method='risk',num_init_S2C=1,init_point_ID=1700,seed=0
    # params_list = [(host_distr, ALPHA, np.exp(logbeta_arr[i]), LOGISTIC_RATE, PREV_FINAL, 'risk', 1, 1700, seeds[i]) for i in range(num_simulations)]
    params_list0 = [(host_distr, ALPHA, np.exp(logb), LOGISTIC_RATE, PREV_FINAL0, 'risk', 1, 1700, seed) for logb, seed in zip(logbeta_arr, seeds)]
    params_list1 = [(host_distr, ALPHA, np.exp(logb), LOGISTIC_RATE, PREV_FINAL1, 'risk', 1, 1700, seed) for logb, seed in zip(logbeta_arr, seeds)]
    
    print(f'Area={area}, prevalence from {PREV_FINAL0} to {PREV_FINAL1}, num of sims:{num_simulations}, alpha, sigma0, logistic rate are: {ALPHA}, {SIGMA0}, {LOGISTIC_RATE}')
    ### parallel processing with timing
    start_time = datetime.datetime.now()
    print("Start time:", start_time, flush=True)
    
    
    ### parallel processing ====================================
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
    
    
        

        
    ### ========================================================
    end_time = datetime.datetime.now()
    print("End time:", end_time, flush=True)
    print("Time taken for simulation:", end_time - start_time, flush=True)
    
    ### write data
    dump(logbeta_arr, OUTPUT_DIR / filename_logbeta)
    dump(time_diff, OUTPUT_DIR / filename_timediff)