##################################
### Run 2000 simulations in 2500km2 area in North DRC (area01). ALPHA=3.5, BETA=np.exp(8), SIGMA0=0.01, LOGISTIC_RATE=0.1693, PREVALENCE=0.05 
### 
### 
##################################

"""
OUTLIER CASE ANALYSIS: RUN EPIDEMIC SIMULATIONS
---------------------------------------------

Purpose
-------
Runs stochastic epidemic spread simulations on selected host distribution
landscapes corresponding to outlier cases in which road-based accessibility
metrics performed poorly.

These simulation realisations are used as inputs for the outlier-case
optimisation scripts, which investigate why certain road configurations lead
to unexpectedly low detection performance.

Inputs
------
Host distribution joblib files located in:

    main_analysis/code/optimisation/outlier_cases/inputs/host_distributions/

Confirmed outlier cases:
- area01.joblib
- area05.joblib

Command-line arguments
----------------------
AREA_JA : area identifier (e.g. "01" or "05")

Example:
    python run_sims_outlier_cases.py 01

Outputs
-------
Simulation outputs are written to:

    main_analysis/Outputs/outlier_cases/simulations/

as:

    df_simsXX.joblib

Notes
-----
This script uses the same epidemiological model and parameters as the main
analysis. Only the host distribution inputs differ.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # limit each process to 1 thread
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
import sys
from pathlib import Path

# --- PATH SETUP: make local modules importable ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]  # script is in main_analysis/code/optimisation/outlier_cases/
sys.path.insert(0, str(PROJECT_ROOT))

# USER: if auto-detection fails, uncomment and set manually:
# PROJECT_ROOT = Path(r"/path/to/your/repository")

AREA_JA = sys.argv[1]

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

if __name__ == '__main__':

    dir_home = PROJECT_ROOT

    # Where to save simulation outputs for outlier-case investigations
    cwd = PROJECT_ROOT / "main_analysis" / "Outputs" / "outlier_cases" / "simulations"
    cwd.mkdir(parents=True, exist_ok=True)
    
    # Where to read host distributions for outlier cases (you uploaded these here)
    dir_area = PROJECT_ROOT / "main_analysis" / "code" / "optimisation" / "outlier_cases" / "inputs" / "host_distributions"

    filename_area = f'area{AREA_JA}.joblib'

    filename_simdata= f'df_sims{AREA_JA}.joblib'
    

    #shp_road = gpd.read_file(f'{dir_home}/shapefiles/road northern DRC (BasSudNodMon_Survey_Road).shp')
    #shp_area = gpd.read_file(f'{dir_home}/shapefiles/production northern DRC (northDRC_prod_1km2).shp')

    #host_distr, road_sub = host_distribution_casava(road=shp_road, area=shp_area, 
    #                                    x_bound=[0.95e6, 0.95e6+50000], y_bound=[3e5, 3e5+50000], 
    #                                    grid_level = True, plot_poly=False, plot_point=False,
    #                                    plot_save_path=f'{cwd}')
                                        
    ### Read host distribution data
    host_distr = load(dir_area / filename_area)

    
    np.random.seed(0)
    num_simulations=2000
    seeds = np.random.choice(np.arange(10000), size=num_simulations, replace=False).tolist()  # generate different seeds for the simulation
    params_list = [(host_distr, ALPHA, BETA, LOGISTIC_RATE, PREVALENCE, 'risk', 1, 1700, seed) for seed in seeds]
    
    print(f'num of sims:{num_simulations}, alpha, beta, sigma0, logistic rate, prev are: {ALPHA}, {BETA}, {SIGMA0}, {LOGISTIC_RATE}, {PREVALENCE}')
    ### parallel processing with timing
    start_time = datetime.datetime.now()
    print("Start time:", start_time, flush=True)
    
    ### parallel processing ======================================================================
    with Pool(48) as p:  # use a pool of 48 worker processors
        result=p.starmap(simulate_ibm, params_list)
    #=============================================================================================
    
    end_time = datetime.datetime.now()
    print("End time:", end_time, flush=True)
    print("Time taken for simulation:", end_time - start_time, flush=True)

    ### write data
    dump(result, cwd / filename_simdata)
