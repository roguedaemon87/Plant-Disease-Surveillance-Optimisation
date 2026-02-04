"""
MAIN ANALYSIS: RUN EPIDEMIC SIMULATIONS
--------------------------------------

Purpose
-------
Runs stochastic epidemic spread simulations on precomputed host distribution
landscapes (area01.joblib, area02.joblib, ...). These simulation realisations
are later used as inputs for surveillance optimisation.

Inputs
------
- main_analysis/host_distributions/areaXX.joblib
** Note ** The code below runs 2000 simulations for area01 only. Please change
the input file (e.g. area02, area03 etc.) if you want to run the epidemic
simulation in a different area.

Outputs
-------
- main_analysis/Outputs/simulations/ (joblib files containing simulation results)

Notes
-----
This script does not perform optimisation. It generates simulation realisations
only.
"""


##################################
### Run 2000 simulations in 2500km2 area in North DRC (area01). ALPHA=3.5, BETA=np.exp(8), SIGMA0=0.01, LOGISTIC_RATE=0.1693, PREVALENCE=0.05 
### 
### 
##################################

import os
os.environ["OMP_NUM_THREADS"] = "1"  # limit each process to 1 thread
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
import sys
from pathlib import Path
# --- PATH SETUP: make local modules importable ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]  # ana00_runSimulation/run_sims.py (or main_analysis/code/simulations/...) -> repo root
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
    cwd = PROJECT_ROOT / "main_analysis" / "Outputs" / "simulations"
    cwd.mkdir(parents=True, exist_ok=True)
    
    dir_area = PROJECT_ROOT / "main_analysis" / "host_distributions"
    filename_area = f'area{AREA_JA}.joblib'

    filename_simdata= f'df_sims{AREA_JA}.joblib'

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
    
    
    
    
    
    
    
    
