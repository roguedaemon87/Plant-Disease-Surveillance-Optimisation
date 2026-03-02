"""
NIGERIA COMPARISON: RUN EPIDEMIC SIMULATIONS (WITH PRECOMPUTED KERNEL)
----------------------------------------------------------------------

Purpose
-------
Runs stochastic epidemic simulations for Nigerian state host landscapes.
For each state, a dispersal kernel is first precomputed using compute_kernel()
(from simulation_nigeria.py), then passed into simulate_ibm() to avoid
recomputing pairwise distances for each simulation replicate.

This script is part of a Nigeria-vs-DRC comparison analysis (Ferris et al. 2024 context).

Inputs
------
- Host distribution joblib files:
    main_analysis/code/nigeria_comparison/host_distributions/

Command-line arguments
----------------------
AREA_JA : Nigerian state name used in filenames (e.g. "Anambra", "Kebbi", ...)

Example:
    python run_sims_nigeria.py Anambra

Outputs
-------
Writes simulation results to:
    main_analysis/code/nigeria_comparison/Outputs/simulations/

Notes
-----
This script writes multiple simulation batch files (e.g. 40 batches of 50)
to build up 2000 simulations total (depending on the original settings).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # limit each process to 1 thread
import sys
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]  # because file is in main_analysis/code/nigeria_comparison/code/simulations/
sys.path.insert(0, str(PROJECT_ROOT))

# USER: if auto-detection fails, uncomment and set manually:
# PROJECT_ROOT = Path(r"/path/to/your/repository")

AREA_JA = sys.argv[1]

from multiprocessing import Pool
import numpy as np
import datetime
from joblib import dump, load
from simulation_nigeria import compute_kernel, simulate_ibm
from CONSTANT import ALPHA, BETA, SIGMA0, LOGISTIC_RATE, PREVALENCE

if __name__ == '__main__':

    BASE_DIR = PROJECT_ROOT / "main_analysis" / "code" / "nigeria_comparison"
    INPUT_HOST = BASE_DIR / "host_distributions"
    SIM_OUTPUT_DIR = BASE_DIR / "Outputs" / "simulations"
    SIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    KERNEL_DIR = SIM_OUTPUT_DIR / "kernels"
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    
    ### Read host distribution data
    filename_area = f"{AREA_JA}_prod.joblib"
    host_distr = load(INPUT_HOST / filename_area)
    kernel=compute_kernel(host_distr,alpha=ALPHA, block_size=1000)

    for i in range(40):
      print(f"{i} out of 40 runs")
      np.random.seed(i)
      num_simulations=50
      seeds = np.random.choice(np.arange(10000), size=num_simulations, replace=False).tolist()  # generate different seeds for the simulation
      params_list = [(host_distr, ALPHA, BETA, LOGISTIC_RATE, PREVALENCE, 'risk', 1, 1700, seed, kernel) for seed in seeds]
      
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
      dump(result, SIM_OUTPUT_DIR / filename_simdata)
    
    
    
    
    
    
    
    