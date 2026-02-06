"""
OUTLIER CASE ANALYSIS (LEGACY SCRIPT)
-----------------------------------

Purpose
-------
Legacy optimisation script used during development of the outlier-case analyses.

This version uses a single numeric scenario identifier (AREA_ROAD_JA) to select
both:
- the simulation file (df_sims{AREA_ROAD}.joblib), and
- the road network file (roadnetwork{AREA_ROAD}.shp).

This design was later replaced by clearer scripts that separate AREA and ROAD
explicitly:

    run_sims_outlier_cases.py
    optimise_sites_with_roads_outlier_cases.py
    optimise_sites_no_roads_outlier_cases.py

Status
------
This script is retained for provenance only and is NOT part of the recommended
reproducible workflow for this repository.

Users should rely on the scripts listed above instead.

Notes
-----
No attempt has been made to refactor this file beyond replacing hard-coded paths.
"""

import os
#os.environ["OMP_NUM_THREADS"] = "1"  # limit each process to 1 thread
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
import sys
from pathlib import Path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]  # script is in main_analysis/code/optimisation/outlier_cases/
sys.path.insert(0, str(PROJECT_ROOT))

# USER: if auto-detection fails, uncomment and set manually:
# PROJECT_ROOT = Path(r"/path/to/your/repository")

#ROAD_JA = sys.argv[1]
#AREA_JA = sys.argv[2]
AREA_ROAD_JA = sys.argv[1]
SURVEY_FREQ_JA = int(sys.argv[2])  # 52, 26, 17 (once a year, twice a year, thrice a year)
NSITES_JA=int(sys.argv[3])  # 5, 10, 20
#P_DETECT_JA = float(sys.argv[1])  # 0.25, 0.5, 0.75, or 0.9  (job array values used for P_DETECT)

task_id = os.environ.get('SLURM_ARRAY_TASK_ID')


from simulation import vlogistic
from CONSTANT import ALPHA, BETA, SIGMA0, LOGISTIC_RATE, PREVALENCE, P_DETECT, NTREES_SURVEY, SURVEY_FREQ, NSITES
from optimisation import read_road, site_loc_allowed, logistic_forOpt, objective_more_optimised, simulated_annealing, opt_metrics

from multiprocessing import Pool
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import datetime
from scipy.spatial.distance import pdist, squareform
from joblib import dump, load
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString, box
from shapely.ops import unary_union
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle

#ROAD=ROAD_JA
#AREA=AREA_JA
AREA_ROAD = AREA_ROAD_JA
NSITES = NSITES_JA  # number of sites to choose for surveillance
SURVEY_FREQ=SURVEY_FREQ_JA

dir_home = PROJECT_ROOT
cwd = PROJECT_ROOT / "main_analysis" / "Outputs" / "outlier_cases" / "optimisation"
cwd.mkdir(parents=True, exist_ok=True)

subdir_sim = "main_analysis/Outputs/outlier_cases/simulations/"
filename_sim=f'df_sims{AREA_ROAD}.joblib'

subdir_road = "main_analysis/code/optimisation/outlier_cases/inputs/road_patterns/"
filename_road=f'roadnetwork{AREA_ROAD}.shp'  

save_dir_metric = 'output_metric/'
save_dir_map = 'output_map/'
save_dir_trace = 'output_trace/'
save_dir_config = 'output_config/'

os.makedirs(os.path.join(cwd, save_dir_metric), exist_ok=True)
os.makedirs(os.path.join(cwd, save_dir_map), exist_ok=True)
os.makedirs(os.path.join(cwd, save_dir_trace), exist_ok=True)
os.makedirs(os.path.join(cwd, save_dir_config), exist_ok=True)

filename_save = f'road{AREA_ROAD}_area{AREA_ROAD}_surveyfreq{SURVEY_FREQ}_nsites{NSITES}'



###############################################################################
####################### RUN OPTIMISATION ######################################
###############################################################################


### start time
start_time = datetime.datetime.now()
print('START TIME:', start_time, flush=True)

### Set seed to ensure reproducability
#seed_used=0
seed_used=np.random.choice(np.arange(10000) )
np.random.seed(seed_used) 

### Read simulation data
sim_data = load(os.path.join(dir_home,subdir_sim,filename_sim))

print(f"seed used:{seed_used}, num of simulations:{len(sim_data)}", flush=True)
print(f"{filename_save}, P(detect)={P_DETECT}, #plants surveyed={NTREES_SURVEY}", flush=True)
print(f'alpha, beta, sigma0, logistic rate, prev are: {ALPHA}, {BETA}, {SIGMA0}, {LOGISTIC_RATE}, {PREVALENCE}', flush=True)

### column bind the x and y coords of the points with hosts >=0 (max of 2500 of them (50x50))
host_xpos = sim_data[0]['x']
host_ypos = sim_data[0]['y']
host_positions = np.column_stack((host_xpos, host_ypos))
host_population=sim_data[0]['host_population']

### Obtain road
shp_road = read_road(file_dir= os.path.join(dir_home, subdir_road, filename_road))
site_loc_allowed_ID, cells_near_road_ID = site_loc_allowed(roadnetwork=shp_road, host_pos=host_positions, host_pop=host_population, accessible_dist=1000)  # 1km accessibility


### initial config (location of the surveillance sites). Set randomly (within road constraints) ================================
config0 = np.random.choice(site_loc_allowed_ID, size=min(len(site_loc_allowed_ID), NSITES), replace=False)  # choose sites randomly from the allowed locations
print('initial config:', config0, flush=True)

### check if len(config0) < len(site_loc_allowed_ID)
print("Check validity of initial config0:", len(config0) < len(site_loc_allowed_ID), flush=True)


### Optimisation (SA) ===========================================================================================================
n_iter =50000 # 
temperature = 10  # find how to set initial temperature!!!!!!!!!!
cooling_rate = 0.9995 # # find how to set cooling rate!!!!!!!!!!
print(f'number of SA trials: {n_iter}, \tstarting temp:{temperature}, \tcooling rate:{cooling_rate}')

best_config,config_list, temp_list, objVal_list, n_iter_actual = simulated_annealing(data_sim=sim_data,   #===
                                                                        obj_fun=objective_more_optimised, 
                                                                        site_loc_allowed=site_loc_allowed_ID, 
                                                                        config0=config0,
                                                                        n_iterations=n_iter, 
                                                                        init_temperature=temperature, 
                                                                        cooling_rate=cooling_rate,
                                                                        ntrees_survey=NTREES_SURVEY,
                                                                        P_detect=P_DETECT, 
                                                                        survey_freq=SURVEY_FREQ) 



road_length, road_coverage_area, road_coverage_plant,RMS_toRoad_wtd,RMS_toRoad, RMS_toAllowed_wtd, RMS_toAllowed, RMS_toOpt_wtd, RMS_toOpt,num_red_optimal, num_red_allowed, prop_red_allowedred,prop_red_allowed = opt_metrics(shp_road, host_positions, host_population, best_config)

metric = {'alpha': ALPHA,
          'beta': BETA,
          'sigma0': SIGMA0,
          'logistic_rate':LOGISTIC_RATE ,
          'prevalence': PREVALENCE,
          'Pdetect': P_DETECT,
          'ntrees_survey': NTREES_SURVEY,
          'road': AREA_ROAD,
          'area': AREA_ROAD,
          'surveyfreq': SURVEY_FREQ,
          'nsites': NSITES,
          'seed_opt': seed_used,
          'num_host_pop_non0': np.sum(host_population>0),
          'sites_allowed': site_loc_allowed_ID,
          'config_init': config0,
          'config_opt': best_config,
          'n_iter_actual': n_iter_actual,
          'road_length': road_length,
          'road_coverage_area': road_coverage_area,
          'road_coverage_plant': road_coverage_plant,
          'RMS_toRoad_wtd': RMS_toRoad_wtd,
          'RMS_toRoad': RMS_toRoad,
          'RMS_toAllowed_wtd': RMS_toAllowed_wtd, 
          'RMS_toAllowed': RMS_toAllowed,
          'RMS_toOpt_wtd': RMS_toOpt_wtd,
          'RMS_toOpt': RMS_toOpt,
          'num_red_optimal': num_red_optimal,
          'num_red_allowed': num_red_allowed,
          'prop_red_allowedred': prop_red_allowedred,
          'prop_red_allowed': prop_red_allowed,
          'objective': objVal_list[-1]
          }



print(f"best config:{best_config}, \nhost population of the best config: {host_population[best_config]}", flush=True)
print(f"best_objective:{objVal_list[-1]}")

### Save data
dump(objVal_list, os.path.join(cwd, save_dir_trace, f'trace_objval_{filename_save}.joblib'))
dump(config_list, os.path.join(cwd, save_dir_config, f'config_{filename_save}.joblib'))  #+++
dump(metric, os.path.join(cwd, save_dir_metric, f'metric_{filename_save}.joblib') ) 
###############

end_time = datetime.datetime.now()
print("End time:", end_time, flush=True)
print("Time taken for simulation:", end_time - start_time, flush=True)
#==================================================================================================================================


### Plot trace of objective value
#fig, ax = plt.subplots(figsize = (10,6))
#plt.plot(objVal_list , label = '')
##plt.xscale('log', base = 10)
#plt.xlabel('Trial')
#plt.ylabel('Objective value')
#plt.title(f'Trace of SA ({n_iter} trials, start temp={temperature}, cooling rate={cooling_rate}), {filename_save}')
## plt.legend()
#plt.savefig(os.path.join(cwd, save_dir_trace, f'plot_OptTrace_{filename_save}.pdf'), dpi=600)
#plt.clf # clear figure
#plt.close(fig)



### Plot optimisation result
fig, ax = plt.subplots(figsize = (10,6))
fig0=ax.scatter(host_positions[:, 0] ,host_positions[:, 1],  marker='s', s=30, linewidths=0, edgecolors='none', alpha=0.7, c=host_population, cmap='RdYlGn_r')
fig.colorbar(fig0, ax=ax, label='number of plants in cell', pad=0.005)
ax.scatter(host_positions[host_population==0][:, 0] ,host_positions[host_population==0][:, 1],  marker='s', s=30, linewidths=0, edgecolors='none', alpha=1, c='white')
shp_road.plot(ax=ax, color='black', linewidth=1, label='road', alpha=0.5)
ax.scatter(host_positions[site_loc_allowed_ID, 0], host_positions[site_loc_allowed_ID, 1], marker='s', s=30, facecolors='none', edgecolors='black', label='sites allowed for survey', alpha=0.6)
ax.scatter(host_positions[best_config, 0], host_positions[best_config, 1], marker='x', s=20, c='black', label='optimal sites', alpha=0.6)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'{filename_save}')
plt.legend()
plt.savefig(os.path.join(cwd, save_dir_map, f'plot_OptResult_{filename_save}.pdf'), dpi=600)
plt.clf # clear figure
plt.close(fig)

