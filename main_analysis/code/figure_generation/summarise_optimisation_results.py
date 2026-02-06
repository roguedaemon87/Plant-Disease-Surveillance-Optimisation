"""
MAIN ANALYSIS: SUMMARISE OPTIMISATION RESULTS
--------------------------------------------

Purpose
-------
Post-processes optimisation outputs (metrics, traces, configurations) and
computes derived road accessibility metrics and site-level diagnostics.

This script was used to assemble summary datasets for downstream plotting and
statistical analysis (e.g., correlations between road metrics and optimisation
performance).

Inputs
------
Expected optimisation outputs produced by scripts in:
    main_analysis/code/optimisation/

This script reads:
- per-area host distributions (area01 ... area08)
- infection visit count summaries (infection_visit_count_8areas.joblib)
- optimisation metrics (metric_*.joblib)
- optimisation traces (trace_objval_*.joblib)
- optimisation configs (config_*.joblib)

Command-line arguments
----------------------
SURVEY_FREQ_JA : survey frequency (integer)
NSITES_JA      : number of sites (integer)

Example:
    python summarise_optimisation_results.py 104 10

Outputs
-------
Writes derived result objects (joblib) to:
    main_analysis/Outputs/results_summaries/

Notes
-----
This is a post-processing script; it does not run simulations or optimisations.
"""

import os
#os.environ["OMP_NUM_THREADS"] = "1"  # limit each process to 1 thread
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
import sys
from pathlib import Path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]  # if placed in main_analysis/code/figure_generation/
sys.path.insert(0, str(PROJECT_ROOT))

# USER: if auto-detection fails:
# PROJECT_ROOT = Path(r"/path/to/your/repository")

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
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from joblib import load, dump
import geopandas as gpd
#from esda.moran import Moran
from libpysal.weights import lat2W, DistanceBand
from scipy.spatial.distance import cdist, pdist
from scipy.stats import kendalltau
#from xicorrelation import xicorr
import dcor


SURVEY_FREQ_JA = int(sys.argv[1])
NSITES_JA=int(sys.argv[2]) 

SURVEY_FREQ=SURVEY_FREQ_JA
NSITES = NSITES_JA

cwd = PROJECT_ROOT / "main_analysis" / "Outputs"

dir_host_distr = PROJECT_ROOT / "main_analysis" / "host_distributions"
dir_infection_visit = cwd / "derived_inputs" / "infection_visit_count"

dir_metric = cwd / f"optimisation/output_metric_sfreq{SURVEY_FREQ}_nsite{NSITES}"
dir_config = cwd / "optimisation" / "output_config"
dir_trace  = cwd / "optimisation" / "output_trace"

dir_data_output = cwd / "results_summaries"
dir_data_output.mkdir(parents=True, exist_ok=True)

filename_save = f"sfreq{SURVEY_FREQ}_nsite{NSITES}"
print(filename_save, flush=True)

"""Read distributions ========================================="""
distr_list = [
    load(dir_host_distr / f"area{i:02d}.joblib") for i in range(1, 9)
]

"""Read infection visit count ========================================="""
visit_count_list = load(dir_infection_visit / "infection_visit_count_8areas.joblib")


"""Read road metric ==========================================="""
### get filenames
filenames_road=[f for f in os.listdir(dir_metric) if ('NoRoad' not in f) & (".joblib" in f)]
print(filenames_road, flush=True)

filenames_noroad=[f for f in os.listdir(dir_metric) if 'NoRoad' in f]
print(filenames_noroad, flush=True)

### read metrics
metric_list = [
    load(os.path.join(dir_metric, f)) for f in filenames_road
]

metric_listNoRoad = [
    load(os.path.join(dir_metric, f)) for f in filenames_noroad
]


### Combine road & no road metrics
def flatten_dict(d):
    """
    d = dictionary
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                out[k] = v.item()
            else:
                out[k] = v.tolist()
        else:
            out[k] = v
    return out

metric_list_flat = [flatten_dict(dic) for dic in metric_list]
metric_df = pd.DataFrame(metric_list_flat)

metric_list_flatNoRoad=[flatten_dict(dic) for dic in metric_listNoRoad]
metric_dfNoRoad=pd.DataFrame(metric_list_flatNoRoad)[['area','objective']].rename(columns={'objective':'best_objective'})

### appending best objective
metric_df = metric_df.merge(metric_dfNoRoad, on='area', how='left')
metric_df=metric_df.sort_values(['area', 'road'], ascending=[True, True])

### Create metrics
metric_df['objective_prop'] = metric_df['objective']/metric_df['best_objective']  #OOP
metric_df['roadgroup']=metric_df['road'].str[:2]
metric_df['sites_allowed_len']=metric_df['sites_allowed'].apply(len)
metric_df['RCAP'] = metric_df['sites_allowed_len']/metric_df['num_host_pop_non0']
metric_df['RCA'] = metric_df['road_coverage_area']
metric_df['RCP'] = metric_df['road_coverage_plant']
metric_df['RMS_All2AP'] = metric_df['RMS_toAllowed']
metric_df['RMS_All2AP_wtd'] = metric_df['RMS_toAllowed_wtd']

areas = metric_df['area'].tolist()
roads = metric_df['road'].tolist()
area_road_perm = np.column_stack([areas, roads])    # permutations between areas and roads
print('len(area_road_perm):', len(area_road_perm), flush=True)




"""Create site metrics ==============================================================="""
def site_metric(area, road, dir_config, dir_trace, road_metric_df, host_distr, topn):
    """
    area='01', road='01_0'
    """
    ID_site_allowed = road_metric_df[(road_metric_df['area']==area) & (road_metric_df['road']==road)]['sites_allowed'].item()    #list
    host_pop_allowed=host_distr['host_population'][ID_site_allowed]     #arr
    # host_pop_allowed_topn_sum = sum(sorted(host_pop_allowed, reverse=True)[:10])
    host_pop_allowed_topn = sorted(host_pop_allowed, reverse=True)[:topn]    # topn population
    topn_ID = np.where(np.isin(host_pop_allowed, host_pop_allowed_topn))[0]   # take the indices of allowed cells with population = any of the topn population
    # print(topn_ID)
    # print(type(ID_site_allowed))
    host_pop_allowed_topn_ID = np.array(ID_site_allowed)[topn_ID]   # locate the site ID for these 10 (or more if some have the same populations)most populated cells
    # print(host_pop_allowed_topn_ID)
    host_pos_allowed = host_distr['xy'][ID_site_allowed]    #nx2 np.array
    pairwisedist_allowed_sum = np.sum(pdist(host_pos_allowed))

    ###
    areaID = int(area)-1
    visit_count_area = visit_count_list[areaID]
    visit_count_area_allowed = visit_count_area[ID_site_allowed]
    visit_allowed_topn = sorted(visit_count_area_allowed, reverse=True)[:topn]
    vtopn_ID = np.where(np.isin(visit_count_area_allowed, visit_allowed_topn))[0]
    visit_allowed_topn_ID = np.array(ID_site_allowed)[vtopn_ID]


    config_list = np.vstack(load(os.path.join(dir_config, f'config_road{road}_area{area}_surveyfreq{SURVEY_FREQ}_nsites{topn}.joblib')))
    trace_list = load(os.path.join(dir_trace, f'trace_objval_road{road}_area{area}_surveyfreq{SURVEY_FREQ}_nsites{topn}.joblib'))
    
    best_objective = trace_list[-1]
    # print(best_objective)
    trace_prop_arr = np.array(trace_list)/best_objective

    ### take the unique configs and their indices
    _, unique_ID = np.unique(config_list, axis=0, return_index=True)
    unique_ID_sorted = np.sort(unique_ID)   # sort to their original order
    config_list = config_list[unique_ID_sorted]   #slicing. take every 25th config
    trace_prop_arr = trace_prop_arr[unique_ID_sorted] #slicing. take every 25th obj value

    """build metrics"""
    
    SCP_list = []     # config_coverage_plant_in_res
    SCCTP_list = [] #site_coverage_cell_topN_pop
    SCCTI_list = []  #site_coverage_cell_topN_infect
    RMS_A2S_list = []     # RMS(shortest distance from allowed sites to config)
    RMS_A2S_wtd_list = []     #RMS weighted by #plants. 
    PDSN_list = []  # closeness of the config, normalised by the closeness of the allowed sites
    for i, config in enumerate(config_list):
        host_pop_config=host_distr['host_population'][config]

        ### site_coverage_plant: for each config, what is the #plant(config)/#plant(allowed) ##################
        site_coverage_plant = np.sum(host_pop_config) / np.sum(host_pop_allowed)
        SCP_list.append(site_coverage_plant) 


        ### % of cells from each config are from the top 10 populated cells in allowed
        site_coverage_cell_topN_pop = len(np.intersect1d(config, host_pop_allowed_topn_ID))/topn
        SCCTP_list.append(site_coverage_cell_topN_pop)

        ### % of cells from each config are from the top 10 most visited cells in allowed
        site_coverage_cell_topN_infect = len(np.intersect1d(config, visit_allowed_topn_ID))/topn
        SCCTI_list.append(site_coverage_cell_topN_infect)


        ### config_RMS_res (CRMSr): for each config, what is the spread of the config of the allowed area #################
        config_pos = host_distr['xy'][config]    #nx2 np.array
        # RMS(pairwise distances between allowed sites and config)
        cross_dist = cdist(host_pos_allowed, config_pos)
        shortest_dist = np.min(cross_dist, axis=1)
        RMS_A2S = (np.mean(shortest_dist**2))**0.5
        RMS_A2S_list.append(RMS_A2S) 

        ### config_RMS_res_wtd (CRMSr_wtd): CRMSr weighted by #plants
        weight_plant = host_pop_allowed/np.sum(host_pop_allowed)
        RMS_A2S_wtd = (np.mean((shortest_dist**2) * weight_plant))**0.5
        RMS_A2S_wtd_list.append(RMS_A2S_wtd)

        ### Config_CLoseness_res (PDSN): for each config, what is the closeness of the config, normalised by the closeness of the allowed area ################
        pairwisedist_config_sum = np.sum(pdist(config_pos))
        PDSN = pairwisedist_config_sum / pairwisedist_allowed_sum
        # print(pairwisedist_allowed_sum, pairwisedist_config_sum, PDSN)
        PDSN_list.append(PDSN)

        

    SCP_arr = np.array(SCP_list)
    # CCPtopnr_arr = np.array(CCPtopnr_list)
    SCCTP_arr = np.array(SCCTP_list)
    SCCTI_arr = np.array(SCCTI_list)
    RMS_A2S_arr = np.array(RMS_A2S_list)
    RMS_A2S_wtd_arr = np.array(RMS_A2S_wtd_list)
    PDSN_arr = np.array(PDSN_list)
        
    dcor_SCP = dcor.distance_correlation(trace_prop_arr, SCP_arr)
    dcor_SCCTP = dcor.distance_correlation(trace_prop_arr, SCCTP_arr)
    dcor_SCCTI = dcor.distance_correlation(trace_prop_arr, SCCTI_arr)
    dcor_RMS_A2S = dcor.distance_correlation(trace_prop_arr, RMS_A2S_arr)
    dcor_RMS_A2S_wtd = dcor.distance_correlation(trace_prop_arr, RMS_A2S_wtd_arr)
    dcor_PDSN = dcor.distance_correlation(trace_prop_arr, PDSN_arr)
    
    # return trace_prop_arr, SCP_arr, CCPtopnr_arr, SCCTP_arr, CRMSr_arr,  dcor_SCP, dcor_CCPtopnr, dcor_SCCTP, dcor_CRMSr
    return trace_prop_arr,  SCP_arr, SCCTP_arr, SCCTI_arr, RMS_A2S_arr, RMS_A2S_wtd_arr, PDSN_arr,  dcor_SCP, dcor_SCCTP, dcor_SCCTI, dcor_RMS_A2S, dcor_RMS_A2S_wtd, dcor_PDSN


object_prop_list = []
SCP_list = []   
SCCTP_list = []
SCCTI_list = []
RMS_A2S_list = []
RMS_A2S_wtd_list = []
PDSN_list = []

dcor_SCP_list = []
dcor_SCCTP_list = []
dcor_SCCTI_list = []
dcor_RMS_A2S_list = []
dcor_RMS_A2S_wtd_list = []
dcor_PDSN_list = []

for i in tqdm(range(len(area_road_perm))):
    host_distr_i = int(area_road_perm[i,0]) - 1  #take the last string of the area name, convert to integer, minus 1
    object_prop, SCP, SCCTP, SCCTI, RMS_A2S, RMS_A2S_wtd, PDSN, \
    dcor_SCP, dcor_SCCTP, dcor_SCCTI, dcor_RMS_A2S, dcor_RMS_A2S_wtd, dcor_PDSN = site_metric(area=area_road_perm[i,0], 
                                                                  road=area_road_perm[i,1], 
                                                                  dir_config=dir_config, 
                                                                  dir_trace=dir_trace, 
                                                                  road_metric_df=metric_df, 
                                                                  host_distr=distr_list[host_distr_i],
                                                                  topn=NSITES)
    object_prop_list.append(object_prop)

    SCP_list.append(SCP)
    SCCTP_list.append(SCCTP)
    SCCTI_list.append(SCCTI)
    RMS_A2S_list.append(RMS_A2S)
    RMS_A2S_wtd_list.append(RMS_A2S_wtd)
    PDSN_list.append(PDSN)

    dcor_SCP_list.append(dcor_SCP)
    dcor_SCCTP_list.append(dcor_SCCTP)
    dcor_SCCTI_list.append(dcor_SCCTI)
    dcor_RMS_A2S_list.append(dcor_RMS_A2S)
    dcor_RMS_A2S_wtd_list.append(dcor_RMS_A2S_wtd)
    dcor_PDSN_list.append(dcor_PDSN)



dump(object_prop_list, os.path.join(dir_data_output, f'{filename_save}_object_prop_list.joblib'))
dump(SCP_list, os.path.join(dir_data_output, f'{filename_save}_SCP_list.joblib'))
dump(SCCTP_list, os.path.join(dir_data_output, f'{filename_save}_SCCTP_list.joblib'))
dump(SCCTI_list, os.path.join(dir_data_output, f'{filename_save}_SCCTI_list.joblib'))
dump(RMS_A2S_list, os.path.join(dir_data_output, f'{filename_save}_RMS_A2S_list.joblib'))
dump(RMS_A2S_wtd_list, os.path.join(dir_data_output, f'{filename_save}_RMS_A2S_wtd_list.joblib'))
dump(PDSN_list, os.path.join(dir_data_output, f'{filename_save}_PDSN_list.joblib'))

dump(dcor_SCP_list, os.path.join(dir_data_output, f'{filename_save}_dcor_SCP_list.joblib'))
dump(dcor_SCCTP_list, os.path.join(dir_data_output, f'{filename_save}_dcor_SCCTP_list.joblib'))
dump(dcor_SCCTI_list, os.path.join(dir_data_output, f'{filename_save}_dcor_SCCTI_list.joblib'))
dump(dcor_RMS_A2S_list, os.path.join(dir_data_output, f'{filename_save}_dcor_RMS_A2S_list.joblib'))
dump(dcor_RMS_A2S_wtd_list, os.path.join(dir_data_output, f'{filename_save}_dcor_RMS_A2S_wtd_list.joblib'))
dump(dcor_PDSN_list, os.path.join(dir_data_output, f'{filename_save}_dcor_PDSN_list.joblib'))







