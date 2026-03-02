"""
OPTIMISATION MODULE (NIGERIA COMPARISON)
=======================================

Purpose
-------
This module implements surveillance-site optimisation routines used in the
Nigeria comparison analysis (Ferris et al. 2024 context). It is a Nigeria-specific
variant of the core optimisation module used elsewhere in this repository.

Core functionality
------------------
Provides functions to:
- read road network shapefiles (if applicable)
- compute which host grid cells are accessible from roads (candidate surveillance sites)
- compute objective values for detection probability under a given configuration
- run simulated annealing to optimise surveillance site placement
- compute road and optimisation metrics used in analysis (e.g. RCP, CPA/RCAP)

Nigeria-specific change
-----------------------
Adds checkpointing support to simulated_annealing(), allowing long optimisation
runs to periodically save progress and be resumed if interrupted.

Notes
-----
This module is used by the Nigeria comparison driver scripts in:
  main_analysis/code/nigeria_comparison/code/
"""

import os
#os.environ["OMP_NUM_THREADS"] = "1"  # limit each process to 1 thread
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
import sys
module_dir = '/home/physics/estszj/research_project'
sys.path.append(module_dir)

from simulation import vlogistic
from CONSTANT import ALPHA, BETA, SIGMA0, LOGISTIC_RATE, PREVALENCE, P_DETECT, NTREES_SURVEY, SURVEY_FREQ
import numpy as np
import datetime
from joblib import dump, load
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.spatial.distance import cdist
import signal


### Restrict where to survey ===============================================
def site_loc_allowed(roadnetwork, host_pos, host_pop, accessible_dist): 
    """
    Generate site locations from the host distribution with road network restrictions.
    Input: 
    - roadnetwork: a shape file.
    - host_pos (array of 2D x-y coordinates)
    - accessible_dist: straight-line distance from the road network that's accessible for surveying
    Output: site_locations, cells_near_road_ID
    """
    ### Allowed site locations are those within the road network.
    if roadnetwork is None:
        site_loc_allowed_ID = np.where(host_pop >0)[0]   # accept all host positions with >0 plants #===
        #cells_near_road_ID = np.arange(2500)    #+++
        cells_near_road_ID = np.arange(len(host_pop))
    else:
        ### convert geo df (shapefile) road network to a shapely object
        roadnetwork_shapely = roadnetwork.geometry.tolist()
        roadnetwork_shapely = unary_union(roadnetwork_shapely)

        points = [Point(x,y) for x,y in zip(host_pos[:, 0], host_pos[:, 1])]
        # print(points)
        distances = roadnetwork_shapely.distance(points)    # compute distance of all host points to the nearest road
        # print(distances)
        site_loc_allowed_ID = np.where((host_pop >0)&(distances <= accessible_dist))[0]   # accept host positions within accessible distance    #===
        cells_near_road_ID = np.where(distances <= 1000)[0]   # accept host positions within accessible distance    #+++
        # print(len(site_loc_allowed_ID))
    return site_loc_allowed_ID,cells_near_road_ID   #===
    

### logistic growth used for optimisation (much faster than logistic()) ===============================================
def logistic_forOpt(time, time_eventS2C, time_eventC2I, host_population, sigma0, c):
    """
    time: current times (or survey times) (1 x num(surveys) array)
    time_eventS2C: time when the point transitioned from S to C (1x #points array) (time_eventS2C must be np.inf if event hasn't occured)
    time_eventC2I: time when the point transitioned from C to I (1x #points array) (time_eventC2I must be np.inf if event hasn't occured)
    host_population: number of trees in the point (1x #points array)
    sigma0: proportion of infected at time of infection (constant)
    c: growth rate parameter for the logistic model (constant)
    Note: len(time_eventS2C)=len(time_eventC2I)=len(host_population)

    Output: (a tuple with:
                num(output) * len(time) * len(test_timeS2C or test_timeC2I, i.e. number of points))
    """
    # host_prop_inC =  np.array([1/ (1 + (1/sigma0 -1) * np.exp(-c * (t - time_eventS2C))) for t in time])
    # print(host_prop_inC)
    # host_num_inC = host_population * host_prop_inC
    # host_prop_inI = np.array([1/ (1 + (1/sigma0 -1) * np.exp(-c * (t - time_eventC2I))) for t in time])
    # host_num_inI = host_population * host_prop_inI

    t_diff_S2C = time[:, np.newaxis]-time_eventS2C[np.newaxis, :]
    t_diff_S2C[t_diff_S2C<0]= -np.inf
    # print(t_diff_S2C<0, t_diff_S2C[t_diff_S2C<0])
    host_prop_inC =  1/ (1 + (1/sigma0 -1) * np.exp(-c * t_diff_S2C))
    # print(t_diff_S2C, '\n',host_prop_inC)
    host_num_inC = host_population * host_prop_inC

    t_diff_C2I = time[:, np.newaxis]-time_eventC2I[np.newaxis, :]
    t_diff_C2I[t_diff_C2I<0]= -np.inf
    host_prop_inI = 1/ (1 + (1/sigma0 -1) * np.exp(-c * t_diff_C2I))
    host_num_inI = host_population * host_prop_inI

    ### update # and % of trees in C state
    host_prop_inC = host_prop_inC - host_prop_inI
    host_num_inC = host_num_inC - host_num_inI

    return host_num_inC, host_prop_inC, host_num_inI, host_prop_inI

### Objective function (optimised as it takes logistic_forOpt() instead of logistic(), and less loops)===============================================
def objective_optimised(data_sim,  
              configID, # prop_detectable, #t_star,
              ntrees_survey, P_detect, survey_freq):
    """
    The evaluation of the objective function using the config.
    Objective: maximise P(detection before prevalence) === minimise 1-P(detection before prevalence)
    Input: 
        - configID (site locations): a list of ID (of x-y coordinates) for surveillance sites.
        - prop_detectable: % of hosts detectable in the config. a list of constants, one for each site in config.
        - ntrees_survey: max number of trees to survey per site.
        - P_detect (probability of successful detection): a constant.
        - survey_freq (dt, time step): a constant, default is 1. meaning surveying weekly, as 1 = 1 week.
        # - t_star: time at prevalence. A constant. unit: week
    Output: objective_val
    """
    num_sim = len(data_sim)     # number of simulations created for pathogen spread realisations
    P_arr = np.zeros(num_sim)  # P(fail to detect across all survey times in the M-th simulation)

    # time at which prevalence is reached across all simulations
    t_star_simsarr= np.array([np.max(np.concatenate([sim['time_1st_S2C'][sim['time_1st_S2C']!=np.inf],
                             sim['time_1st_C2I'][sim['time_1st_C2I']!=np.inf]])) for sim in data_sim])  # (1 x num(sims))
    
    ### survey times are 0, dt, 2dt, ..., ndt such that ndt < t_star
    survey_times_simsarr = [np.arange(0, t_star, survey_freq) for t_star in t_star_simsarr]     # a list of num(sims) entries, each is an array of varying length

    host_population = data_sim[0]['host_population'][configID]   # same across sims so don't need for loop
    
    time_1st_S2C_ofConfig_sims = [sim['time_1st_S2C'][configID] for sim in data_sim]
    time_1st_C2I_ofConfig_sims = [sim['time_1st_C2I'][configID] for sim in data_sim]

    for s in range(num_sim):
        # print(survey_times_sims[s], '\t',time_1st_S2C_ofConfig_sims[s])
        host_num_inC, host_prop_inC, host_num_inI, host_prop_inI = logistic_forOpt(time=survey_times_simsarr[s], 
                                                                                   time_eventS2C=time_1st_S2C_ofConfig_sims[s], 
                                                                                   time_eventC2I=time_1st_C2I_ofConfig_sims[s], 
                                                                                   host_population=host_population, 
                                                                                   sigma0=SIGMA0, 
                                                                                   c=LOGISTIC_RATE) #this gives 4 arrays, each have dimension len(time) * num(cells)
        prop_detectable = host_prop_inC + host_prop_inI         # proportion of hosts detectable in the config at time t
        mtrees_survey = np.clip(host_population, a_min=1, a_max=ntrees_survey)     # number of trees to survey across all sites in config
        f2 = (1- P_detect * prop_detectable)**mtrees_survey     # P(fail to detect disease from each site)
        G = np.prod(f2)             # P(fail to detect across all survey sites across all survey times in the current simulation s)
        P = 1 - G                   # P(detect disease at any survey time in the current simulation s)
        P_arr[s] = P

    objective_val = np.mean(P_arr)  # average P to obtain the objective (P(successfully detect))
    # print('P=', P_arr)
    return objective_val
    
    
### Objective function (MORE optimised as if the config doesn't overlap with any infected cells then P of detecting within the simulation is 0)===============================================
def objective_more_optimised(data_sim,  
              configID, # prop_detectable, #t_star,
              ntrees_survey, P_detect, survey_freq):
    """
    The evaluation of the objective function using the config.
    Objective: maximise P(detection before prevalence) === minimise 1-P(detection before prevalence)
    Input: 
        - configID (site locations): a list of ID (of x-y coordinates) for surveillance sites.
        - prop_detectable: % of hosts detectable in the config. a list of constants, one for each site in config.
        - ntrees_survey: max number of trees to survey per site.
        - P_detect (probability of successful detection): a constant.
        - survey_freq (dt, time step): a constant, default is 1. meaning surveying weekly, as 1 = 1 week.
        # - t_star: time at prevalence. A constant. unit: week
    Output: objective_val
    """
    num_sim = len(data_sim)     # number of simulations created for pathogen spread realisations
    P_arr = np.zeros(num_sim)  # P(fail to detect across all survey times in the M-th simulation)

    # time at which prevalence is reached across all simulations
    t_star_simsarr= np.array([np.max(np.concatenate([sim['time_1st_S2C'][sim['time_1st_S2C']!=np.inf],
                             sim['time_1st_C2I'][sim['time_1st_C2I']!=np.inf]])) for sim in data_sim])  # (1 x num(sims))
    
    survey_times_simsarr = [np.arange(0, t_star, survey_freq) for t_star in t_star_simsarr]     # a list of num(sims) entries, each is an array of varying length

    host_population = data_sim[0]['host_population'][configID]   # same across sims so don't need for loop
    
    time_1st_S2C_ofConfig_sims = [sim['time_1st_S2C'][configID] for sim in data_sim]
    time_1st_C2I_ofConfig_sims = [sim['time_1st_C2I'][configID] for sim in data_sim]

    for s in range(num_sim):
        # print(survey_times_simsarr[s], '\t',time_1st_S2C_ofConfig_sims[s])  #COMMENT OUT LATER
        ID_infected_in_sim=np.where(data_sim[s]['state']!='S')[0]
        if np.intersect1d(configID, ID_infected_in_sim).size==0:
            P=0
        else:
            host_num_inC, host_prop_inC, host_num_inI, host_prop_inI = logistic_forOpt(time=survey_times_simsarr[s], 
                                                                                    time_eventS2C=time_1st_S2C_ofConfig_sims[s], 
                                                                                    time_eventC2I=time_1st_C2I_ofConfig_sims[s], 
                                                                                    host_population=host_population, 
                                                                                    sigma0=SIGMA0, 
                                                                                    c=LOGISTIC_RATE) #this gives 4 arrays, each have dimension len(time) * num(cells)
            prop_detectable = host_prop_inC + host_prop_inI         # proportion of hosts detectable in the config at time t
            mtrees_survey = np.clip(host_population, a_min=1, a_max=ntrees_survey)     # number of trees to survey across all sites in config
            f2 = (1- P_detect * prop_detectable)**mtrees_survey     # P(fail to detect disease from each site)
            G = np.prod(f2)             # P(fail to detect across all survey sites across all survey times in the current simulation s)
            P = 1 - G                   # P(detect disease at any survey time in the current simulation s)
        P_arr[s] = P

    objective_val = np.mean(P_arr)  # average P to obtain the objective (P(successfully detect))
    # print('objective:', objective_val)
    # print('P=', P_arr)
    return objective_val





### SIMULATED ANNEALING ==========================================================================
def simulated_annealing(data_sim, obj_fun, site_loc_allowed, config0, 
                        n_iterations, init_temperature, cooling_rate,
                        ntrees_survey, P_detect, survey_freq, 
                        lastN=5000, earlystop_tol=0.001,
                        checkpoint_path=None,
                        checkpoint_every=1000,
                        resume=True,
                        ):
                        
    """
    Simulated annealing optimiser for surveillance site placement.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path, optional
        If provided, optimisation progress is periodically saved to this location.
        This allows runs to be resumed after interruption (e.g., cluster timeouts).

    checkpoint_every : int, optional
        Frequency (in iterations) at which to write checkpoints.

    Returns
    -------
    best_config, config_list, temp_list, objVal_list, n_iter_actual
        Same return objects as the core optimisation workflow.
    """
     
    def _save_ckpt(i, config_curr, objVal_curr, temp_curr, config_list, temp_list, objVal_list):
        if not checkpoint_path:
            return
        ckpt = {
            "i": int(i),
            "config_curr": np.asarray(config_curr),
            "objVal_curr": float(objVal_curr),
            "temp_curr": float(temp_curr),
            "config_list": config_list,
            "temp_list": temp_list,
            "objVal_list": objVal_list,
            "rng_state": np.random.get_state(),
            "saved_at": datetime.datetime.now().isoformat(),
        }
        tmp_path = checkpoint_path + ".tmp"
        dump(ckpt, tmp_path)
        os.replace(tmp_path, checkpoint_path)
        
    # ---- Resume if checkpoint exists ----
    start_iter = 0
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        ckpt = load(checkpoint_path)
        start_iter = int(ckpt["i"]) + 1
        config_curr = np.asarray(ckpt["config_curr"])
        objVal_curr = float(ckpt["objVal_curr"])
        temp_curr = float(ckpt["temp_curr"])
        config_list = ckpt.get("config_list", [config_curr])
        temp_list = ckpt.get("temp_list", [temp_curr])
        objVal_list = ckpt.get("objVal_list", [objVal_curr])
        rng_state = ckpt.get("rng_state", None)
        if rng_state is not None:
            np.random.set_state(rng_state)
        print(
            f"[resume] Loaded checkpoint {checkpoint_path}. "
            f"Continuing at iteration {start_iter}/{n_iterations}.",
            flush=True,
        )
    else:
        # ---- Fresh start ----
        objVal0 = obj_fun(data_sim, config0, ntrees_survey, P_detect, survey_freq)
        config_curr = np.asarray(config0)
        objVal_curr = objVal0
        temp_curr = init_temperature
        temp_list = [temp_curr]
        objVal_list = [objVal_curr]
        config_list = [config_curr]

    # ---- Handle Slurm time-limit termination (SIGTERM) ----
    def _handle_sigterm(signum, frame):
        print("[signal] SIGTERM received; saving checkpoint and exiting.", flush=True)
        _save_ckpt(start_iter - 1, config_curr, objVal_curr, temp_curr, config_list, temp_list, objVal_list)
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except Exception:
        pass
         
        
    num_sites = len(config0)

    for i in range(start_iter, n_iterations):
        if i%1000==0: print(
              f"Iteration {i} at time= {datetime.datetime.now()} "
              f"config={config_curr}, objective={objVal_curr:.4f}", 
              flush=True
              )
        
        ### Early stopping
        if i>=30000: 
            # calclulate the variation in the last 5000 objective values
            lastNtrial=np.array(objVal_list[-lastN:])
            maxAbsChanges=abs(np.diff(lastNtrial)).max()
            if maxAbsChanges < earlystop_tol: 
                break
        
        # ---- Propose candidate ----
        ### sites that are not chosen in the current config
        not_chosen_sites = np.setdiff1d(site_loc_allowed, config_curr)

        ### replace a site in the current config with the candidate site
        config_candidate = config_curr.copy()  # make a copy of the current config
        j = np.random.choice(len(config_curr))  # choose a random index in the current config
        config_candidate[j] = np.random.choice(not_chosen_sites)  # replace the site at index j with the candidate site

        ### accept or reject the candidate config
        objVal_candidate = obj_fun(data_sim, config_candidate, ntrees_survey, P_detect, survey_freq)            # evaluate objective of candidate
        
        # ---- Accept/reject ----
        if objVal_candidate > objVal_curr:                          #accept candidate
            config_curr, objVal_curr = config_candidate, objVal_candidate
        else:                                                       # decide to acccept or reject
            
            diff = objVal_candidate - objVal_curr                   # calculate the difference in objective values
            acceptance_rate = np.exp(diff / temp_curr)             # this is for maximisation.
            rand_number = np.random.uniform()
            # print(f"deciding: acc rate = {acceptance_rate}, rand num = {rand_number}")
            if acceptance_rate > rand_number:  # accept candidate
                # print("accept")
                config_curr, objVal_curr = config_candidate, objVal_candidate
                
        # ---- Cool + record ----
        temp_curr = temp_curr * cooling_rate                    # update temperature
        temp_list.append(temp_curr)                              # store temperature
        objVal_list.append(objVal_curr)                          # store objective value
        config_list.append(config_curr)                          # store current config
        
        
        # ---- Periodic checkpoint ----
        if checkpoint_path and (i % checkpoint_every == 0) and i != start_iter:
            _save_ckpt(i, config_curr, objVal_curr, temp_curr, config_list, temp_list, objVal_list)

    # final save
    _save_ckpt(i, config_curr, objVal_curr, temp_curr, config_list, temp_list, objVal_list)
    
    return config_curr,config_list, temp_list, objVal_list, i
    

def opt_metrics(road, host_pos, host_pop, optimal_sites):
    """
    road: gpd dataframe
    ###site_allowed: site_loc_allowed_ID
    sim0: the first simulation from the batch simulation data (data_sim[0]). a dictionary
    host_pos: position of the 2500 cells. same as the parameter used in site_loc_allowed. nx2 np.array
    host_pop: plant population of the 2500 cells. np.array
    optimal_sites: best config (ID). np.array
    """
    
    if road is None:
        road_length = np.inf
        sites_allowed_ID,cells_near_road_ID = site_loc_allowed(None, host_pos, host_pop, accessible_dist=1000)  #===
        road_coverage_area=1
        road_coverage_plant=1
        RCAP=1
        
        
    else:
        road_length = road.geometry.length.sum()
        
        sites_allowed_ID,cells_near_road_ID = site_loc_allowed(road, host_pos, host_pop, accessible_dist=1000)  #===
        
        len_cells = len(host_pop)
        len_non0 = len(host_pop[host_pop>0])
        road_coverage_area = len(sites_allowed_ID)/len_cells
        RCAP = len(sites_allowed_ID)/len_non0
        road_coverage_plant=np.sum(host_pop[sites_allowed_ID])/np.sum(host_pop)
        print(f'len_cells = {len_cells}')
        print(f'len_non0 = {len_non0}')

    ### SUM SQUARED DISTANCE (weighted by plant proportion) (from each cell with plant to all allowed or opt cells)
    weights_plant = host_pop/np.sum(host_pop)
    cross_dist = cdist(host_pos, host_pos[cells_near_road_ID])    # cross distance between each cell to all the allowed cells   #===
    shortest_dist = np.min(cross_dist, axis=1)        #shortest distance from each cell to the allowed cells

    RMS_toRoad_wtd = (np.mean((shortest_dist**2)*weights_plant))**0.5
    RMS_toRoad = (np.mean(shortest_dist**2))**0.5
    
    cross_dist = cdist(host_pos, host_pos[sites_allowed_ID])    # cross distance between each cell to all the allowed cells   #===
    shortest_dist = np.min(cross_dist, axis=1)        #shortest distance from each cell to the allowed cells
    
    RMS_toAllowed_wtd = (np.mean((shortest_dist**2)*weights_plant))**0.5
    RMS_toAllowed = (np.mean(shortest_dist**2))**0.5

    cross_dist = cdist(host_pos, host_pos[optimal_sites])    # cross distance between each cell to all the optimal cells
    shortest_dist = np.min(cross_dist, axis=1)        #shortest distance from each cell to the optimal cells
    RMS_toOpt_wtd = (np.mean((shortest_dist**2)*weights_plant))**0.5
    RMS_toOpt = (np.mean(shortest_dist**2))**0.5      # root mean squared distance to road 


    optimal_sites_pop = host_pop[optimal_sites]
    allowed_sites_pop = host_pop[sites_allowed_ID]
    reddest_pop = host_pop.max()
    num_red_optimal=np.sum(optimal_sites_pop==reddest_pop)
    num_red_allowed=np.sum(allowed_sites_pop==reddest_pop)
    num_allowed=len(sites_allowed_ID)
    prop_red_allowedred=num_red_optimal / num_red_allowed
    prop_red_allowed = num_red_optimal/ num_allowed
    
    print(f'road_coverage_area={road_coverage_area}')
    print(f'RCAP={RCAP}')
    print(f'road_coverage_plant = {road_coverage_plant}')    

    return road_length, road_coverage_area, RCAP, road_coverage_plant,RMS_toRoad_wtd,RMS_toRoad, RMS_toAllowed_wtd, RMS_toAllowed, RMS_toOpt_wtd, RMS_toOpt,num_red_optimal, num_red_allowed, prop_red_allowedred,prop_red_allowed
