"""
OPTIMISATION MODULE: Surveillance site selection
================================================

This file contains the optimisation routines used to identify surveillance
site locations that maximise the probability of detecting infection before a
specified prevalence threshold is reached.

It implements:
- Objective functions for evaluating surveillance strategies
- Accessibility constraints based on road networks
- Simulated annealing routines for searching over candidate site locations

Key functionality includes:
--------------------------
- Reading road network shapefiles
- Determining which spatial locations are allowed as surveillance sites
- Evaluating detection performance across epidemic simulations
- Optimising site placement under user-defined constraints

This module does NOT run analyses by itself.

It is imported by scripts in:
- main_analysis/code/optimisation/

Those scripts provide inputs (simulation outputs, road patterns, parameters)
and call the functions defined here to generate optimisation results.

"""

import os
#os.environ["OMP_NUM_THREADS"] = "1"  # limit each process to 1 thread
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
import sys
from pathlib import Path
# --- PATH SETUP: make local modules importable ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[0]   # optimisation.py is at repo root
sys.path.insert(0, str(PROJECT_ROOT))

# USER: if auto-detection fails, uncomment and set manually:
# PROJECT_ROOT = Path(r"/path/to/your/repository")

method = sys.argv[1]  # 'risk' or 'fix'

from simulation import vlogistic
from CONSTANT import ALPHA, BETA, SIGMA0, LOGISTIC_RATE, PREVALENCE, P_DETECT, NTREES_SURVEY, SURVEY_FREQ
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

### change: SURVEY_FREQ, number of simulations used

### Read road network ===============================================
### amuse yourself with this :)
def read_road(file_dir):
    road = gpd.read_file(file_dir)
    return road


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
        cells_near_road_ID = np.arange(2500)    #+++
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


### Objective function (TOO SLOW. DO NOT USE) ===============================================
def objective(data_sim,  
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

    # G_arr = np.zeros(num_sim)
    P_arr = np.zeros(num_sim)  # P(fail to detect across all survey times in the M-th simulation)

    for M in range(num_sim):
        sim = data_sim[M]  # get the M-th simulation
        t_star = max(sim['time_1st_S2C'][sim['time_1st_S2C']!=np.inf].max(),
                     sim['time_1st_C2I'][sim['time_1st_C2I']!=np.inf].max()) # time at which prevalence is reached, in weeks.
        # k_max = int(t_star/survey_freq)+1  # max number of surveys done before t*. Survey times are: 0, dt, 2dt, ..., (k_max-1)dt
        ### survey times are 0, dt, 2dt, ..., ndt such that ndt < t_star
        survey_times = np.arange(0, t_star, survey_freq)

        ### get host population and times at infection for the chosen survey sites
        # sim_sites = sim[configID]  # get the survey sites in the m-th simulation
        host_population = sim['host_population'][configID].copy()
        time_1st_S2C = sim['time_1st_S2C'][configID].copy()
        time_1st_C2I = sim['time_1st_C2I'][configID].copy()
        
        F2_arr = np.zeros(len(survey_times))
        for i, t in enumerate(survey_times):
            time_1st_S2C_dummy = time_1st_S2C.copy()
            time_1st_C2I_dummy = time_1st_C2I.copy()
            time_1st_S2C_dummy[time_1st_S2C> t] = np.inf
            time_1st_C2I_dummy[time_1st_C2I> t] = np.inf
            # host_num_inC, host_prop_inC, host_num_inI, host_prop_inI = vgompertz(time=t,
            #                                                                     time_eventS2C = time_1st_S2C_dummy,
            #                                                                     time_eventC2I = time_1st_C2I_dummy,
            #                                                                     host_population = host_population,
            #                                                                     b = -np.log(0.01),
            #                                                                     c = 0.0962) 
            host_num_inC, host_prop_inC, host_num_inI, host_prop_inI = vlogistic(time=t,
                                                                                time_eventS2C = time_1st_S2C_dummy,
                                                                                time_eventC2I = time_1st_C2I_dummy,
                                                                                host_population = host_population,
                                                                                sigma0 = SIGMA0,
                                                                                c = LOGISTIC_RATE) 
            prop_detectable = host_prop_inC + host_prop_inI         # proportion of hosts detectable in the config at time t
            mtrees_survey = np.clip(host_population, a_min=1, a_max=ntrees_survey)     # number of trees to survey across all sites in config
            f2 = (1- P_detect * prop_detectable)**mtrees_survey     # P(fail to detect disease from each site)
            F2 = np.prod(f2)                                        # P(fail to detect disease across all sites)
            F2_arr[i] = F2

        G = np.prod(F2_arr)        # P(fail to detect across all survey times in the M-th simulation)
        P = 1 - G                  # P(detect disease at some survey time in the M-th simulation)
        P_arr[M] = P
        # print(G)
    objective_val = np.mean(P_arr)  # average P to obtain the objective (P(successfully detect))
    
    return objective_val



### SIMULATED ANNEALING ==========================================================================
def simulated_annealing(data_sim, obj_fun, site_loc_allowed, config0, 
                        n_iterations, init_temperature, cooling_rate,
                        ntrees_survey, P_detect, survey_freq, 
                        lastN=5000, earlystop_tol=0.001
                        ):
   
    num_sites = len(config0)
    ### evaluate objective value on the initial config
    objVal0 = obj_fun(data_sim, config0, ntrees_survey, P_detect, survey_freq)

    ### Set current config and objective value
    config_curr = config0
    objVal_curr = objVal0
    temp_curr = init_temperature
    # print(f"Iteration 00, temperature: {temp_curr},  Initial Objective: {objVal_curr}, Initial Config: {config_curr}")

    temp_list = [temp_curr]  # store temperatures for each iteration
    objVal_list = [objVal_curr]  # store objective values for each iteration
    config_list = [config_curr]  # store configurations for each iteration
    
    #for i in tqdm(range(n_iterations)):
    for i in range(n_iterations):
        if i%1000==0: print(f"Iteration {i} at time=", datetime.datetime.now(), f"config={config_curr}, objective={objVal_curr:.4f}", flush=True)
        ### Early stopping
        if i>=30000: 
            # calclulate the variation in the last 5000 objective values
            lastNtrial=np.array(objVal_list[-lastN:])
            maxAbsChanges=abs(np.diff(lastNtrial)).max()
            if maxAbsChanges < earlystop_tol: break
        
        ### sites that are not chosen in the current config
        not_chosen_sites = np.setdiff1d(site_loc_allowed, config_curr)

        ### replace a site in the current config with the candidate site
        config_candidate = config_curr.copy()  # make a copy of the current config
        j = np.random.choice(len(config_curr))  # choose a random index in the current config
        config_candidate[j] = np.random.choice(not_chosen_sites)  # replace the site at index j with the candidate site

        ### accept or reject the candidate config
        objVal_candidate = obj_fun(data_sim, config_candidate, ntrees_survey, P_detect, survey_freq)            # evaluate objective of candidate
        
        if objVal_candidate > objVal_curr:                          #accept candidate
            # print("accept")
            config_curr, objVal_curr = config_candidate, objVal_candidate
        else:                                                       # decide to acccept or reject
            
            diff = objVal_candidate - objVal_curr                   # calculate the difference in objective values
            # temp_curr = temp_curr / float(i+1)                      # calculate temperature of current iteration. Is this fast annealing schedule?????????????
            # acceptance_rate = np.exp(-diff / temp_curr)             # check if this is correct!!! - this is for minimisation, but we are maximising the objective function.
            acceptance_rate = np.exp(diff / temp_curr)             # check if this is correct!!!!!!!!!!!!!!!!!!!!! - this is for maximisation.
            rand_number = np.random.uniform()
            # print(f"deciding: acc rate = {acceptance_rate}, rand num = {rand_number}")
            if acceptance_rate > rand_number:  # accept candidate
                # print("accept")
                config_curr, objVal_curr = config_candidate, objVal_candidate
        temp_curr = temp_curr * cooling_rate                    # update temperature
        temp_list.append(temp_curr)                              # store temperature
        objVal_list.append(objVal_curr)                          # store objective value
        config_list.append(config_curr)                          # store current config
        # print(f"current iter, temperature: {temp_curr}, Objective Value: {objVal_curr}, Config: {config_curr}")
    
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
        
        
    else:
        road_length = road.geometry.length.sum()
        
        sites_allowed_ID,cells_near_road_ID = site_loc_allowed(road, host_pos, host_pop, accessible_dist=1000)  #===
        
        road_coverage_area = len(sites_allowed_ID)/2500     #number of allowed covered out of all 2500 cells    #===
        
        road_coverage_plant=np.sum(host_pop[sites_allowed_ID])/np.sum(host_pop)

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


    ### #(reddest chosen)/#(all reddest in allowed)
    optimal_sites_pop = host_pop[optimal_sites]
    allowed_sites_pop = host_pop[sites_allowed_ID]
    reddest_pop = host_pop.max()
    num_red_optimal=np.sum(optimal_sites_pop==reddest_pop)
    num_red_allowed=np.sum(allowed_sites_pop==reddest_pop)
    num_allowed=len(sites_allowed_ID)
    prop_red_allowedred=num_red_optimal / num_red_allowed
    prop_red_allowed = num_red_optimal/ num_allowed

    return road_length, road_coverage_area, road_coverage_plant,RMS_toRoad_wtd,RMS_toRoad, RMS_toAllowed_wtd, RMS_toAllowed, RMS_toOpt_wtd, RMS_toOpt,num_red_optimal, num_red_allowed, prop_red_allowedred,prop_red_allowed



###############################################################################
####################### RUN OPTIMISATION ######################################
###############################################################################

# ### start time
# start_time = datetime.datetime.now()
# print('START TIME:', start_time, flush=True)

# ### Set seed to ensure reproducability
# np.random.seed(0)  

# ### Read simulation data
# sim_data = load(os.path.join(dir_home,subdir_sim,filename_sim))
# print("num of simulations:", len(sim_data), "Survey_freq:", SURVEY_FREQ, "P(detect)=", P_DETECT, flush=True)

# ### column bind the x and y coords of the points with hosts >=0 (2500 of them)
# host_xpos = sim_data[0]['x']
# host_ypos = sim_data[0]['y']
# host_positions = np.column_stack((host_xpos, host_ypos))
# host_population=sim_data[0]['host_population']

# ### Obtain road
# shp_road = read_road(file_dir= os.path.join(dir_home, subdir_road, filename_road))
# site_loc_allowed_ID = site_loc_allowed(roadnetwork=shp_road, host_pos=host_positions, accessible_dist=1000)  # 1km accessibility



# ### Plot road and allowed sites
# #fig, ax = plt.subplots(figsize = (10,6))
# #plt.scatter(host_positions[:, 0] ,host_positions[:, 1], marker='s', s=30, linewidths=0, edgecolors='none', alpha=0.5, c=host_population, cmap='RdYlGn_r')
# #plt.scatter(host_positions[site_loc_allowed_ID, 0], host_positions[site_loc_allowed_ID, 1], marker='s', s=30, facecolors='none', edgecolors='black', label='sites allowed for survey')
# #shp_road.plot(ax=ax, color='red', linewidth=1, label='road')
# #plt.xlabel('x')
# #plt.ylabel('y')
# #plt.legend()
# #plt.savefig(os.path.join(save_dir, 'plot_hostWithRoadRes.pdf'), dpi=600)
# #plt.clf # clear figure
# #plt.close(fig)


# ### initial config (location of the surveillance sites). Set randomly (within road constraints) ================================
# num_sites = 5  # number of sites to choose for surveillance
# config0 = np.random.choice(site_loc_allowed_ID, size=min(len(site_loc_allowed_ID), num_sites), replace=False)  # choose sites randomly from the allowed locations
# print('initial config:', config0, flush=True)
# ### check if len(config0) < len(site_loc_allowed_ID)
# print("Check validity of initial config0:", len(config0) < len(site_loc_allowed_ID), flush=True)

# ### Optimisation (SA) ===========================================================================================================
# n_iter = 50000 # 
# temperature = 10  # find how to set initial temperature!!!!!!!!!!
# cooling_rate = 0.9995 # # find how to set cooling rate!!!!!!!!!!
# print(f'number of SA trials: {n_iter}, \tstarting temp:{temperature}, \tcooling rate:{cooling_rate}')

# best_config, temp_list, objVal_list = simulated_annealing(data_sim=sim_data, obj_fun=objective_optimised, 
#                                                         site_loc_allowed=site_loc_allowed_ID, config0=config0,
#                                                         n_iterations=n_iter, init_temperature=temperature, cooling_rate=cooling_rate,
#                                                         ntrees_survey=NTREES_SURVEY,P_detect=P_DETECT, survey_freq=SURVEY_FREQ) 


# print("best config:",  best_config, flush=True)

# ### Plot trace of objective value
# fig, ax = plt.subplots(figsize = (10,6))
# plt.plot(objVal_list , label = '')
# #plt.xscale('log', base = 10)
# plt.xlabel('Trial')
# plt.ylabel('Objective value')
# plt.title(f'Trace of SA ({n_iter} trials, start temp={temperature}, cooling rate={cooling_rate})')
# # plt.legend()
# plt.savefig(os.path.join(save_dir, f'plot_optimisationTrace_{method}_Pdetect{P_DETECT}.pdf'), dpi=600)
# plt.clf # clear figure
# plt.close(fig)

# ### Plot optimisation result
# fig, ax = plt.subplots(figsize = (10,6))
# ax.scatter(host_positions[:, 0] ,host_positions[:, 1],  marker='s', s=30, linewidths=0, edgecolors='none', alpha=0.7, c=host_population, cmap='RdYlGn_r')
# shp_road.plot(ax=ax, color='black', linewidth=1, label='road', alpha=0.5)
# ax.scatter(host_positions[site_loc_allowed_ID, 0], host_positions[site_loc_allowed_ID, 1], marker='s', s=30, facecolors='none', edgecolors='black', label='sites allowed for survey', alpha=0.6)
# ax.scatter(host_positions[best_config, 0], host_positions[best_config, 1], marker='x', s=20, c='black', label='optimal sites', alpha=0.6)
# plt.xlabel('x')
# plt.ylabel('y')
# #plt.title('')
# plt.legend()
# plt.savefig(os.path.join(save_dir, f'plot_OptResult_{method}_Pdetect{P_DETECT}.pdf'), dpi=600)
# plt.clf # clear figure
# plt.close(fig)

# end_time = datetime.datetime.now()
# print("End time:", end_time, flush=True)
# print("Time taken for simulation:", end_time - start_time, flush=True)