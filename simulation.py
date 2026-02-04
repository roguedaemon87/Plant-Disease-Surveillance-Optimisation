"""
SIMULATION MODULE: Epidemic spread model
=======================================

This file defines the core epidemic spread model used throughout the project.

It contains functions that:
- Construct host distributions on a spatial grid
- Simulate stochastic pathogen spread across the landscape
- Track transitions between epidemiological states (e.g., susceptible,
  exposed, infectious)
- Return epidemic timing metrics used in both calibration and optimisation

Key functions:
--------------
- host_distribution_casava(...)
    Builds a spatial host distribution from GIS inputs or precomputed grids.

- simulate_ibm(...)
    Runs an individual-based stochastic epidemic simulation on a host
    landscape, returning infection timing and prevalence information.

This module does NOT run analyses by itself.

Instead, it is imported by:
- beta_parameterisation scripts (for calibrating transmission rate beta)
- main_analysis scripts (for generating epidemic simulations used in optimisation)

All scientific model behaviour is defined here. Other scripts simply call
these functions with different parameters or inputs.

"""

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
import os
from CONSTANT import ALPHA, BETA, SIGMA0, LOGISTIC_RATE, PREVALENCE, SURVEY_FREQ, P_DETECT, NTREES_SURVEY, PREV_FINAL


#print('cd:', os.getcwd())
#print('got here1')
def host_distribution(xmax, ymax,
        num_points_x, num_points_y,
        num_study_points,
        host_min, host_max,
        hetero_points=False):
    """ 
    Generate the host distribution
    Inputs:
        - xmax, ymax: max width & length of the landscape
        - num_points_x, num_points_y: number of points per axis
        - num_study_points: number of points selected to place trees
        - host_min: minimum of 0 tree.
        - host_max: maximum number of trees in a point.
    """
    ### create gridded landscape
    x_coords = np.linspace(0, xmax, num_points_x)
    y_coords = np.linspace(0, ymax, num_points_y)
    xy = np.array(np.meshgrid(x_coords, y_coords)).reshape(2, -1).T # gives (num_points_x*num_points_y)x2 array, eacy row is the xy coord of a point
    
    ### determine the locations of the trees in the landscape
    indices_host_loc = np.random.choice(xy.shape[0], size=num_study_points, replace=False)
    xy_host = xy[indices_host_loc]

    ### determine in each location (point), the number of trees
    host_population = np.random.choice(range(host_min, host_max+1), size=num_study_points, replace=True)

    host_data = {
        "xy": xy_host,
        # "x": xy_host[:,0],
        # "y": xy_host[:,1],
        "host_population": host_population,
        "state": ["S"]*num_study_points,
        "rates_S2C": [0.0]*num_study_points,
        "rates_C2I": [0.0]*num_study_points,
        "time_1st_S2C": [np.inf]*num_study_points,
        "time_1st_C2I": [np.inf]*num_study_points,
        "num_inC": [0.0]*num_study_points,
        "prop_inC": [0.0]*num_study_points,
        "num_inI": [0.0]*num_study_points,
        "prop_inI": [0.0]*num_study_points,
        "num_infected": [0.0]*num_study_points,
        "prop_infected": [0.0]*num_study_points
    }

    return host_data

def host_distribution_casava(road, area, x_bound, y_bound):
    """
    Calculate the distribution of production along the road.
    
    Parameters:
    road (GeoDataFrame): GeoDataFrame containing road geometries.
    production (GeoDataFrame): GeoDataFrame containing production geometries.
    x_bound (list): List containing the x-coordinate bounds [min_x, max_x].
    y_bound (list): List containing the y-coordinate bounds [min_y, max_y].
    
    Returns:
    host_data: host distribution (dictionary)
    road_sub: road network (shapefile)
    """

    bbox = box(x_bound[0], y_bound[0], x_bound[1], y_bound[1])
    road_sub = road[road.geometry.intersects(bbox)].copy()
    road_sub['geometry'] = road_sub['geometry'].apply(lambda geom: geom.intersection(bbox))
    road_sub['geometry'] = road_sub['geometry'].apply(lambda geom: translate(geom, xoff= -x_bound[0], yoff= -y_bound[0]))

    area = area[area.layer > 0]  # filter out zero values
    # area = area.reset_index(drop=True)
    area_sub = area[area.geometry.intersects(bbox)].copy()
    area_sub = area_sub.reset_index(drop=True)
    area_sub['geometry'] = area_sub['geometry'].apply(lambda geom: translate(geom, xoff= -x_bound[0], yoff= -y_bound[0]))

    centers = area_sub.geometry.centroid
    centers_arr = np.array([[points.x, points.y] for points in centers])
    prod_vals = area_sub['layer'].values
    
    # print(centers_arr.shape, prod_vals.shape)


    PROD_PER_HA = 10 # tonnes per hectare
    FIELD_SIZE = 1  # in hectares
    NUM_PLANTS_PER_HA = 10000  # number of plants per hectare (assume 1 plant per 1 m^2)
    NUM_PLANTS_PER_FIELD = 10000  # number of plants per field
    num_plants_per_cell = (prod_vals/ PROD_PER_HA * NUM_PLANTS_PER_HA).round(0).astype(int)
    # area_per_cell = np.array(area_sub.geometry.area)
    area_per_cell= 1000000  #m^2
    #prop_plants_per_cell= num_plants_per_cell/area_per_cell # proportion of the cell area is occupied by plants
    


    ### clean cells
    grid_x, grid_y = np.linspace(0, 50000, 51), np.linspace(0, 50000, 51) # this gives the ticks, including the ends
    id_x,id_y = np.digitize(centers_arr[:,0], grid_x)-1,np.digitize(centers_arr[:,1], grid_y)-1 #place each centre in relevant grid
    id_valid=(id_x>=0)&(id_x<50)&(id_y>=0)&(id_y<50)
    num_plants1 = np.zeros((50,50))
    for id_x, id_y, count in zip(id_x[id_valid], id_y[id_valid], num_plants_per_cell[id_valid]):
        num_plants1[id_y, id_x] += count
    num_plants = np.clip(num_plants1.flatten(),0,1000000).copy()
    prop_plant = num_plants/area_per_cell
    num_study_points = len(num_plants)
    xy_host = np.array(np.meshgrid(grid_x[:-1]+50000/50/2,grid_y[:-1]+50000/50/2)).reshape(2,-1).T


        
    host_data = {
        "xy": xy_host,
        # "x": xy_host[:,0],
        # "y": xy_host[:,1],
        "host_population": num_plants,
        'prop_plant': prop_plant,
        "state": ["S"]*num_study_points,
        "rates_S2C": [0.0]*num_study_points,
        "rates_C2I": [0.0]*num_study_points,
        "time_1st_S2C": [np.inf]*num_study_points,
        "time_1st_C2I": [np.inf]*num_study_points,
        "num_inC": [0.0]*num_study_points,
        "prop_inC": [0.0]*num_study_points,
        "num_inI": [0.0]*num_study_points,
        "prop_inI": [0.0]*num_study_points,
        "num_infected": [0.0]*num_study_points,
        "prop_infected": [0.0]*num_study_points
    }


    return host_data, road_sub


#  ================== Gompertz function ==================
def gompertz(time, time_eventS2C, time_eventC2I, host_population, b, c):
    """
    time: current time
    time_eventS2C: time when the point transitioned from S to C (1x #points array) (time_eventS2C must be np.inf if event hasn't occured)
    time_eventC2I: time when the point transitioned from C to I (1x #points array) (time_eventC2I must be np.inf if event hasn't occured)
    host_population: number of trees in the point (1x #points array)
    b: parameter for the Gompertz model = -ln(0.01) (constant)
    c: growth rate parameter for the Gompertz model (constant)
    """
    host_prop_inC = np.exp(-b * np.exp(-c * (time - time_eventS2C)))
    host_num_inC = host_population * host_prop_inC
    host_prop_inI = np.exp(-b * np.exp(-c * (time - time_eventC2I)))
    host_num_inI = host_population * host_prop_inI

    ### update # and % of trees in C state
    host_prop_inC = host_prop_inC - host_prop_inI
    host_num_inC = host_num_inC - host_num_inI

    return host_num_inC, host_prop_inC, host_num_inI, host_prop_inI

vgompertz = np.vectorize(gompertz, excluded=['time', 'b', 'c'])


#  ================== Logistic growth function ==================
def logistic(time, time_eventS2C, time_eventC2I, host_population, sigma0, c):
    """
    time: current time
    time_eventS2C: time when the point transitioned from S to C (1x #points array) (time_eventS2C must be np.inf if event hasn't occured)
    time_eventC2I: time when the point transitioned from C to I (1x #points array) (time_eventC2I must be np.inf if event hasn't occured)
    host_population: number of trees in the point (1x #points array)
    sigma0: proportion of infected at time of infection (constant)
    c: growth rate parameter for the logistic model (constant)
    """
    host_prop_inC =  1/ (1 + (1/sigma0 -1) * np.exp(-c * (time - time_eventS2C)))
    host_num_inC = host_population * host_prop_inC
    host_prop_inI = 1/ (1 + (1/sigma0 -1) * np.exp(-c * (time - time_eventC2I)))
    host_num_inI = host_population * host_prop_inI

    ### update # and % of trees in C state
    host_prop_inC = host_prop_inC - host_prop_inI
    host_num_inC = host_num_inC - host_num_inI

    return host_num_inC, host_prop_inC, host_num_inI, host_prop_inI

vlogistic = np.vectorize(logistic, excluded=['time', 'sigma0', 'c'])


### find logistic growth rate given initial, end proportion, and time duration
def logistic_rate(init_prop, final_prop, time_in_weeks):
    rate=np.log( (1/final_prop-1)/ (1/init_prop-1))/ (-time_in_weeks)
    return rate
    
### find time in weeks it takes to grow given initial, end proportion, and logistic rate
def logistic_time(init_prop, final_prop, rate):
    time_in_weeks=np.log( (1/final_prop-1)/ (1/init_prop-1))/ (-rate)
    return time_in_weeks






# ==================== Pathogen spread simulation (power law kernel, no Gielliespie, entry method)====================
def simulate_ibm(host_distr,
                #survey_freq, # =1 means survey once per week.
                alpha, beta, #gamma, 
                logistic_rate,
                prev_threshold,
                entry_method='risk',
                num_init_S2C=1,
                init_point_ID=1700,
                seed=0
                ):
    """ 
    Simulate the spread of disease. Always start with t=0.
    Inputs:
        - host_distr,
        - num_init_S2C,
        - alpha: parameter in the kernel
        - beta: used to calculate the rate of transition from state S to C.
        - gamma: rate of transition from state C to I.
        - logistic_rate: rate of transition from state S to C.
        - entry_method: 'risk' (propability based on population size), 'uniform' or 'fixed'
    """

    ### set the random seed
    np.random.seed(seed)  # for reproducibility

    ### Unpack host_distr
    xy = np.array(host_distr['xy'])
    host_population = np.array(host_distr['host_population'])
    prop_plant = np.array(host_distr['prop_plant'])
    state = np.array(host_distr['state'])
    rates_S2C = np.array(host_distr['rates_S2C'])
    rates_C2I = np.array(host_distr['rates_C2I'])
    time_1st_S2C = np.array(host_distr['time_1st_S2C'])
    time_1st_C2I = np.array(host_distr['time_1st_C2I'])
    prop_inC = np.array(host_distr['prop_inC'])
    num_inC = np.array(host_distr['num_inC'])
    prop_inI = np.array(host_distr['prop_inI'])
    num_inI = np.array(host_distr['num_inI'])
    prop_infected = np.array(host_distr['prop_infected'])
    num_infected = np.array(host_distr['num_infected'])

    symptom_low = 26/7 #weeks= 26 days
    symptom_high = 60/7 #weeks = 60 days

    ID_nonzero_points = np.where(host_population > 0)[0]  # get indices of host points with >0 trees (same as above, but more efficient)
    prop_infected_init = SIGMA0                         # when a point gets infected, assume 0.05% of the trees in the point are infected.

    ### Kernel
    distance_between_points = squareform(pdist(xy))   # pairwise distance between points
    distance_between_points_adj = distance_between_points.copy()
    np.fill_diagonal(distance_between_points_adj, np.inf)  # Set diag to inf so that inf**(-alpha) = 0
    A = 1/ np.sum(np.triu(distance_between_points_adj**(-alpha), k=1))  # A is a constant to ensure the kernel sums to 1
    kernel = A* distance_between_points_adj**(-alpha)       # power law kernel with A as a constant

    ### initialise time_1st_S2C for all cells
    time_1st_C2I_dummy_arr = np.array([np.inf]*len(xy)) 
    # print('time_1st_C2I_arr', time_1st_C2I_dummy_arr)

    ### sample (num_init_S2C number of) points that are infected (S -> C) by the pathogen entry =============================
    time_eventS2C = np.random.uniform(0, 1)       # time when first infection happens since pathogen entry at t=0. Sampled uniformly between t=0 and t=week1.
    
    ### ENTRY POINT
    if entry_method=='risk':
        ID_init_S2C = np.random.choice(ID_nonzero_points, size=num_init_S2C, p=host_population[ID_nonzero_points]/np.sum(host_population) , replace=False)  #===
    elif entry_method=='uniform':
        ID_init_S2C = np.random.choice(ID_nonzero_points, size=num_init_S2C, replace=False)     # index (indices) of first infected point(s)
    else:
        ID_init_S2C = init_point_ID
    

    state[ID_init_S2C] = 'C'                                                                # update state
    rates_S2C[ID_init_S2C] = 0
    # rates_C2I[ID_init_S2C] = gamma                                                          # rate of transition from C to I of the points in C. should this stay constant as more trees become C in the point?
    time_1st_S2C[ID_init_S2C] = time_eventS2C                                               # record time of state change S2C of the 1st infected point
    time_1st_C2I_dummy_arr[ID_init_S2C] = time_eventS2C + np.random.uniform(low=symptom_low, high=symptom_high)            # determine when the cell is detectable
    # print('time for the entry point to be detectable:', time_1st_C2I_dummy_arr[ID_init_S2C])
    
    prop_inC[ID_init_S2C] = prop_infected_init                                         # % of trees infected at time of infection
    num_inC[ID_init_S2C] = host_population[ID_init_S2C]*prop_infected_init             # number of trees infected at time of infection
    prop_infected[ID_init_S2C] = prop_infected_init                                         # % of trees infected at time of infection
    num_infected[ID_init_S2C] = host_population[ID_init_S2C]*prop_infected_init             # number of trees infected at time of infection

    ### rate of all uninfected points being infected by the infected C/I points:
    ### using % of infected in each infected point and kernel of the concerned uninfected point to each infected point.
    def calc_rates_S2C(kernel, beta):
        #### ========= step1: get % of infected in each infected point
        # ID_infected = np.array([i for i, s in enumerate(state) if s in ('C', 'I')])
        ID_infected = np.where(state != 'S')[0]  # get indices of infected points in ID_nonzero_points
        #### ========= step2: get the ID of not infected points that have trees
        # ID_notInfected = np.array([i for i in ID_nonzero_points if i not in ID_infected])
        ID_notInfected = np.where((host_population > 0) & (state == 'S'))[0]  # get indices of not infected points in ID_nonzero_points
        #### ========= step3: get the kernel of these and calculate rate for each non-infected point
        #### K(L1, L_1), K(L1, L_2), ..., K(L1, L_m),
        #### K(L2, L_1), K(L2, L_2), ..., K(L2, L_m),
        #### ...
        #### K(Ln, L_1), K(Ln, L_2), ..., K(Ln, L_m)
        #### where L1,...,Ln are not infected with #trees>0; L_1,...,L_m are infected
        
        kernel2 = kernel[ID_notInfected[:, None], ID_infected[None, :]]  # Broadcasting to get the kernel values between non-infected and infected points.#rows=#(not infected), #cols=#(infected)
        
        #### ========= step4: get the weights of infected according to % of the infected cell being occupied by plants
        w_infected = (prop_infected* prop_plant)[ID_infected]
        
        rate_vals = np.matmul(kernel2, w_infected) * beta * prop_plant[ID_notInfected]  # Dot product used for sum to get the rate of infection for each non-infected point. 
        rates_S2C_arr = np.zeros(kernel.shape[0])
        rates_S2C_arr[ID_notInfected] = rate_vals

        return rates_S2C_arr, ID_infected

    #### step5: get the rates of S2C for non-infected points
    rates_S2C, ID_infected = calc_rates_S2C(kernel, beta)

    ### Initialisation of the while loop
    time_event = time_eventS2C  ### initialise time_event
    prevalence = np.sum(num_infected)/ np.sum(host_population)    ### get % of infected in the entire tree population

    # print('time:', time_event, '\tprevalence:', prevalence)

    while (prevalence < prev_threshold) and (np.any(state!='I')):
        # print(np.all(state!='I'))
        # print(state)
        rates_total = np.sum(rates_S2C)
        time_NextS2C = time_event + np.random.exponential(scale= 1/rates_total)
        
        ### check the earliest C2I time
        time_earliestC2I = np.min(time_1st_C2I_dummy_arr)

        # print('total rate', rates_total, '\tnext S2C time:', time_NextS2C, '\tnext C2I time:', time_earliestC2I)
        if time_NextS2C < time_earliestC2I: # then infect a point (S2C)
            # print('S2C')
            time_event = time_NextS2C

            ##### update the % and # in state C and I using Logistic for time= (next)time_event, before state change
            host_num_inC, host_prop_inC, host_num_inI, host_prop_inI = vlogistic(time= time_event, 
                                                                                time_eventS2C=time_1st_S2C[ID_infected], 
                                                                                time_eventC2I=time_1st_C2I[ID_infected], 
                                                                                host_population=host_population[ID_infected], 
                                                                                sigma0=prop_infected_init, c=logistic_rate)
            num_inC[ID_infected] = host_num_inC
            prop_inC[ID_infected] = host_prop_inC
            num_inI[ID_infected] = host_num_inI
            prop_inI[ID_infected] = host_prop_inI
            num_infected[ID_infected] = host_num_inC + host_num_inI
            prop_infected[ID_infected] = host_prop_inC + host_prop_inI

            #### choose a point to transition S2C
            ID_transition = np.random.choice(ID_nonzero_points, size=1, p=rates_S2C[ID_nonzero_points]/rates_total)
            state[ID_transition]= 'C'   # update state
            time_1st_S2C[ID_transition] = time_event
            time_1st_C2I_dummy_arr[ID_transition] = time_event + np.random.uniform(low=symptom_low, high=symptom_high)     # estimate time when it's detectable
            
            ### update % and # of trees being infected
            prop_inC[ID_transition] = prop_infected_init
            num_inC[ID_transition] = host_population[ID_transition]*prop_infected_init                                                 # record time of state change S2C
            prop_infected[ID_transition] = prop_inC[ID_transition]         # % of trees infected at time of infection
            num_infected[ID_transition] = num_inC[ID_transition]           # of trees infected at time of infection
            ### update rates
            rates_S2C, ID_infected = calc_rates_S2C(kernel, beta)

        else:   # make THE point with earliest time(C2I) detectable
            # print('C2I')
            time_event = time_earliestC2I

            ##### update the % and # in state C and I using Logistic for time= (next)time_event, before state change
            host_num_inC, host_prop_inC, host_num_inI, host_prop_inI = vlogistic(time= time_event, 
                                                                                time_eventS2C=time_1st_S2C[ID_infected], 
                                                                                time_eventC2I=time_1st_C2I[ID_infected], 
                                                                                host_population=host_population[ID_infected], 
                                                                                sigma0=prop_infected_init, c=logistic_rate)
            num_inC[ID_infected] = host_num_inC
            prop_inC[ID_infected] = host_prop_inC
            num_inI[ID_infected] = host_num_inI
            prop_inI[ID_infected] = host_prop_inI
            num_infected[ID_infected] = host_num_inC + host_num_inI
            prop_infected[ID_infected] = host_prop_inC + host_prop_inI

            ### choose THE cell to transition C2I
            ID_transition = np.argmin(time_1st_C2I_dummy_arr)

            ### 'remove' time_earliestC2I from time_1st_C2I_dummy_arr so it won't be considered again.
            time_1st_C2I_dummy_arr[ID_transition] = np.inf

            state[ID_transition]= 'I'   # update state            
            time_1st_C2I[ID_transition] = time_event
            ### initialise % and # of trees transitioned from C2I
            prop_inI[ID_transition] = prop_infected_init
            num_inI[ID_transition] = host_population[ID_transition]*prop_infected_init   
            ### update % and # of trees still in C
            prop_inC[ID_transition] = prop_inC[ID_transition] - prop_inI[ID_transition]
            num_inC[ID_transition] = num_inC[ID_transition] - num_inI[ID_transition]
            # no need to update prop_infected and num_infected as they don't change
            ### update rates
            rates_S2C, ID_infected = calc_rates_S2C(kernel, beta)
            # rates_C2I[ID_transition] = 0

        prevalence = np.sum(num_infected)/ np.sum(host_population)    ### get % of infected in the entire tree population
        # print('time:', time_event, '\tprevalence:', prevalence)

    host_data_simulate = {
        "x": xy[:,0],
        "y": xy[:,1],
        "host_population": host_population,
        'prop_plant': prop_plant,
        "state": state,
        "rates_S2C": rates_S2C,
        # "rates_C2I": rates_C2I,
        "time_1st_S2C": time_1st_S2C,
        "time_1st_C2I": time_1st_C2I,
        "num_inC": num_inC,
        "prop_inC": prop_inC,
        "num_inI": num_inI,
        "prop_inI": prop_inI,
        "num_infected": num_infected,
        "prop_infected": prop_infected
    }

    return host_data_simulate








#print('got here3')
# # ================== Run simulation ==================
# np.random.seed(42)  # for reproducibility
# host_distr=host_distribution(xmax=50000, ymax=50000, num_points_x=100, num_points_y=100,
#         num_study_points=2500,
#         host_min=0, host_max=31250,
#         hetero_points=False)

# test_sim= simulate_ibm(host_distr=host_distr,
#                 num_init_S2C=1,
#                 # inv_lambda,
#                 survey_freq=1, # =1 means survey once per week.
#                 sat_prop=0,
#                 alpha=2270, beta=5.246864e-03, gamma=1/52, gomp_rate=0.0962,
#                 tDuration=52*10,
#                 prev_threshold=1
#                 )   # reach 90% prevalence in 386 weeks ( years)

# # alpha=1210.811    #***
# # beta=1.246864e-05 #***
# # beta=7.246864e-03

# # scenatios:
# # alpha=22700, beta=7.246864e-05, gamma=1/52, gomp_rate=0.0962 # reach 90% prevalence in 386 weeks ( years)
# # alpha=2270, beta=5.246864e-03, gamma=1/52, gomp_rate=0.0962 # reach 50% prevalence in 240 weeks ( years)


# test_sim_df = pd.DataFrame(test_sim)
# display(test_sim_df)


# np.random.seed(42)  # for reproducibility
# host_distr=host_distribution(xmax=50000, ymax=50000, num_points_x=100, num_points_y=100,
#        num_study_points=2500,
#        host_min=0, host_max=31250,
#        hetero_points=False)

#df_forSA = []
#num_simulations = 1000

#print('start simulation')
#start_time = datetime.datetime.now()
#print("Start time:", start_time)
## mean dispersion = 2*alpha
#for i in range(num_simulations):
#        sim= simulate_ibm(host_distr=host_distr,
#                    num_init_S2C=1,
#                    # inv_lambda,
#                    survey_freq=1, # =1 means survey once per week.
#                    # sat_prop=0,
#                    alpha=2270, beta=5.246864e-03, gamma=1/52, gomp_rate=0.0962,
#                    # tDuration=52*10,
#                    prev_threshold=0.05
#                    )
#        df_forSA.append(sim)

# print('end simulation')
#end_time = datetime.datetime.now()
#print("End time:", end_time)
#print("Time taken for simulation:", end_time - start_time)

#print('write data')
#cwd= '/home/physics/estszj/research_project'
#dump(df_forSA, f"{cwd}/df_forSA.joblib")
#print('finished writing data')

# if __name__=="__main__":
#     simulate_ibm(host_distr,
#                 num_init_S2C,
#                 # inv_lambda,
#                 survey_freq, # =1 means survey once per week.
#                 sat_prop,
#                 alpha, beta, gamma, gomp_rate,
#                 tDuration,
#                 prev_threshold
#                 )


### DO NOT USE
def host_distribution_casava_OLD(road, area, x_bound, y_bound,  
                             grid_level=True, plot_poly=True, plot_point=True,
                             plot_save_path='g:/My Drive/00 Postgraduate Warwick/ES98C-Research/03 Work/99 Main code/'):
    """
    Calculate the distribution of production along the road.
    
    Parameters:
    road (GeoDataFrame): GeoDataFrame containing road geometries.
    production (GeoDataFrame): GeoDataFrame containing production geometries.
    x_bound (list): List containing the x-coordinate bounds [min_x, max_x].
    y_bound (list): List containing the y-coordinate bounds [min_y, max_y].
    
    Returns:
    host_data: host distribution (dictionary)
    road_sub: road network (shapefile)
    """

    bbox = box(x_bound[0], y_bound[0], x_bound[1], y_bound[1])
    road_sub = road[road.geometry.intersects(bbox)].copy()
    road_sub['geometry'] = road_sub['geometry'].apply(lambda geom: geom.intersection(bbox))
    area = area[area.layer > 0]  # filter out zero values
    # area = area.reset_index(drop=True)
    area_sub = area[area.geometry.intersects(bbox)].copy()
    area_sub = area_sub.reset_index(drop=True)

    centers = area_sub.geometry.centroid
    centers_arr = np.array([[points.x, points.y] for points in centers])
    prod_vals = area_sub['layer'].values
    
    # print(centers_arr.shape, prod_vals.shape)


    # PROD_PER_HA = 10 # tonnes per hectare
    # FIELD_SIZE = 0.1  # in hectares
    # NUM_PLANTS_PER_HA = 10000  # number of plants per hectare (assume 1 plant per 1 m^2)
    # NUM_PLANTS_PER_FIELD = 1000  # number of plants per field

    PROD_PER_HA = 10 # tonnes per hectare
    FIELD_SIZE = 1  # in hectares
    NUM_PLANTS_PER_HA = 10000  # number of plants per hectare (assume 1 plant per 1 m^2)
    NUM_PLANTS_PER_FIELD = 10000  # number of plants per field
    num_plants_per_cell = (prod_vals/ PROD_PER_HA * NUM_PLANTS_PER_HA).round(0).astype(int)
    area_per_cell= 1000000  #m^2
    prop_plants_per_cell= num_plants_per_cell/area_per_cell # proportion of the cell area is occupied by plants
    
    if grid_level:
        xy_host = centers_arr.copy()  # copy of the host points for plotting
        num_plants = num_plants_per_cell  # number of plants per cell
        prop_plant = prop_plants_per_cell
        num_study_points = len(xy_host)
    
    else:
        num_haPerCell = prod_vals / PROD_PER_HA
        num_fieldsPerCell = np.maximum(1, (num_haPerCell / FIELD_SIZE).round(0)).astype(int)
        num_study_points = np.sum(num_fieldsPerCell)
        num_plants = [NUM_PLANTS_PER_FIELD] * num_study_points  # number of plants per field
        prop_plant = [1.0]*num_study_points   # not applicable
        xy_host = []
        ### in each cell, sample points representing fields
        for i in range(len(centers_arr)):
            bounds = area_sub.geometry[i].bounds
            x_min, y_min, x_max, y_max = bounds
            dx = (FIELD_SIZE*10000)**0.5 /2 + 2 # min distance between adjacent points. 1000 m^2 area, so 1000^0.5 m side length, plus 2 m for buffer
            x_coords = np.arange(x_min, x_max - dx, dx)
            y_coords = np.arange(y_min, y_max - dx, dx)
            xy = np.array(np.meshgrid(x_coords, y_coords)).reshape(2, -1).T
            ### determine the locations of the trees in the landscape
            indices_host_loc = np.random.choice(xy.shape[0], size=num_fieldsPerCell[i], replace=False)
            xy_host_i = xy[indices_host_loc]
            xy_host.append(xy_host_i)

        xy_host = np.vstack(xy_host)
        # print(xy_host)
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(xy_host[:,0], xy_host[:,1], s=0.1, alpha=0.5, c='green')
        road_sub.plot(ax=ax, color='red', linewidth=2)
        # plt.show(block=False)
        plt.savefig(f'{plot_save_path}/host_distr_fieldLevel.pdf', dpi=600)
        plt.clf # clear figure
        plt.close(fig)

    if plot_poly:
        fig, ax = plt.subplots(figsize=(10, 10))
        area_sub.plot(ax=ax, column='layer', cmap='viridis', linewidth=0, edgecolor='none', legend=True)
        road_sub.plot(ax=ax, color='red', linewidth=2)
        # plt.show()
        plt.savefig(f'{plot_save_path}/host_distr_gridLevel_s.pdf', dpi=600)
        plt.clf # clear figure
        plt.close(fig)
        
    if plot_point:
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(centers_arr[:,0], centers_arr[:,1], c=num_plants_per_cell, s=num_plants_per_cell**0.3, cmap='RdYlGn_r', label='number of plants')
        plt.colorbar(scatter, ax=ax, label='number of plants')
        road_sub.plot(ax=ax, color='red', linewidth=2)
        # plt.show()
        plt.savefig(f'{plot_save_path}/host_distr_gridLevel_p.pdf', dpi=600)
        plt.clf # clear figure
        plt.close(fig)

        

    host_data = {
        "xy": xy_host,
        # "x": xy_host[:,0],
        # "y": xy_host[:,1],
        "host_population": num_plants,
        'prop_plant': prop_plant,
        "state": ["S"]*num_study_points,
        "rates_S2C": [0.0]*num_study_points,
        "rates_C2I": [0.0]*num_study_points,
        "time_1st_S2C": [np.inf]*num_study_points,
        "time_1st_C2I": [np.inf]*num_study_points,
        "num_inC": [0.0]*num_study_points,
        "prop_inC": [0.0]*num_study_points,
        "num_inI": [0.0]*num_study_points,
        "prop_inI": [0.0]*num_study_points,
        "num_infected": [0.0]*num_study_points,
        "prop_infected": [0.0]*num_study_points
    }

    
    # return road_sub, area_sub, centers_arr, prod_vals, num_plants
    return host_data, road_sub
    
    

# ==================== Pathogen spread simulation (OLD DO NOT USE) ====================
def simulate_ibm_OLD(host_distr,
                num_init_S2C,
                # inv_lambda,
                survey_freq, # =1 means survey once per week.
                # sat_prop,
                alpha, beta, gamma, gomp_rate,
                # tDuration,
                prev_threshold
                ):
    """ 
    Simulate the spread of disease. Always start with t=0.
    Inputs:
        - host_distr,
        - num_init_S2C,
        - inv_lambda: the inverse of lambda used in the exponential distribution Exp(lambda). Mean waiting time of first infection since pathogen entry.
        - alpha: parameter in the kernel
        - beta: used to calculate the rate of transition from state S to C.
        - gamma: rate of transition from state C to I.
        - gomp_rate: rate of transition from state S to C.
    """
    ### Unpack host_distr
    xy = np.array(host_distr['xy'])
    host_population = np.array(host_distr['host_population'])
    prop_plant = np.array(host_distr['prop_plant'])
    state = np.array(host_distr['state'])
    rates_S2C = np.array(host_distr['rates_S2C'])
    rates_C2I = np.array(host_distr['rates_C2I'])
    time_1st_S2C = np.array(host_distr['time_1st_S2C'])
    time_1st_C2I = np.array(host_distr['time_1st_C2I'])
    prop_inC = np.array(host_distr['prop_inC'])
    num_inC = np.array(host_distr['num_inC'])
    prop_inI = np.array(host_distr['prop_inI'])
    num_inI = np.array(host_distr['num_inI'])
    prop_infected = np.array(host_distr['prop_infected'])
    num_infected = np.array(host_distr['num_infected'])
    
    def display_data():
        data = pd.DataFrame({
        "x": xy[:,0],
        "y": xy[:,1],
        "host_population": host_population,
        "state": state,
        "rates_S2C": rates_S2C,
        "rates_C2I": rates_C2I,
        "time_1st_S2C": time_1st_S2C,
        "time_1st_C2I": time_1st_C2I,
        "num_inC": num_inC,
        "prop_inC": prop_inC,
        "num_inI": num_inI,
        "prop_inI": prop_inI,
        "num_infected": num_infected,
        "prop_infected": prop_infected
        })
        display(data)
    
    # ID_nonzero_points = np.array([i for i, n in enumerate(host_population) if n!=0])        # get indices of host points with >0 trees
    ID_nonzero_points = np.where(host_population > 0)[0]  # get indices of host points with >0 trees (same as above, but more efficient)
    # num_detectable_dummy = host_population * 0.01     # when a point gets infected, assume 1% of the trees in the point are infected.
    prop_infected_init = 0.01                         # when a point gets infected, assume 1% of the trees in the point are infected.
    b = -np.log(0.01)                                 # = 4.605 = -ln(% of infected at first time of infection / 100%). Used in Gompertz (% infected in point = N = ae^(-b*e^(-ct))). 

    ### Kernel
    distance_between_points = squareform(pdist(xy))   # pairwise distance between points
    kernel = np.exp(-distance_between_points/ alpha)  # Ornstein-Uhlenbeck kernel


    ### sample (num_init_S2C number of) points that are infected (S -> C) by the pathogen entry =============================
    # time_eventS2C = np.random.exponential(scale=inv_lambda)                                 # time when first infection happens since pathogen entry at t=0
    time_eventS2C = np.random.uniform(0, survey_freq)                                       # time when first infection happens since pathogen entry at t=0. Sampled uniformly between t=0 and t=1.
    ID_init_S2C = np.random.choice(ID_nonzero_points, size=num_init_S2C, replace=False)     # index (indices) of first infected point(s)
    state[ID_init_S2C] = 'C'                                                                # update state
    rates_S2C[ID_init_S2C] = 0
    rates_C2I[ID_init_S2C] = gamma                                                          # rate of transition from C to I of the points in C. should this stay constant as more trees become C in the point?
    time_1st_S2C[ID_init_S2C] = time_eventS2C                                               # record time of state change S2C of the 1st infected point
    prop_inC[ID_init_S2C] = prop_infected_init                                         # % of trees infected at time of infection
    num_inC[ID_init_S2C] = host_population[ID_init_S2C]*prop_infected_init             # number of trees infected at time of infection
    prop_infected[ID_init_S2C] = prop_infected_init                                         # % of trees infected at time of infection
    num_infected[ID_init_S2C] = host_population[ID_init_S2C]*prop_infected_init             # number of trees infected at time of infection

    
    ### rate of all uninfected points being infected by the infected C/I points:
    ### using % of infected in each infected point and kernel of the concerned uninfected point to each infected point.
    def calc_rates_S2C(kernel, beta):
        #### ========= step1: get % of infected in each infected point
        # ID_infected = np.array([i for i, s in enumerate(state) if s in ('C', 'I')])
        ID_infected = np.where(state != 'S')[0]  # get indices of infected points in ID_nonzero_points
        #### ========= step2: get the ID of not infected points that have trees
        # ID_notInfected = np.array([i for i in ID_nonzero_points if i not in ID_infected])
        ID_notInfected = np.where((host_population > 0) & (state == 'S'))[0]  # get indices of not infected points in ID_nonzero_points
        #### ========= step3: get the kernel of these and calculate rate for each non-infected point
        #### K(L1, L_1), K(L1, L_2), ..., K(L1, L_m),
        #### K(L2, L_1), K(L2, L_2), ..., K(L2, L_m),
        #### ...
        #### K(Ln, L_1), K(Ln, L_2), ..., K(Ln, L_m)
        #### where L1,...,Ln are not infected with #trees>0; L_1,...,L_m are infected
        
        kernel2 = kernel[ID_notInfected[:, None], ID_infected[None, :]]  # Broadcasting to get the kernel values between non-infected and infected points
        rate_vals = np.matmul(kernel2, prop_infected[ID_infected]) * beta  # Dot product to get the rate of infection for each non-infected point
        rates_S2C_arr = np.zeros(kernel.shape[0])
        rates_S2C_arr[ID_notInfected] = rate_vals

        # rates_S2C_arr = np.zeros(kernel.shape[0])
        # for id_ninf in ID_notInfected:
        #     rate = np.sum([kernel[id_ninf, id_inf] * prop_infected[id_inf] for id_inf in ID_infected]) * beta
        #     rates_S2C_arr[id_ninf] = rate
        return rates_S2C_arr, ID_infected
    
    #### step4: get the rates of S2C for non-infected points
    # rates_S2C = ... # kernel * number of hosts in C/I points ??? if a point is already infected (!=S), then change its rate_S2C to -1 (or 0, or 999? which is better?)
    # rates_S2C[ID_0tree] = 0  # if there's no tree in a point, rate=0
    rates_S2C, ID_infected = calc_rates_S2C(kernel, beta)

    

    ### Initialisation of the while loop
    # ID_inS = np.array([i for i in ID_nonzero_points if state[i]=='S'])  ### get ID in S
    # ID_inS = np.where((host_population> 0) & (state == 'S'))[0]  # get indices of points in S with trees
    # time_eventC2I = -1  ### initialise time_eventC2I
    time_event = time_eventS2C  ### initialise time_event
    prevalence = np.sum(num_infected)/ np.sum(host_population)    ### get % of infected in the entire tree population

    ### get time and ID of the next S2C point.
    ### get time and ID of the next C2I point.
    ### if time_S2C < time_C2I then take the S2C time & ID. update rates_S2C for all points. 
    ##### otherwise take the C2I time & ID. update rates_S2C for all points, AND update rates_C2I for the C points.
    ### repeat.
    # print('init time event:', time_event)
    # print('init prevalence:', prevalence)
    # while (time_event < tDuration) & (prevalence < prev_threshold):    # maybe take out tDuration later as it depends on when prevalence threshold is reached.
    while prevalence < prev_threshold:
        # print("ID infected:", ID_infected)
        # display_data()
        ### Gillespie's algorithm: sample when when next event happens, and which event happens next. 
        ##### step 1: calcluate total rates & individual rates
        rates = rates_S2C + rates_C2I
        rates_total = np.sum(rates)
        # print("rates: ", rates)
        # print("total rates: ", rates_total)
        ##### step 2: sample time for next event & update time
        time_toNextEvent = np.random.exponential(scale= 1/rates_total)
        time_event = time_event + time_toNextEvent
        # print('time event:', time_event)
        ##### step 3: update the % and # in state C and I using Gompertz for time= (next)time_event, before state change
        # for id in ID_infected:
        #     host_num_inC, host_prop_inC, host_num_inI, host_prop_inI = gompertz(time= time_event, 
        #                                                                         time_eventS2C=time_1st_S2C[id], 
        #                                                                         time_eventC2I=time_1st_C2I[id], 
        #                                                                         host_population=host_population[id], 
        #                                                                         b=b, c=gomp_rate)
        #     num_inC[id] = host_num_inC
        #     prop_inC[id] = host_prop_inC
        #     num_inI[id] = host_num_inI
        #     prop_inI[id] = host_prop_inI
        #     num_infected[id] = host_num_inC + host_num_inI
        #     prop_infected[id] = host_prop_inC + host_prop_inI
        
        host_num_inC, host_prop_inC, host_num_inI, host_prop_inI = vgompertz(time= time_event, 
                                                                            time_eventS2C=time_1st_S2C[ID_infected], 
                                                                            time_eventC2I=time_1st_C2I[ID_infected], 
                                                                            host_population=host_population[ID_infected], 
                                                                            b=b, c=gomp_rate)
        num_inC[ID_infected] = host_num_inC
        prop_inC[ID_infected] = host_prop_inC
        num_inI[ID_infected] = host_num_inI
        prop_inI[ID_infected] = host_prop_inI
        num_infected[ID_infected] = host_num_inC + host_num_inI
        prop_infected[ID_infected] = host_prop_inC + host_prop_inI


        ##### step 4: sample which point experiences the state transition
        ID_transition = np.random.choice(ID_nonzero_points, size=1, p=rates[ID_nonzero_points]/rates_total)
        # print(rates[ID_nonzero_points]/rates_total)
        if state[ID_transition]=='S':
            state[ID_transition]= 'C'   # update state
            time_1st_S2C[ID_transition] = time_event 
            ### update % and # of trees being infected
            prop_inC[ID_transition] = prop_infected_init
            num_inC[ID_transition] = host_population[ID_transition]*prop_infected_init                                                 # record time of state change S2C
            prop_infected[ID_transition] = prop_inC[ID_transition]         # % of trees infected at time of infection
            num_infected[ID_transition] = num_inC[ID_transition]           # of trees infected at time of infection
            ### update rates
            rates_S2C, ID_infected = calc_rates_S2C(kernel, beta)
            rates_C2I[ID_transition] = gamma    
            # print("state change S2C")
        elif state[ID_transition]=='C':
            state[ID_transition]= 'I'   # update state            
            time_1st_C2I[ID_transition] = time_event
            ### get % and # of trees transitioned from C2I
            prop_inI[ID_transition] = prop_infected_init
            num_inI[ID_transition] = host_population[ID_transition]*prop_infected_init   
            ### update % and # of trees still in C
            prop_inC[ID_transition] = prop_inC[ID_transition] - prop_inI[ID_transition]
            num_inC[ID_transition] = num_inC[ID_transition] - num_inI[ID_transition]
            # no need to update prop_infected and num_infected as they don't change
            ### update rates
            rates_S2C, ID_infected = calc_rates_S2C(kernel, beta)
            rates_C2I[ID_transition] = 0
            # print("state change C2I")
        else:
            print("There's an error in the state of chosen ID: ", state[ID_transition])
            break
        
        prevalence = np.sum(num_infected)/ np.sum(host_population)    ### get % of infected in the entire tree population
        # print('prevalence:', prevalence)


    host_data_simulate = {
        "x": xy[:,0],
        "y": xy[:,1],
        "host_population": host_population,
        "state": state,
        "rates_S2C": rates_S2C,
        "rates_C2I": rates_C2I,
        "time_1st_S2C": time_1st_S2C,
        "time_1st_C2I": time_1st_C2I,
        "num_inC": num_inC,
        "prop_inC": prop_inC,
        "num_inI": num_inI,
        "prop_inI": prop_inI,
        "num_infected": num_infected,
        "prop_infected": prop_infected
    }

    return host_data_simulate