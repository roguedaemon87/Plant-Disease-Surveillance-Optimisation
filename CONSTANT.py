"""
MODEL PARAMETERS (CONSTANTS)
===========================

This file defines baseline parameter values used throughout the modelling
framework.

These parameters include (but are not limited to):
- Epidemiological parameters (e.g., transmission and growth rates)
- Detection and prevalence thresholds
- Survey-related settings

Values defined here are imported by multiple scripts, including:
- beta_parameterisation scripts
- main_analysis simulation scripts
- optimisation scripts

This file provides a single central location for model settings so that
parameter values are consistent across workflows.

In the main analysis, the transmission parameter beta was selected manually
based on outputs from beta_parameterisation and then specified here.

Changing values in this file will affect all downstream simulations and
optimisation results.

"""

import numpy as np
### simulation params
ALPHA=3.5  #3.75
BETA=np.exp(8)
SIGMA0=0.01  #0.01/20 #0.00075  #SIGMA0=0.01/20  # SIGMA0=0.01 
LOGISTIC_RATE=0.1693  #0.1966  #0.1317  #LOGISTIC_RATE=0.25
PREVALENCE=0.05

### optimisation params
SURVEY_FREQ=26
P_DETECT=0.75
NTREES_SURVEY=30
NSITES=5

### parameterising beta params
PREV_FINAL=0.24  # test how long each beta takes to reach 24% prevalence
