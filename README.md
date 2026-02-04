Plant Disease Surveillance Optimisation
=====================================

This repository contains the code and data used to simulate plant pathogen
spread and to optimise surveillance site locations, as described in the
associated manuscript.

The repository is organised into two main folders:

---------------------------------------------------------------------------
1) main_analysis/
---------------------------------------------------------------------------

This folder contains everything required to reproduce the main manuscript
results.

Inputs:
- main_analysis/host_distributions/
  Precomputed host distribution landscapes (.joblib files) used as inputs to
  the epidemic spread simulations.

- main_analysis/road_patterns/
  Road network shapefiles used during optimisation to represent accessibility
  constraints for candidate surveillance locations.

Code:
- main_analysis/code/
  Scripts used to:
    - run epidemic spread simulations
    - optimise surveillance site locations
    - generate summary results and figures

Users interested only in reproducing the manuscript results should focus on
the main_analysis folder.


---------------------------------------------------------------------------
2) beta_parameterisation/
---------------------------------------------------------------------------

This folder contains OPTIONAL code used to calibrate (parameterise) the
transmission parameter beta prior to the main analysis.

This workflow is NOT required to reproduce the manuscript results.

It includes:
- beta_parameterisation/code/
  Scripts that sweep over candidate beta values and measure epidemic timing
  metrics.

- beta_parameterisation/Outputs/
  Result files from the beta calibration experiments.

- beta_parameterisation/shapefiles_raw/
  Raw GIS input layers used for beta calibration (not hosted here due to file
  size limits).

After inspecting beta calibration outputs, a beta value was selected manually
and then specified in CONSTANT.py for the main analysis.


---------------------------------------------------------------------------
Core model files
---------------------------------------------------------------------------

At the top level of the repository are three core Python files that define the
model behaviour and are used by both main_analysis and beta_parameterisation:

- simulation.py
  Contains the epidemic spread model. This file defines how infection spreads
  across the landscape, how hosts become infected over time, and how stochastic
  simulations are run.

- optimisation.py
  Contains the optimisation routines used to identify surveillance site
  locations that maximise the probability of detecting infection before a
  prevalence threshold is reached.

- CONSTANT.py
  Defines baseline parameter values used throughout the model (e.g. infection
  parameters, logistic growth rates, and detection thresholds). These parameters
  are imported by multiple scripts and provide a central location for model
  settings.

Together, these files implement the core modelling framework, while the scripts
in main_analysis/ and beta_parameterisation/ call these functions to produce the
results reported in the manuscript.


---------------------------------------------------------------------------
Notes
---------------------------------------------------------------------------

- The beta parameterisation workflow is optional.
- Raw GIS shapefiles used for beta calibration are not included due to file size
  constraints.
- All paths in scripts are relative to this repository.

For questions about the code or data, please contact the corresponding author.
