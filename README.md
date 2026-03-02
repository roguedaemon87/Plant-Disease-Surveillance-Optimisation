# Plant Disease Surveillance Optimisation

This repository contains the code and data used to simulate plant pathogen
spread and to optimise surveillance site locations, as described in the
associated manuscript.

---

# Repository Structure

The repository is organised into two primary analytical workflows:

- `main_analysis/` — Reproduction of the main manuscript results  
- `beta_parameterisation/` — Optional calibration workflow  

Core model files are located in the repository root.

---

# 1. Main Analysis (`main_analysis/`)

This folder contains everything required to reproduce the main manuscript results.

## Inputs

- `main_analysis/host_distributions/`  
  Precomputed host distribution landscapes (`.joblib` files) used as inputs to
  the epidemic spread simulations.

- `main_analysis/road_patterns/`  
  Road network shapefiles used during optimisation to represent accessibility
  constraints for candidate surveillance locations.

## Code

- `main_analysis/code/`  
  Scripts used to:
  - Run epidemic spread simulations  
  - Optimise surveillance site locations  
  - Generate manuscript figures and summary results  

- `main_analysis/code/optimisation/outlier_cases/`  
  Diagnostic scripts used to investigate specific outlier configurations where
  simple road accessibility metrics performed poorly.  
  These analyses are supplementary and are **not required** to reproduce the
  main manuscript results.

- `main_analysis/code/nigeria_comparison/`  
  Nigeria comparison analysis.
  Contains the inputs, simulation and optimisation workflows used to compare
  Nigerian road networks with those of the DRC.
  These analyses are supplementary and are **not required** to reproduce the
  main manuscript results.

Users interested only in reproducing the manuscript results should focus on
the `main_analysis/` folder.

---

## Execution Order (Main Analysis)

1. **Run epidemic simulations**
  - 'main_analysis/code/simulations/run_sims.py

2. **Optimise surveillance site locations**
  - Run the relevant scripts in: 'main_analysis/code/optimisation/

3. **Generate figures and summaries**
  - 'main_analysis/code/figure_generation/
  -  See individual script headers for required arguments.

---

# 2. Beta Parameterisation (`beta_parameterisation/`)

This folder contains OPTIONAL code used to calibrate (parameterise) the
transmission parameter `beta` prior to the main analysis.

This workflow is **not required** to reproduce the manuscript results.

It includes:

- `beta_parameterisation/code/`  
Scripts that sweep over candidate beta values and measure epidemic timing metrics.

- `beta_parameterisation/Outputs/`  
Result files from the beta calibration experiments.

- `beta_parameterisation/shapefiles_raw/`  
Raw GIS input layers used for beta calibration  
(not hosted here due to file size limits).

After inspecting beta calibration outputs, a beta value was selected manually
and specified in `CONSTANT.py` for the main analysis.

---

# Core Model Files

The following Python modules in the repository root define the core modelling
framework and are used by both `main_analysis/` and `beta_parameterisation/`:

- `simulation.py`  
Implements the epidemic spread model (IBM), including infection dynamics and
stochastic simulations.

- `optimisation.py`  
Implements surveillance optimisation routines (simulated annealing) to
maximise the probability of detecting infection before a prevalence threshold
is reached.

- `CONSTANT.py`  
Defines baseline model parameters (infection parameters, logistic growth
rates, detection thresholds, etc.).

Together, these files implement the modelling framework, while scripts in
`main_analysis/` and `beta_parameterisation/` call these functions to produce
the results reported in the manuscript.

---

# Quick Start

To reproduce the main manuscript results:

1. Run simulations: 'main_analysis/code/simulations/run_sims.py <AREA_ID>
2. Run optimisation scripts in: 'main_analysis/code/optimisation/
3. Generate figures in: 'main_analysis/code/figure_generation/

---

# Notes

- The beta parameterisation workflow is optional.
- Raw GIS shapefiles used for beta calibration are not included due to file size constraints.
- All script paths are relative to the repository root.
- Outlier case analyses are supplementary diagnostics.
- For questions about the code or data, please contact the corresponding author.
