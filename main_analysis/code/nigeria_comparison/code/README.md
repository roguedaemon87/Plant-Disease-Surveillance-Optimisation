Nigeria Comparison – Code
=========================

This folder contains the Python scripts used to run epidemic simulations and
surveillance optimisation for the Nigeria comparison analysis described in
the manuscript.

Overview
--------
These scripts replicate the modelling framework used in the main analysis,
but applied to Nigerian state host landscapes and road networks. The purpose
is to:

1. Quantify differences in road accessibility metrics between Nigeria and DRC.
2. Evaluate how restricting surveillance to road-accessible sites affects
   detection probability under Nigerian conditions.

Workflow
--------

1. Preprocess host distributions
   --------------------------------
   Host distribution shapefiles must first be converted into `.joblib`
   files and placed in:

       nigeria_comparison/host_distributions/

   See that folder’s README for required structure.

2. Run epidemic simulations
   --------------------------------
   run_sims_nigeria.py

   - Computes a dispersal kernel for the state landscape
   - Runs stochastic epidemic simulations using simulate_ibm()
   - Writes outputs to:

       nigeria_comparison/Outputs/simulations/

3. Optimise surveillance site placement
   --------------------------------

   a) Road-constrained optimisation:
      optimise_sites_with_roads_nigeria.py

   b) No-road (ideal baseline) optimisation:
      optimise_sites_no_roads_nigeria.py

   Both scripts:
   - Use simulated annealing
   - Call functions from optimisation_nigeria.py
   - Write outputs to:

       nigeria_comparison/Outputs/optimisation/

Modules
-------

simulation_nigeria.py
    Defines:
    - compute_kernel() for precomputing the power-law dispersal kernel
    - simulate_ibm() with optional kernel input

optimisation_nigeria.py
    Defines:
    - Objective functions
    - Road accessibility filtering
    - Simulated annealing with checkpointing support
    - Road and optimisation performance metrics (e.g., RCP, CPA)

run_sims_nigeria.py
    Driver script for generating simulation datasets.

optimise_sites_with_roads_nigeria.py
    Performs surveillance optimisation under road-access constraints.

optimise_sites_no_roads_nigeria.py
    Computes idealised baseline optimisation without road constraints.

Notes
-----
- This module is separate from the main DRC analysis and is not required
  to reproduce the core manuscript results.
- Road metric naming in code may use "RCAP", while the manuscript refers
  to the same concept as "CPA".
- Raw GIS layers are not included in the repository due to file size limits.
