Nigeria comparison workflow (Ferris et al. 2024 context)
========================================================

Purpose
-------
This folder contains the code used to run epidemic simulations and surveillance
optimisation for the five Nigerian states analysed in Ferris et al. (2024),
to support comparison with the Democratic Republic of Congo (DRC) results in
the manuscript.

This workflow was used to:
1) quantify differences in road accessibility between Nigeria and DRC using the
   road metrics developed in this repository (RCP and CPA), and
2) assess whether these road metrics help explain why road-constrained
   surveillance can have little effect in Nigeria but a strong effect in DRC.

Folder contents
---------------
code/
  - simulation_nigeria.py
      Nigeria-specific simulation module. Adds compute_kernel() to precompute
      the power-law dispersal kernel efficiently, and allows simulate_ibm()
      to use a precomputed kernel.

  - run_sims_nigeria.py
      Runs compute_kernel() first, then runs epidemic simulations using
      simulate_ibm(..., kernel=<precomputed kernel>).

  - optimisation_nigeria.py
      Nigeria-specific optimisation module. Core functionality matches the main
      optimisation module, with added checkpointing support in simulated_annealing()
      so long runs can be resumed.

  - ana11_opt.py
      Driver script for road-constrained optimisation (surveillance locations
      restricted to road-accessible host cells). Uses checkpointing.

  - ana11_opt_NoRoad.py
      Driver script for no-road baseline optimisation (surveillance locations
      can be placed anywhere on the host grid). Uses checkpointing.

host_distributions/
  Precomputed host distribution landscapes (.joblib) for Nigerian states.

road_networks/
  Road network shapefiles for Nigerian states.

Metrics terminology
-------------------
Some code uses the name RCAP for the road accessibility proportion:
    RCAP = (# host cells accessible from roads) / (# host cells with hosts > 0)

In the manuscript, this metric is referred to as CPA.
RCP terminology is unchanged.

Notes
-----
- This Nigeria workflow is separate from the main DRC analysis and is not
  required to reproduce the main manuscript results.
- All paths in scripts should be set relative to the repository.
