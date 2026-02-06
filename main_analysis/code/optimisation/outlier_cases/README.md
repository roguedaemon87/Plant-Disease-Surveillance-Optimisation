Outlier case analysis
--------------------

This folder contains diagnostic scripts used to investigate specific cases
where simple road accessibility metrics performed poorly.

Workflow:

1. run_sims_outlier_cases.py
   Runs epidemic simulations for selected host distributions (area01, area05).

2. optimise_sites_with_roads_outlier_cases.py
   Optimises surveillance locations constrained to specific road networks.

3. optimise_sites_no_roads_outlier_cases.py
   Computes an idealised baseline (no road constraints).

4. optimise_sites_with_roads_outlier_cases_legacy.py
   Legacy exploratory script retained for provenance only.

Inputs:
- host distributions:
    inputs/host_distributions/
- road networks:
    inputs/road_patterns/

Outputs:
- main_analysis/Outputs/outlier_cases/

These analyses are supplementary diagnostics and are not required to reproduce
the main manuscript results.
