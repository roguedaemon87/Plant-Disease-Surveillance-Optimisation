Nigeria Host Distributions
==========================

This folder contains preprocessed host landscape files used for the Nigeria
comparison analysis.

These files must be stored in `.joblib` format (e.g. `Anambra_prod.joblib`,
`Kebbi_prod.joblib`, etc.).

Important
---------
The simulation workflow does NOT read raw GIS shapefiles directly.
Host distribution shapefiles must first be converted into `.joblib`
files before running:

    run_sims_nigeria.py

Expected Filenames
------------------
- Anambra_prod.joblib
- Kebbi_prod.joblib
- Nasarawa_prod.joblib
- Ogun_prod.joblib
- Plateau_prod.joblib

Required Structure
------------------
Each `.joblib` file must contain a dictionary with at least the following keys:

- 'x'               : x-coordinates of host grid cells
- 'y'               : y-coordinates of host grid cells
- 'xy'              : Nx2 NumPy array of coordinates
- 'host_population' : number of host plants per cell
- 'prop_plant'      : proportional planting intensity (used in kernel normalisation)

These objects are used by:

- simulation_nigeria.py (kernel computation and IBM simulation)
- run_sims_nigeria.py (batch simulation driver)
- optimisation_nigeria.py (surveillance optimisation)

Conversion from Shapefiles
--------------------------
Raw host distribution shapefiles are not included in this repository due to
file size constraints.

They must be converted into the required `.joblib` format prior to running
the simulation and optimisation workflows.

The conversion process can be implemented using the host distribution logic
in `simulation_nigeria.py`.

Note
----
This preprocessing requirement mirrors the structure used in the main
analysis (DRC landscapes), where host distributions are also stored as
precomputed `.joblib` files.
