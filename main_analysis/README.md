MAIN MANUSCRIPT ANALYSIS (REPRODUCIBILITY INPUTS)

This folder contains the data inputs required to reproduce the main manuscript
analysis.

Subfolders:
- host_distributions/ : precomputed host distribution landscapes (.joblib) used
  by the epidemic spread simulations. REQUIRED.

- road_patterns/ : road network shapefiles used during optimisation to apply
  accessibility constraints. REQUIRED for reproducing optimisation results.

The optional beta calibration workflow is located separately in:
beta_parameterisation/
