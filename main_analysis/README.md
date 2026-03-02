# Main Manuscript Analysis  
*(Reproducibility Inputs & Workflows)*

This folder contains the data inputs and code required to reproduce the
main manuscript analysis.

---

## Folder Structure

### `host_distributions/`
Precomputed host distribution landscapes (`.joblib` files) used by the
epidemic spread simulations.  

**REQUIRED** for reproducing simulation results.

---

### `road_patterns/`
Road network shapefiles used during optimisation to apply accessibility
constraints to candidate surveillance locations.  

**REQUIRED** for reproducing optimisation results.

---

### `code/`
Scripts used to:

- Run epidemic spread simulations  
- Optimise surveillance site locations  
- Generate manuscript figures and summary results  
- Perform additional diagnostic and comparison analyses  

See `main_analysis/code/README.md` for detailed workflow descriptions.

---

## Optional Workflows

The beta calibration workflow is located separately in:
beta_parameterisation/

This workflow is not required to reproduce the main manuscript results.
