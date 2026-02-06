This folder contains the Python scripts used to parameterise (calibrate) the
transmission parameter beta for the spread model.

Scripts:
- parameterise_beta.py       : baseline beta sweep
- parameterise_beta_2yrs.py  : beta sweep using a 2-year simulation horizon
- parameterise_beta_7yrs.py  : beta sweep using a 7-year simulation horizon

Inputs:
- Raw GIS shapefiles (roads + production/host layer) are expected in:
  data_inputs/beta_parameterisation/shapefiles_raw/

Outputs:
- Results are written as .joblib files to:
  parameterise_beta/Outputs/

Important:
This calibration workflow is OPTIONAL and is not required to reproduce the main
manuscript results. The main analysis uses precomputed host distributions.
A beta value was selected manually based on these sweep results and then set
in CONSTANT.py (or config.py).
