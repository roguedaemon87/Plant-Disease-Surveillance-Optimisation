This folder contains outputs from the beta parameterisation experiments.

Each calibration run produces two joblib files:

1) logbeta_*.joblib
   These files store the candidate transmission parameter values (log(beta))
   that were tested during the sweep.

2) timediff_*.joblib
   These files store the corresponding epidemic growth metrics for each
   candidate beta value. Specifically, for each log(beta), the code computes
   the time difference between reaching two prevalence thresholds
   (PREV_FINAL0 and PREV_FINAL1). Smaller time differences indicate faster
   epidemic growth.

The logbeta and timediff files are paired by index:
logbeta[i] corresponds to timediff[i].

These files are outputs only. The beta value used in the main analysis was
selected manually based on these results and then specified in CONSTANT.py.

Note:
Beta parameterisation is optional and not required to reproduce the main
manuscript results, which use precomputed host distributions.
