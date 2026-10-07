# Spread simulations

The stochastic spread simulations behind every result in the manuscript, one
file per sub-landscape.

    df_sims01.joblib  ...  df_sims08.joblib

Each holds 2,000 independent simulations of cassava brown streak disease
spread, run to the point at which landscape-level prevalence reaches 5%.

## Loading

    from joblib import load
    sims = load("df_sims01.joblib")      # list of 2,000 simulations

The files are stored gzipped to keep the repository to a workable size,
roughly 50 MB each in place of 517 MB. `joblib.load` detects the
compression and reads them directly, so no manual unpacking is needed and
nothing in the code has to change.

## Contents of a single simulation

Each entry is a dictionary describing one realisation across the
sub-landscape's 1 km grid cells.

| Key | Meaning |
|---|---|
| `x`, `y` | cell centroid coordinates, metres |
| `host_population` | cassava plants in the cell |
| `state` | final cell state, susceptible, cryptic or symptomatic |
| `time_1st_S2C` | time at which the cell first became infected |
| `time_1st_C2I` | time at which infection in the cell first became symptomatic |
| `num_infected`, `num_inC`, `num_inI` | plant counts by state |
| `prop_infected`, `prop_inC`, `prop_inI` | the same as proportions of the cell |
| `prop_plant` | the cell's share of the landscape's plants |
| `rates_S2C` | per-cell infection rates |

## Produced by

    main_analysis/code/simulations/run_sims.py <AREA_ID>

Re-running regenerates the simulations, but the draws will differ from
these unless the seeds are reproduced exactly, so these files are the
record of the realisations the published results were computed from.
