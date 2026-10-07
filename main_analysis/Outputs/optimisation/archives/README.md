# Full annealing traces and configurations

The complete record of every simulated annealing run in the main analysis,
1,296 of each, held as compressed archives because the uncompressed files
come to roughly 5.3 GB in total and would make the repository impractical
to clone.

One archive per surveillance intensity:

    output_trace_sfreq{FREQ}_nsite{N}.tar.gz     144 files,  ~16 MB
    output_config_sfreq{FREQ}_nsite{N}.tar.gz    144 files,  ~10 MB

FREQ is the interval between surveys in weeks, one of 26, 52 or 104, and N
the number of surveillance sites, one of 5, 10 or 15. Each archive holds the
136 road-constrained study areas plus the 8 unconstrained sub-landscape
baselines.

## Unpacking

Extract into the parent directory, which puts the files into
`output_trace/` and `output_config/` beside the ones already there:

    cd main_analysis/Outputs/optimisation
    tar xzf archives/output_trace_sfreq52_nsite10.tar.gz
    tar xzf archives/output_config_sfreq52_nsite10.tar.gz

Uncompressed, the traces are about 426 MB in total and the configurations
about 4.9 GB, so take only the intensities you need.

## What is in them

A trace file is a list of the objective value at every iteration, so its
length is the number of iterations that run took. A configuration file is
the corresponding list of surveillance site configurations, one per
iteration, each an array of N cell indices. The configurations are large
for that reason, holding N values per iteration where a trace holds one.

## You probably do not need these

Nothing in the manuscript is computed from the full set. Every reported
figure and statistic comes from the metric files in the
`output_metric_sfreq{FREQ}_nsite{N}/` directories, which are committed
uncompressed and include each run's stopping iteration in the
`n_iter_actual` field. The only script that reads traces or configurations
is `figure4.py`, which opens three of each for sub-landscape 01, and those
six files are already present in `output_trace/` and `output_config/`.

These archives are here so that the annealing history behind every run can
be inspected, not because anything depends on them.
