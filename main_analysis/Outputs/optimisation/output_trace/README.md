# Optimisation traces

One file per optimisation run, holding the objective value at every
simulated annealing iteration. These are the traces plotted in the lower
row of Figure 4.

## Contents

Each file is a Python list, saved with joblib, whose length is the number
of iterations the run took. Element *i* is the objective value of the
configuration accepted at iteration *i*, so the series is non-decreasing
only in stretches where no worsening move was accepted. Lengths differ
between runs because the search stops early once the objective settles,
which is described in the Supplementary Materials.

Load one with:

    from joblib import load
    trace = load("trace_objval_road01_4_area01_surveyfreq52_nsites10.joblib")

## File names

    trace_objval_road{ROAD}_area{AREA}_surveyfreq{FREQ}_nsites{N}.joblib
    trace_objval_NoRoad_area{AREA}_surveyfreq{FREQ}_nsites{N}.joblib

where ROAD is a road network identifier such as `01_4`, AREA a
sub-landscape from `01` to `08`, FREQ the interval between surveys in
weeks, and N the number of surveillance sites. The `NoRoad` form is the
unconstrained run for that sub-landscape, in which any populated cell may
be chosen, and it provides the denominator for the optimal objective
proportion.

## Produced by

    main_analysis/code/optimisation/optimise_sites_with_roads.py
    main_analysis/code/optimisation/optimise_sites_no_roads.py
