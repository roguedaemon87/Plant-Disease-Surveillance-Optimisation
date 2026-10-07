# Figure generation

One script per figure. Each resolves its own paths from the repository root,
so they can be run from any working directory and need no editing:

```
python main_analysis/code/figure_generation/figure5.py
```

Output is written to `main_analysis/Outputs/figures/`.

## Scripts

| Script | Produces | Reads |
|---|---|---|
| `figure2.py` | Figure 2, host distributions, road networks and infection probabilities for two sub-landscapes | host distributions, road networks, infection visit counts |
| `figure3.py` | Figure 3, dispersal distance distribution, and Figure S6, the same by sub-landscape | dispersal distances |
| `figure4.py` | Figure 4, optimal sites and annealing traces under three accessibility scenarios | optimisation, host distributions, road networks, infection visit counts |
| `figure5.py` | Figure 5, correlation strength between the twelve road network metrics and OOP | optimisation, host distributions |
| `figure6.py` | Figure 6, OOP against four accessibility metrics with the Nigerian states overlaid | optimisation, host distributions, Nigeria comparison |
| `figureS3.py` | Figure S3, cassava host abundance across the eight sub-landscapes | host distributions |
| `figureS4.py` | Figure S4, the 17 road networks grouped by series | road networks |
| `figureS5.py` | Figure S5, convergence of the objective estimate with simulation count | simulations |
| `figureS7.py` | Figure S7, variance decomposition and bootstrap rank stability | simulations, optimisation |
| `figureS8.py` | Figure S8, sensitivity to within-cell variation in disease progression | noise test |
| `figureS9.py` | Figure S9, pairwise correlations among the twelve metrics | optimisation, host distributions |
| `figureS10.py` | Figure S10, distribution of OOP by sub-landscape and by road network | optimisation |
| `figureS11.py` | Figure S11, study areas whose road network is paired with its own sub-landscape | optimisation |
| `figureS12.py` | Figure S12, the study areas falling furthest below the fitted RCP-OOP trend | optimisation, host distributions, road networks, infection visit counts |
| `figureS13.py` | Figure S13, controlled test cases for the hotspot explanation | outlier cases, main optimisation |
| `figureS14.py` | Figure S14, annealing traces for the five Nigerian states | Nigeria comparison |
| `summarise_optimisation_results.py` | Site-level metrics reported in Supplementary Table S3 | optimisation, host distributions |

## Where the inputs live

| Name used above | Path |
|---|---|
| optimisation | `main_analysis/Outputs/optimisation/` |
| simulations | `main_analysis/Outputs/simulations/` |
| dispersal distances | `main_analysis/Outputs/dispersal/` |
| noise test | `main_analysis/Outputs/noise_test/` |
| outlier cases | `main_analysis/Outputs/outlier_cases/` |
| host distributions | `main_analysis/host_distributions/` |
| road networks | `main_analysis/road_patterns/` |
| infection visit counts | `main_analysis/Outputs/infection_visit_counts/` |
| Nigeria comparison | `main_analysis/code/nigeria_comparison/Outputs/` |

The host distributions and road networks are in the repository. The remaining
inputs are produced by the simulation and optimisation scripts, so run those
first if the directories are empty. The execution order is given in the
repository README.

## Figures not produced here

Figure 1 and Supplementary Figures S1 and S2 are reproduced photographs or
maps drawn in GIS rather than computed, so they have no script. Everything
else in the manuscript is produced by the scripts above.

## A note on the figures as published

The published figures were adjusted by hand after being generated, mainly
to reposition colourbars and enlarge labels for print. These scripts
reproduce the content of each figure, not every typographic detail of the
version that appears in the journal.
