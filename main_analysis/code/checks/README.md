# Checks

Analyses that support statements in the manuscript without producing a
numbered figure.

| Script | Purpose |
|---|---|
| `score_noise_test.py` | Scores every baseline optimum against simulations with within-cell variation in disease progression, writing `noise_test_scores.csv` for `figureS8.py`. Answers Reviewer 1's comment 7. |
| `timing_comparison.py` | Times the accessibility preprocessing against a single objective evaluation, to show that imposing the constraint costs a one-off preprocessing step rather than a per-iteration overhead. Answers Reviewer 1's comment 1. |
