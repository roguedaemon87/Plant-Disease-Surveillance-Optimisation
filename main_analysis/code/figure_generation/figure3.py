"""
Figure 3: distribution of simulated dispersal distances, with the power-law
form of the kernel overlaid, plus a fitted exponent for comparison with
published CBSD kernel estimates.

Panel a: histogram on linear axes (as before), with the fitted density.
Panel b: the same on logarithmic axes, where a power law is a straight line,
         with the expected slope 1 - alpha under a uniform host distribution.

Also writes per-sub-landscape histograms for Fig. S6.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from joblib import load
from scipy.optimize import minimize_scalar

from pathlib import Path

# ==========================================================================
# All paths are resolved relative to the repository root.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

OUTPUTS   = PROJECT_ROOT / "main_analysis" / "Outputs"
DIR_SIMS  = OUTPUTS / "simulations"
DIR_OPT   = OUTPUTS / "optimisation"
DIR_DISTR = PROJECT_ROOT / "main_analysis" / "host_distributions"
DIR_OUT   = OUTPUTS / "figures"
DIR_IN  = OUTPUTS / "dispersal"
# ==========================================================================
ALPHA = 3.5
DMIN = 1.0          # km; one cell width, the shortest possible dispersal
N_BOOT = 200        # bootstrap resamples for the confidence interval
# ==========================================================================

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif']
rcParams['font.weight'] = 500
rcParams['mathtext.fontset'] = 'cm'


def neg_loglik(gamma, d, dmin, dmax):
    """Truncated power-law density f(d) proportional to d^gamma on [dmin, dmax]."""
    if abs(gamma + 1) < 1e-9:
        norm = np.log(dmax / dmin)
    else:
        norm = (dmax**(gamma + 1) - dmin**(gamma + 1)) / (gamma + 1)
    if norm <= 0:
        return np.inf
    return -(gamma * np.log(d).sum() - len(d) * np.log(norm))


def fit_exponent(d, dmin, dmax):
    d = d[(d >= dmin) & (d <= dmax)]
    res = minimize_scalar(neg_loglik, bounds=(-6, 2), method='bounded',
                          args=(d, dmin, dmax))
    return res.x, d


def density(x, gamma, dmin, dmax):
    if abs(gamma + 1) < 1e-9:
        norm = np.log(dmax / dmin)
    else:
        norm = (dmax**(gamma + 1) - dmin**(gamma + 1)) / (gamma + 1)
    return x**gamma / norm


if __name__ == "__main__":
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    distances = load(os.path.join(DIR_IN, "dispersal_distances.joblib"))
    pooled = np.concatenate(list(distances.values()))
    dmax = pooled.max()

    gamma, used = fit_exponent(pooled, DMIN, dmax)
    rng = np.random.default_rng(0)
    boot = [fit_exponent(rng.choice(used, size=len(used), replace=True), DMIN, dmax)[0]
            for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"{len(pooled):,} dispersal events, {DMIN}-{dmax:.1f} km")
    print(f"fitted exponent  {gamma:.2f} (95% CI {lo:.2f} to {hi:.2f})")
    print(f"expected under a uniform host distribution: {1 - ALPHA:.2f}")
    print(f"implied kernel alpha {-gamma + 1:.2f} (input {ALPHA})")
    print(f"median {np.median(pooled):.1f} km, "
          f"95th pct {np.percentile(pooled, 95):.1f} km, max {dmax:.1f} km")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    xx = np.logspace(np.log10(DMIN), np.log10(dmax), 300)
    fitted = density(xx, gamma, DMIN, dmax)
    expected = density(xx, 1 - ALPHA, DMIN, dmax)

    # panel a: linear
    ax = axes[0]
    ax.hist(used, bins=np.linspace(DMIN, min(40, dmax), 60), density=True,
            color='0.75', edgecolor='none')
    ax.plot(xx, fitted, color='#1f77b4', lw=1.6,
            label=f'fitted, $d^{{{gamma:.2f}}}$')
    ax.plot(xx, expected, color='#d62728', lw=1.2, ls='--',
            label=f'kernel, $d^{{{1-ALPHA:.1f}}}$')
    ax.set_xlim(0, min(40, dmax))
    ax.set_xlabel('Dispersal distance (km)', fontweight=500)
    ax.set_ylabel('Density', fontweight=500)
    ax.set_title('a.', loc='left', fontweight=700)
    ax.legend(frameon=False, fontsize=9)

    # panel b: log-log
    ax = axes[1]
    bins = np.logspace(np.log10(DMIN), np.log10(dmax), 40)
    ax.hist(used, bins=bins, density=True, color='0.75', edgecolor='none')
    ax.plot(xx, fitted, color='#1f77b4', lw=1.6)
    ax.plot(xx, expected, color='#d62728', lw=1.2, ls='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dispersal distance (km)', fontweight=500)
    ax.set_ylabel('Density', fontweight=500)
    ax.set_title('b.', loc='left', fontweight=700)

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    out = os.path.join(DIR_OUT, "Figure3.png")
    fig.savefig(out, dpi=300)
    print(f"\nSaved {out}")

    # per-sub-landscape version for Fig. S6
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
    for ax, (area, d) in zip(axes.ravel(), sorted(distances.items())):
        ax.hist(d[d >= DMIN], bins=np.linspace(DMIN, min(40, dmax), 40),
                density=True, color='0.75', edgecolor='none')
        g, _ = fit_exponent(d, DMIN, dmax)
        ax.plot(xx, density(xx, g, DMIN, dmax), color='#1f77b4', lw=1.4)
        ax.set_xlim(0, min(40, dmax))
        ax.set_title(f'{area}.  $d^{{{g:.2f}}}$', loc='left', fontweight=700, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('Dispersal distance (km)', fontweight=500)
    for ax in axes[:, 0]:
        ax.set_ylabel('Density', fontweight=500)
    fig.tight_layout()
    out_s = os.path.join(DIR_OUT, "FigureS6.png")
    fig.savefig(out_s, dpi=300)
    print(f"Saved {out_s}")
