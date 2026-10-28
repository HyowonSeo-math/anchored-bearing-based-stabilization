# Anchored Bearing-Based Stabilization

Numerical experiments for *Anchored Bearing-Based Stabilization of Double-Integrator Multi-Agent Systems with Collision Singularities*.

## Requirements

Python 3.13 and the packages in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

## Reproduction

```bash
python anchored_bearing_based_stabilization.py
```

The command generates the manuscript figures and writes numerical results to `results/`.

To generate only the gain-sensitivity figure:

```bash
python anchored_bearing_based_stabilization.py --gain-only
```

Additional numerical diagnostics:

```bash
python anchored_bearing_based_stabilization.py --supplementary
```

Regression tests:

```bash
python -m unittest -v
```

All random initial states use NumPy seed `2026`.

## Figures

| File | Experiment |
|---|---|
| `figures/convergence.pdf` | Noncollision, equal-velocity and transversal collision initializations; inset shows initial separation |
| `figures/rank_comparison.pdf` | Rank-deficient and full-rank sensing graphs |
| `figures/gain_sensitivity.pdf` | Effects of damping and anchor gains |
| `figures/collision_refinement.pdf` | Time-step and regularization sensitivity |
| `figures/collision_energy.pdf` | Energy decay from a collision initial state |

The `--supplementary` flag also generates the initial-state-scale comparison and further diagnostic figures.

## Manuscript source

- `subsection_C_gain_sensitivity.tex`: replacement for Section VI-C.
- `subsection_A_collision_replacement.tex`: replacement for Section VI-A's collision paragraphs and figure.

## Numerical methods

The regularized ODE is integrated using DOP853. An implicit Euler discretization of the original convex subgradient inclusion is solved using ADMM. Numerical parameter values and computed statistics are recorded in `results/metrics.json`. The implicit discretization retains the original nonsmooth potential; numerical refinements do not constitute a proof of convergence to an inclusion solution.
