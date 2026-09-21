# Simplicial contagion on random complexes

Reproducibility package for:

> **Higher-order contagion on random simplicial complexes: critical-mass transitions, structural robustness, and cascading-failure control**
> Dong Li, Chaos, Solitons & Fractals (submitted).

## What this repository contains

| Script | Reproduces |
|---|---|
| `simplicial_contagion.py` | Core agent-based simulator (SIS on pairwise + simplex channels) |
| `run_reproduction.py` | Fig. 1, Fig. 2(a), Table 1 — reproduction of Iacopini et al. (2019) |
| `finite_size_scaling.py` | Fig. 2(b), Table S1 — finite-size critical mass |
| `structural_scan.py` | Table 3, Table S2, Fig. 3(a) — heterogeneity and overlap |
| `pair_approx.py` | Fig. 3(b) — pair-approximation estimate |
| `intervention_scan.py` | Table 5, Table S3, Fig. 5(a) — targeted hardening on IEEE 118 |
| `recovery_scan.py` | Table 6, Table S4, Fig. 5(b) — state-dependent recovery coupling |
| `survival_ramp.py` | Table 2 — survival probability vs seed fraction |
| `cascade_infrastructure.py` | Table 4, Fig. 4(b,c) — cascading failures on power grid |
| `make_combined_figures.py` / `vector_figures.py` | Vector figure generation |

## Requirements

- Python 3.8+
- numpy, scipy, matplotlib, networkx
- (optional) MATPOWER `case118.m` for the power-grid example

## Reproduce

```bash
pip install numpy scipy matplotlib networkx
python run_reproduction.py
python finite_size_scaling.py
python structural_scan.py
python intervention_scan.py
python recovery_scan.py
```

All runs complete on an ordinary laptop within minutes to hours.

## License

MIT.
