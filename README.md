# Finite Expression Method for Turbulent Dynamics with High-Order Moment Recovery

**Two-stage data-driven modeling for turbulent dynamical systems — symbolic drift discovery + generative stochastic correction.**

This repository implements the finite expression framework for turbulent dynamics with high-order moment recovery:

1. **Stage 1 (`1stage_deterministic.py`)** learns a compact symbolic/feature-based drift model (FEX).
2. **Stage 2 (`2stage_stochastic_time_*.py`)** learns stochastic corrections (single-net or ensemble) on top of the deterministic core.

The repo also includes a WSINDy comparison workflow for MC-triad datasets (`comparison_wsindy.py`) and plotting/diagnostics utilities for learned dynamics.

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [WSINDy workflow](#wsindy-workflow)
- [Repository layout](#repository-layout)
- [Dependencies](#dependencies)

## Install

For this repo setup, use the direct config-based install flow:

```bash
cd src
python config.py
```



## Quickstart

### 1) Stage 1: deterministic drift discovery

```bash
cd src
python 1stage_deterministic.py --params_name cascade --DEVICE cpu
```

### 2) Stage 2: stochastic model learning (time-dependent or time-independent)

```bash
cd src
python 2stage_stochastic_time_dependent.py
# or
python 2stage_stochastic_time_independent.py
```

Use the time-dependent or time-independent Stage-2 script according to your final discussion setup.  
The discussion/test run generates the full set of discussion outputs, and the WSINDy workflow below is part of that same evaluation flow.

## WSINDy workflow (included in discussion/evaluation)

`comparison_wsindy.py` fits WSINDy on MC-triad bundles and can produce side-by-side diagnostics and rollout plots.

Example (batch mode, triad-enforced dynamics, with plots):

```bash
cd src
python comparison_wsindy.py --batch --cases equipart cascade --noise_level 1.0 \
  --wsindy_only --no_standardize --triad_interaction_only \
  --plot_wsindy --prediction_t_max 20
```

Common outputs in each case folder:

- `wsindy_fit_comparison.png`
- `wsindy_prediction_extended.png`
- `wsindy_truth_paired_tmax.png` (when `--paired_truth` is enabled)

Useful flags:

- `--all` to run all supported triad cases
- `--paired_truth` and `--paired_seed` for paired-noise validation
- `--stoch_paths`, `--plot_seed`, `--stoch_clip_margin` for rollout diagnostics
- `--no_keep_constant_terms` to drop constant terms in triad-enforced exports

## Repository layout

```text
TurbulentFEX/
├── src/
│   ├── 1stage_deterministic.py
│   ├── 2stage_stochastic_time_dependent.py
│   ├── 2stage_stochastic_time_independent.py
│   ├── discussion_test.py
│   ├── comparison_wsindy.py
│   ├── NN_comparison.py
│   ├── config.py
│   ├── utils/
│   │   ├── FEX.py
│   │   ├── FEX_with_force.py
│   │   ├── ODEParser.py
│   │   ├── wsindy.py
│   │   ├── plot.py
│   │   └── ...
│   └── Example/
│       ├── MC_triad/MC_triad.py
│       └── Tree_structure.svg
└── README.md
```

## Dependencies

Core packages used by the project include:

- `torch`
- `numpy`
- `scipy`
- `matplotlib`
- `sympy`
- `pandas`
- `scikit-learn`

## Notes

- Main experiment data/results are under `src/Example/MC_triad/Results/`.
- For detailed script-level behavior and options, see comments/docstrings in the corresponding files under `src/`.

## Contact

If you have questions, please email:

- `xingjianxu@ufl.edu`
- `qidi@purdue.edu`
- `chunmei.wang@ufl.edu`