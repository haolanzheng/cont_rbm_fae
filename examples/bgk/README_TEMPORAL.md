> Historical experiment notes. For current entry points and data requirements, see the root README and docs/EXPERIMENTS.md. Generated results and caches are not included in the Git repository.

# Temporal mechanism experiment

This package studies whether Adam's endogenous sequence of parameter choices
explains delayed crossover against fixed-pool greedy. It is the established
nonlinear BGK screening model, not a full reproduction of the Boltzmann paper.

## Read results

- `temporal_output/BGK_TEMPORAL_REPORT.md`: interpretation and exact comparisons.
- `temporal_output/BGK_TEMPORAL_SUMMARY.json`: fixed-stratum and trajectory data.
- `temporal_output/BGK_TEMPORAL_RESULTS.png`: errors and selection difficulty.
- `temporal_output/BGK_TEMPORAL_MAP.png`: parameter locations and error ratios.
- `temporal_output/*_pool.json`, `*_adam.json`, `old_adam_swapped.json`: pointwise
  errors at every rank, selected parameters and optimizer traces.
- `temporal_output/BGK_TEMPORAL_BRANCH.json`: common-prefix ordering control.
- `temporal_output/BGK_TEMPORAL_QUALITY.json`: numerical checks.
- `.npz` files: snapshots, selected parameters and independent POD reference.

`old_*` means original selected parameters replayed with tighter truth fields.
`new_*` means selection itself was rerun with tighter solves. The fixed pool is
the original pool of 50 parameters. It is not a newly generated random pool.

## Dependencies

Python 3, NumPy, SciPy, Matplotlib, and a C++ compiler. Use one BLAS thread per
worker. All grids are full grids; there is no collocation or hyperreduction.

## Reproduce

Run commands from the directory containing these scripts:

```bash
g++ -O3 -std=c++17 -shared -fPIC bgk_exact_kernel.cpp -o bgk_exact_kernel.so
python bgk_decisive.py init --workers 4
```

The first command builds the exact full-grid residual/Jacobian kernel. The
second regenerates the omitted 384-point truth cache (about 300 MiB) from its
fixed Sobol seed and deterministic boundary points. The decisive-experiment
driver is used here only for this cache and shared helper functions.

To inspect packaged results after generating the cache:

```bash
python summarize_temporal.py
python validate_temporal.py
python refine_temporal_endpoint.py
```

To rerun every temporal calculation without overwriting packaged results:

```bash
mv temporal_output temporal_output_saved
python bgk_temporal.py init
python bgk_temporal.py run --names old_pool old_adam new_pool new_adam old_adam_swapped --workers 4
python bgk_temporal_branch.py
python summarize_temporal.py
python validate_temporal.py
python refine_temporal_endpoint.py
```

`report_temporal.py` combines machine-generated tables with the human-authored
`temporal_output/interpretation.md`; preserve that file when regenerating the
report. The report's interpretation must be reconsidered if any configuration
is changed. Completed run files are reused; partial runs are not resumable by
this driver and are recomputed from the beginning.

## Interpretation limits

The rank-one and independent POD proxies define *fixed* difficulty. They do
not establish a unique intrinsic hardness of a parameter. Typical denotes the
uniform parameter law, not all interior points. The original 20-point test set
is also retained; its maximum must not be substituted for the 256-point maximum.

Reordering the same snapshots changes prefix spaces but preserves the final
space. These are conditional timing interventions, not complete new greedy
policies. The low-difficulty third branch was added exploratorily after the
first two branch results, selected using the two independent difficulty labels.

This deterministic case study establishes neither universal Adam dominance nor
a statistical result over optimizer seeds. With five starts (four corners and
center), the optimizer has no randomized initial points.
