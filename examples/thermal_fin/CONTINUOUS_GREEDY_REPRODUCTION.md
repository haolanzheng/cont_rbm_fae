> Historical experiment notes. For current entry points and data requirements, see the root README and docs/EXPERIMENTS.md. Generated results and caches are not included in the Git repository.

# Reproducing the September 2026 continuous-greedy study

This is an independent P1 finite-element implementation of the thermal-fin
setting in the supplied `Continuous_RBM.pdf`, with the requested alpha-free
indicator. It is not the original GH200 implementation. The reference mesh,
material/geometry chart, norm, seed conventions and test-set recipe follow the
PDF. Triangle diagonals use SW--NE; the PDF does not specify the original
diagonal convention. The main domain is the full normalized BOX, not the ball.

## Environment

Python 3, NumPy, SciPy, Matplotlib, threadpoolctl. Exact versions used are in
`environment_versions.json`. Use one BLAS thread per process. CPU timing is not
a prediction of GH200 timing.

## Scope

- `(d,N) = (11,50), (21,80), (33,100)`; seeds `11,22,33`.
- One center snapshot plus `N-1` enrichments. One initial point per continuous
  search round, shared across continuous-method variants. No multistart.
- `pool2000`, `pool4000`, `resamp2000`, `lbfgs5`, `lbfgs10`, `lbfgs20`,
  `lbfgs100`, `local`, and legacy `adam100`: 81 primary training runs.
- New local methods use L-BFGS-B box constraints; only legacy Adam uses clipping.
- The local protocol has a 1000-iteration L-BFGS-B cap and same-trajectory SLSQP
  polishing when the terminal projected gradient exceeds 1e-5. Numerical
  stationarity/curvature failures are retained and reported, never discarded.
- Fixed pools cache coefficient-dependent contractions. Resampled pools require
  fresh contractions each round. Timing comparisons in the report use the
  efficient fixed pool, not the more expensive resampled implementation.

## Commands

Run in the extracted archive directory. Generate the omitted, regenerable
full-order held-out arrays (about 369 MB per dimension):

```bash
OPENBLAS_NUM_THREADS=1 python fin_local_study.py --d 11 --task test --out fin_d11
OPENBLAS_NUM_THREADS=1 python fin_local_study.py --d 21 --task test --out fin_d21
OPENBLAS_NUM_THREADS=1 python fin_local_study.py --d 33 --task test --out fin_d33
```

Validate assembly and gradients for each dimension:

```bash
OPENBLAS_NUM_THREADS=1 python fin_local_study.py --d 11 --task validate --out fin_d11
OPENBLAS_NUM_THREADS=1 python fin_local_study.py --d 21 --task validate --out fin_d21
OPENBLAS_NUM_THREADS=1 python fin_local_study.py --d 33 --task validate --out fin_d33
```

An individual training run (overwrites only the corresponding run files):

```bash
OPENBLAS_NUM_THREADS=1 python fin_local_study.py --d 11 --rank 50 --method local --seed 11 --task run --out fin_d11
```

`fin_batch.py` runs all missing main experiments with at most three simultaneous
single-thread subprocesses. It skips final JSON files with the requested rank;
to rerun all runs, use a separate output workspace or remove the relevant final
result JSON files after making a copy. It expects test arrays to exist first.

After training, run the audits, then the timing script with no competing compute
jobs, then aggregate and plot:

```bash
OPENBLAS_NUM_THREADS=1 python fin_local_witness_audit.py
OPENBLAS_NUM_THREADS=1 python fin_isolated_timings.py
OPENBLAS_NUM_THREADS=1 python fin_final_error_audit.py
OPENBLAS_NUM_THREADS=1 python fin_aggregate.py
OPENBLAS_NUM_THREADS=1 python fin_make_figure.py
OPENBLAS_NUM_THREADS=1 python local_separation_theory.py
```

## Included and omitted files

The archive contains all 81 final run JSON files, selected parameters, saved
bases, all nine local-witness audits, test coordinates/norms, direct-error and
gradient validation, sequential timings, analytic-model results, theory notes,
the report and plots. Per-run search timings in the main JSON files came from
concurrent jobs. Use `fin_isolated_timings.json` for the timing table.

Intermediate checkpoints, duplicate training-only JSON files, redundant timing
bases and the three large test-solution arrays are omitted. Basis files and
test coordinates are sufficient to recompute all reported errors; the omitted
arrays follow deterministically from the documented full-order solves.

## Interpretation

The thermal-fin maximum is over a fixed 4608-point boundary-enriched test set,
not a mathematical supremum. Mean and p99 use only its 4096 uniform Sobol
points. Hessian/KKT checks are numerical diagnostics, not interval-certified
proofs. The analytic Poisson example is a separate, deliberately constructed,
rank-three solution family with exact full-domain supremum formulas.
