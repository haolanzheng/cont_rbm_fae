# Analytic diffusion baseline

Three independent parameter domains share the same diffusion model and reduced model:

| Domain | Entry point | Update |
|---|---|---|
| Unit sphere | `sphere/run.py` | Tangent gradient followed along a great circle |
| Unit ball | `ball/run.py` | Normalized gradient step and radial projection |
| Cube `[-1,1]^d` | `cube/run.py` | Normalized gradient step and coordinate clipping |

## Run

From the project root, with the CPU environment installed:

```bash
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.baseline.sphere.run --output runs/sphere_new
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.baseline.ball.run --output runs/ball_new
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.baseline.cube.run --output runs/cube_new
```

Each domain has an explicit `config.json`; pass `--config PATH` to override it.
Use a new output directory for each run. Existing results are never overwritten.
`--model-cache PATH` optionally reuses assembled model matrices. The default cache
is `runs/consistent_fem_20261113`. Missing models are assembled by
`consistent_fem.py`: retain the weak-form loads B and solve AS=B for the HF
snapshots. The current indicator uses the diffusion-energy dual residual norm, R.T A^{-1} R. With AS=B, it equals the squared HF energy error. Set `residual_norm` to `reference_laplacian` only to reproduce the older norm. Caches
with an inconsistent sampled-analytic HF solution are rejected.
The model remains rho=0.78, contrast=0.25, P3 grid40. Grid and spectrum changes
require rebuilding the model.

### Current energy-dual results (November 20)

```bash
PYTHONPATH=. .venv_cpu/bin/python -u -m examples.baseline.common.run_energy_residual
PYTHONPATH=. .venv_cpu/bin/python -u -m examples.baseline.common.validate_energy_residual
PYTHONPATH=. .venv_cpu/bin/python -m examples.baseline.common.report_energy_residual
PYTHONPATH=. .venv_cpu/bin/python -m examples.baseline.common.sync_energy_documents
```

Results are saved under `runs/energy_residual_20261120`. The HF models and test
points are reused from `runs/consistent_fem_20261113`; only the residual metric
changes. In A-orthonormal solution coordinates the new metric is the identity.
The runner preserves seeds, pools, initializations and updates. Timings are
repeated three times per seed in a separate pass, excluding diagnostics.
The report updates `slides_short` and `slides_meeting_20261113`; the final
command synchronizes the writeup and notebook tables. Historical run files
and the previous reference-norm runners remain available for reproduction.

### Cached-pool timing alignment (November 20)

```bash
PYTHONPATH=. .venv_cpu/bin/python -u -m examples.baseline.common.benchmark_cached_pool --output runs/energy_cached_pool_20261120
```

The current energy-dual runner and the domain entry points cache each fixed
pool's energy-coordinate solutions. After enrichment, candidate squared errors
are updated by subtracting the squared coefficient along the new orthonormal
basis vector. Periodic residual refreshes prevent cancellation at tiny errors;
refreshes are included in training time. No pool-specific search matrix is
assembled. Reference-Laplacian runs retain direct scoring because the energy
projection identity does not apply to that residual metric.

The aligned benchmark validates all saved greedy histories before timing,
then uses five randomized, sequential repeats per seed on one CPU/BLAS thread.
Candidate generation, pool precomputation, score maintenance, search, and basis
enrichment are included. Shared HF construction, tests and diagnostics are
excluded. New logs are separate from the historical measurements. The reporter
`common/report_cached_pool.py` updates only timings and the associated cost
explanation in the current writeup, short/meeting slides and notebook.

## Organization

- `common/experiment.py`: shared greedy loop, timing, fixed test sets and saved bases.
- `common/box_bounds.py`: vertex maximization bounds and cube width bounds.
- `cube/search.py`: cube update, independent of probability-proof experiments.
- `geometry_cpu.py`: sphere/ball updates and coefficient-space Galerkin algebra.
- `consistent_fem.py`, `continuous_fem.py`: consistent HF solves and spatial assembly.
- `analytic_model.py`: archived sampled-analytic model, retained for reproducibility.
- `cube/report.py`: cube result figures and a Beamer fragment, generated from saved runs.
- `sphere/`, `ball/`, `cube/`: separate reproducible configurations and entry points.

Defaults: one CPU thread, float64, alpha=0.6, 40 updates for sphere/ball and
80 updates for cube, one uniform initial
parameter per greedy round, eight seeds, fixed pools of 256/4096/10000, N=d/2.
Each method constructs its own reduced space. All methods at a given dimension
share fixed test sets. Sphere/ball retain their 16384 uniform points. Cube uses
only vertices: up to 65536 representatives modulo the symmetry z -> -z.
Tables and curves now report maximum absolute energy error, not mean or relative error.
Cube grids are included only at d=4,6.
The shared runner currently supplies tensor grids only for the cube; historical
sphere/ball angular-grid comparisons use `run_analytic_baseline.py` and remain
unchanged. The new sphere/ball entry points reproduce GD and random-pool studies.

## Evaluation

Training time includes candidate generation, residual-matrix construction,
search and enrichment. It excludes model assembly, test evaluation and
post-training maximum/width bounds. Do not compare wall times across machines.

Cube tables report the maximum absolute energy error on a shared vertex test
set, divided by exact analytic width sqrt(sum_{j=N+1}^d rho^{2(j-1)}).
Fix z_1=+1 because the squared error is even. Enumerate all 2^(d-1) representatives
when this is at most 65536; otherwise sample 65536 distinct representatives
uniformly without replacement. A sampled maximum is not a certified supremum.
Global error and sampled-model width bounds remain separately in saved logs.

All curves report max_z ||S z-u_N(z)||_A on the fixed test set, followed by the
median over training seeds. Cube tables divide the same final maximum by the
exact analytic width. Vertex sampling uses seed test_seed+10000000+d. The same
vertices are reused for all N, methods and seeds. Training sampling is unchanged.
Previous Sobol/uniform/mixed mean-error results are archived, not used in current figures.

To reevaluate saved cube bases without retraining:

```bash
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.baseline.cube.evaluate_test_set --run runs/cube_slides_a06_t80_20261102
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.baseline.cube.report --run runs/cube_slides_a06_t80_20261102 --figures paper/slides_short/figs
```

The empirical attainment plot uses the final selection that builds V_N, evaluated
on V_(N-1), at gamma=0.9. It summarizes eight full greedy runs, with each method's
own reduced space; it is not a fixed-V_n Monte Carlo estimate. Unresolved threshold
classifications are displayed as intervals. Test curves report the maximum absolute energy error, followed by the median over seeds.

All run configuration, selected parameters, basis coefficients, common test
parameters and evaluation bounds are saved under the requested `runs/` directory.
Figures go to the explicitly selected figure directory. Historical research
scripts are retained for reproducibility; see `HISTORICAL_RUNS.md`.

To regenerate maximum-error figures for all three domains from saved runs:

```bash
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.baseline.common.maximum_error_report --source runs/analytic_hf_a06_t40_20261022 --cube runs/cube_slides_a06_t80_20261102 --figures paper/slides_short/figs
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.baseline.cube.report --run runs/cube_slides_a06_t80_20261102 --figures paper/slides_short/figs
```

## Dimension sweep

`common/dimension_sweep.py` adds d=12,20,24,28,36,40,44,52,56,60 to the
previous d=8,16,32,48,64 studies, giving d=8,12,...,64 at N=d/2. Each added
case uses eight seeds, alpha=0.6, T=40 (sphere/ball) or T=80 (cube), and the same
pool sizes. Model matrices are leading blocks of the existing nested 64-mode
family; agreement with independent stored builds is checked in validation.json.
Only final error is evaluated in this sweep; training and test evaluation are
separate. Existing dimensions and training logs are not overwritten.

```bash
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.baseline.common.dimension_sweep --source runs/analytic_hf_a06_t40_20261022 --output runs/baseline_dimension_sweep_20261102
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.baseline.common.plot_dimension_sweep --source runs/analytic_hf_a06_t40_20261022 --cube runs/cube_slides_a06_t80_20261102 --sweep runs/baseline_dimension_sweep_20261102 --figures paper/slides_short/figs
```

The sweep resumes completed records. Ratio plots use the empirical test maximum
and exact analytic width in every domain; these are not certified global-error
ratios. The cube final-error table retains only the five original dimensions.
