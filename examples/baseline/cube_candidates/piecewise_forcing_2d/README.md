# Two-dimensional Poisson with blockwise source amplitudes

On the fixed physical square (0,1)^2, solve -Delta u = sum z_j 1_cell_j
with homogeneous Dirichlet data. There are m by m equal square cells,
d=m^2 independent amplitudes in [-1,1], and no source normalization or
prescribed modal weights. The norm is the Dirichlet energy norm.

The operator is fixed, so Galerkin energy error equals the dual residual
norm. Both search and reported error use this same norm. Reported errors
are divided by the energy norm of the solution to the constant source f=1.
This reporting normalization does not change the source or optimization.

## Numerical model and error control

The cell-load energy Gram is computed using the exact sine integrals on
each cell. On a uniform partition, integrated sine modes alias into an
orthogonal discrete sine basis; the two-dimensional Gram is diagonal in
its tensor product. The script sums physical sine frequencies k,l<=L.

Parseval gives an explicit upper bound on each omitted Gram eigenvalue.
The reported epsilon satisfies G_L <= G_exact <= (1+epsilon) G_L.
Taking the minimum over reduced coefficients gives the same multiplicative
bound for squared Galerkin errors for any fixed snapshot parameter set.
The script also computes the Gram with cutoff 2L for verification.
Analytic bound formulas are evaluated in float64, not interval arithmetic.

## Search and evaluation

CPU, float64, one BLAS thread; four seeds 1000--1003. All spaces start empty.
Each GD greedy round starts from one independent uniform parameter and uses
80 projected gradient steps, alpha=.6, with the same Frobenius normalizer
and 1/sqrt(1+gradient_norm_squared) step factor as the one-dimensional test.
Random pools are fixed, uniform, and nested with sizes 256,4096,10000.

Every space in one run is evaluated on the same union of uniform interior
points, random vertices, constant-sign vertices, and independently searched
vertices for every compared space. For d=16, all 65536 vertices are included;
convexity of the quadratic error makes this a global search for the spectral
model. For d>16, reported test maxima are lower bounds on global maxima.
The additional bound sqrt(d*lambda_max(H)) gives a global upper bound.
The full-spectrum correction sqrt(1+epsilon) can be applied to that upper
bound and compared against a pool-space tested lower bound.

## Recorded runs

Results and logs are in runs/piecewise_forcing_2d_20261112.
The first run used N<=d/2; extended_N retained the first run and extended
to N<=7d/8. Different test unions may give slightly different maxima at
shared N; compare methods within the same run.

From the repository root:

    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/run.py
    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/run.py --max-fraction .875 --output runs/piecewise_forcing_2d_20261112/extended_N
    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/audit.py
    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/same_space_gain.py

The existing output files are overwritten if these commands are rerun with
the same output directory. Use a new --output directory to preserve a run.

### Per-snapshot error reduction through N=d-1

`enrichment_study.py` continues the saved `dimension_study` prefixes for
d=16,24,32,48,64 through N=d-1, preserving the GD initialization stream.
GD uses T=80 and alpha=.6; the three fixed nested pools and four seeds are
unchanged. Results, figures, snapshot parameters, and the shared test set
are saved in `runs/piecewise_forcing_2d_20261112/enrichment_study`.

For each seed, the experiment records E_N/E_0, (E_{N-1}-E_N)/E_0, and
(E_{N-1}-E_N)/E_{N-1}. Figures show medians after computing each seed's
reductions. The methods follow their own greedy paths; these differences
are not comparisons from a common intermediate V_n. The test union is
expanded for the additional ranks and all previous ranks are reevaluated.

At N=d-1, let w be the unit vector perpendicular to the reduced solution
space in the energy coordinates B. Then J(z)=((B^T w)^T z)^2, so the exact
cube worst error of the spectral model is ||B^T w||_1. Its maximizing sign
vector is included in the common test set and direct residual evaluation
checks this formula. At other ranks, d16 is globally evaluated by complete
vertex enumeration; the higher-dimensional curves report test maxima.

From the repository root:

    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/enrichment_study.py

`benchmark_enrichment.py` times the complete training again from V_0,
without reusing the saved prefixes. It validates every resulting parameter
path and final error against the enrichment study before timing. Each of
the four seeds is repeated nine times in randomized method/seed order on
one CPU BLAS thread. `timing.json`, `timing.csv`, and `timing.log` are saved
beside the enrichment results. The summary uses all 36 training timings
per method and dimension. Included costs are sampling, method-specific
precomputation, search, score updates, and snapshot enrichment. Shared B/G
construction is timed separately; testing, plotting, validation, and I/O
are excluded. These are timings of the compressed spectral implementation,
whose snapshot computation is Bz rather than a new PDE linear solve.

    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/benchmark_enrichment.py

### Inner convergence and enrichment from identical reduced spaces

`convergence_study.py` retains the old GD80 and pool trajectories and trains
a new trajectory with the same per-round random starts and step rule until
the projected update reaches a fixed point. It checks the normalized
projected-gradient residual and continues 100 more steps as a stability
check. All 716 rounds in the recorded run reached an exact floating-point
fixed point, with no change in the extra 100 steps. This does not assert
global optimality of the quadratic maximization.

All trajectories are evaluated on a common enlarged test set. Additionally,
at every space on the new trajectory, three candidate snapshots (GD80,
converged GD, pool10000) are separately added to that identical space.
Their errors are measured on the same test set; matched-space candidate
searches contribute equally to that set. Two d16 cases are independently
checked using QR reconstruction and complete vertex enumeration, including
verification of the selected objective against its global maximum.
Results are in `runs/piecewise_forcing_2d_20261112/convergence_study`.

    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/convergence_study.py

### Square partitions through d=256 at N=floor(d/2)

`square_dimension_study.py` uses m=5,...,16, d=m^2, N=floor(d/2), and
the original T=80, alpha=.6 rule with four seeds. For odd square dimensions
the final reduced dimension is rounded down. Existing d64 and d256
snapshot prefixes are preserved; new rounds consume the same initialization
stream. No source normalization, optimizer, or PDE change is introduced.

For each d, all methods and seeds share 4096 uniform interior points,
4096 random vertices, constant-sign vertices, signs of leading residual
eigenvectors, and two independent 128-start vertex-search batches for each
final space. The second batch's effect on the measured maximum is recorded.
These are tested worst errors, not certified global maxima. The error is
normalized by the global initial maximum E0, the constant-load energy norm.
The reported curves and tables use the median over the four seeds.
Results and the run log are in
`runs/piecewise_forcing_2d_20261112/square_dimension_study`.

    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/square_dimension_study.py

## Interpretation

### Natural source and operator variants

`natural_variants.py` compares d=64,144 at N=d/2, T=80 and alpha=.6.
The original Poisson/block-source case is compared with constant
anisotropy kappa=4,16, interior-node bilinear source interpolation, and
a reaction coefficient of 100 as a contrasting case. Sources remain
linear in d independent cube parameters. The objective is the exact
squared error in each operator's own energy norm. Interior hat sources
vanish at the boundary; all-ones forcing is therefore not constant, and
E0 is recomputed for each family.

All four seeds use pools of 10000 and 100000 points and the original
problem's time-matched size. These are fixed-size comparisons, not a new
equal-time calibration. Direct Fourier Gram construction and independent
hat-function quadrature validate the model; tail bounds and doubled
cutoff checks are saved. All compared methods share a test union per case.
Results are in `runs/piecewise_forcing_2d_20261112/natural_variants`.

`natural_variant_directions.py` compares GD and independent fresh pools
on the same saved intermediate space for the original and bilinear-source
families, using the exact mean-square error reduction of each snapshot.

    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/natural_variants.py
    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/natural_variant_directions.py

### Separate comparison at equal training time

`equal_time_pool.py` compares d=25,64,144,256 at N=floor(d/2), retaining
GD T=80 and alpha=.6. Nested pools contain 256,4096,10000,25000,50000,
100000 points, plus one size calibrated using training time only. The
comparison pool is the largest measured size whose median wall time does
not exceed GD's. CPU time is recorded as a cross-check.

Each method is retrained from V0 on four seeds, with five timing repeats
in randomized sequential order after warmup. Sampling, pool precomputation,
cached score updates, GD search and enrichment are included; shared model
construction, test evaluation and I/O are excluded. Existing parameter
paths are checked against the saved square-dimension study. All spaces
are reevaluated on a common expanded test set containing the previous
tests and two independent vertex-search batches for every new space.

Results, timings, snapshot parameters, test points and the error/time plot
are in `runs/piecewise_forcing_2d_20261112/equal_time_pool`.

    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/equal_time_pool.py

### Earlier fixed-pool comparisons

The screening result is mixed: d=16 does not show consistent GD superiority;
d=64 shows a moderate improvement; d=256 shows a clearer improvement over
the 10000-point pool. Absolute worst errors decrease more slowly than in
the one-dimensional block-source problem. This is not an equal-wall-time
comparison or a theorem ordering the single-step benefits of all candidates.

At the first greedy round, positivity of the Green function makes constant
sign loads global maximizers of J. Independently computed mean-error gains
for those snapshots are about 95.5--95.7 percent of the best possible single
snapshot mean gain. This percentage is not the fraction of total error
removed and does not imply a per-round worst-error guarantee.

The independent fixed-space check uses 20 saved pool-trained spaces, with a
fresh 10000-point pool and one fresh GD80 start in each. GD finds a larger J
in 20/20 cases and gives a larger mean-square error decrease in 19/20.
The one counterexample is retained: d64,N32,seed1002 has a J ratio of about
3.156 but a mean-gain ratio of .930. For d256, mean gain improves in all 12
screened spaces, with a median ratio of 1.563. These are finite-sample
observations, not a universal ordering theorem.

### Cached gradient implementation (2026-11-16)

`run.ascend` reuses `H @ z` for both the objective value and the next gradient.
It performs `T+1` matrix-vector products instead of `2T+1`, and no longer
allocates `2*H` at every step. The step rule, clipping, normalizer, number of
updates, and best-iterate selection are unchanged. Residual-matrix assembly
retains its original formula to preserve the saved greedy trajectories.

`validate_cached_gradient.py` compares the implementation with the original
update on zero, isotropic, dense and ill-conditioned matrices, and checks
complete saved GD paths and common-test error curves across all slide dimensions.
`benchmark_slides.py --output <new-directory>` repeats full training timings;
choose a new directory so earlier timing logs are preserved.

Validation and timings for this change are retained in
`runs/piecewise_forcing_2d_20261112/slides_summary/optimized_gd_20261116`.
The current slides report absolute energy errors on the common 4096-interior
plus 4096-vertex test sets recorded by `slides_summary.py`; earlier studies
above used their separately documented error normalization and test unions.

### Pool size 40000 comparison (2026-11-16)

`compare_pool_size.py` compares optimized GD80, pool10000 and pool40000 at
N=floor(d/2), with the same four seeds and unchanged 8192-point slide test
sets. The smaller pool is a prefix of the larger pool. Each method builds
its own reduced space. All three methods are re-timed in randomized order
(five repeats per seed); testing and validation are excluded from timing.
Complete records, snapshot parameters and a CSV summary are retained in
`runs/piecewise_forcing_2d_20261112/slides_summary/pool40000_20261116`.

    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/compare_pool_size.py --sizes 10000 40000 --output <new-output-directory>

### Step-budget study: T=40 (2026-11-16)

`rerun_slides.py` retrains GD with an explicit step budget, all three slide
pools and the d=4 uniform meshes. The additional 40000-point pool is rerun at
the seven tabulated dimensions. It preserves alpha=0.6, seeds 1000--1003,
the saved 8192-point test sets, and each original error-curve rank range.
Final errors and full-training timings use N=floor(d/2). Timings use five
randomly ordered repeats per seed on one CPU BLAS thread, excluding shared
model construction, tests and diagnostics. Pool and mesh error curves are
checked against the earlier results.

Both source-history figures use the first eight snapshots of the new GD40
trajectory, followed by its ninth initialization. Every candidate is added
independently to this same V8. The one-enrichment error identity is verified
against direct projection on the common test set.

    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/rerun_slides.py --steps 40 --output runs/piecewise_forcing_2d_20261112/slides_summary/gd40_20261116

The output includes all raw paths, validation records, timings, CSV summaries
and the four slide figures. Use `--figures-only` to regenerate figures from
this run without repeating the timings. Earlier T=80 and T=20 logs remain
unchanged; older study scripts retain their historical defaults.

The slides were subsequently restored to the T=80 results and original T=80
fixed-space source/enrichment figures. The complete T=40 study is retained
in its output directory for comparison.

### Error decay through N=d-1 (2026-11-16)

`full_rank_decay.py` extends GD80 and pools 256/4096/10000 through N=d-1
for all 14 slide dimensions, with the same seeds, alpha and saved test sets.
Uniform meshes are included at d=4. Every existing error-curve prefix is
checked, and each GD snapshot prefix is required to match exactly.
Errors use orthogonal-complement coordinates, avoiding subtraction of nearly
equal squared norms near full rank; direct projection checks are also saved.
Interval-wise power and exponential fits are descriptive finite-range fits.

    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/full_rank_decay.py --output runs/piecewise_forcing_2d_20261112/slides_summary/full_rank_20261116

The output preserves per-seed curves, snapshots, validation records, fitted
exponents, checkpoint tables, and representative/all-dimension PDF plots.
These curves remain test-set maxima, not certified global worst errors.

### Projected Adam comparison (2026-11-17)

`adam_study.py` compares GD80 with projected Adam80 at learning rates
0.01, 0.03, 0.1, 0.3 and 1. Adam uses beta1=0.9, beta2=0.999, epsilon=1e-8,
bias correction, zero initial moments each greedy round, and no weight decay.
The gradient is that of J=z^T H z / ||H||_F; no extra gradient normalization
is applied to Adam. Both methods return the best encountered feasible iterate.

Dimensions are 4,16,25,64,100,144,256, with four matched seeds and the unchanged
8192-point tests. Curves extend through N=d-1. Five-repeat full-training timings
end at N=floor(d/2), in randomized order after warmup on one CPU thread.
Fixed-space searches at n=floor(d/2)-1 use the same GD-generated space and
initial point for every optimizer, separately from the full greedy comparison.

    .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing_2d/adam_study.py --output runs/piecewise_forcing_2d_20261112/slides_summary/adam_20261117

This is an exploratory learning-rate sweep: all rates are reported, without
claiming a held-out evaluation of a tuned Adam configuration. Existing slides
and historical experiments are unchanged. Raw paths, validation, timings,
per-seed errors and fixed-space objectives are retained in the output folder.
