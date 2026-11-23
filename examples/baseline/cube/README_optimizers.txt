CUBE BASELINE: OPTIMIZER COMPARISON

The problem, FEM coordinate matrices, normalizer, enrichment, and test parameters
are the existing consistent_fem_20261113 baseline. No slides or old data are edited.

Main comparators
  GD: existing normalized projected updates, alpha=0.6, T=80.
  SQP: SciPy SLSQP, analytic gradient, ftol=1e-9, maxiter=500.
  Reflective trust region: Fides 0.7.8, Coleman-Li scaling and reflection,
    two-dimensional Newton/gradient subspace, analytic gradient and Hessian. fatol=frtol=xtol=1e-10,
    gatol=1e-8, maxiter=500.
Supplementary comparator
  SciPy trust-constr: interior-point trust region, analytic gradient/Hessian,
    gtol=1e-12, xtol=barrier_tol=1e-10, maxiter=500.

These are Python implementations on the common objective -J_n. They are NOT full
replications of the output-oriented objectives, initialization rules and entire
algorithms in Urban et al. (2014) or Bui-Thanh et al. (2008). In particular, a
fixed uniform pool of 256 points replaces the exponentially large coarse grid.
SLSQP is not claimed to be identical to MATLAB fmincon's SQP implementation.

All methods use one CPU thread. Every method and initialization combination has
eight runs, with seeds 1000,...,1007 and d=4,8,16,32,64,128. One random uniform
start is compared with the best of 256 uniform pool points (pool reused over
rounds). Pool generation and scoring are timed. No extra random start is taken
for pool initialization. Each method builds its own reduced space.

Final tables use N=d/2; curves extend to d-1 for d=16,32,64. Test points are loaded
unchanged from the old baseline. They contain all antipodal vertex representatives
for d<=17, and 65536 distinct representatives for larger d. Worst errors are test
maxima, not certified global maxima in high dimensions.

Timing includes objective-matrix construction, initialization, optimization,
enrichment and small solver bookkeeping. HF setup, test-error evaluation and file
I/O are excluded. Budgets are NOT equal iteration counts or equal time; function,
gradient/Hessian evaluations, actual iterations and termination status are logged.

The frozen-space study uses the saved GD space at n=d/2-1 for seed1000; every
optimizer receives the same H and start for a paired comparison. It separates
inner-search quality from the consequences of constructing different spaces.

Reproduce from repository root (with NumPy, SciPy, threadpoolctl, Matplotlib and
fides==0.7.8 available):
  PYTHONPATH=. .venv_cpu/bin/python -u -m examples.baseline.cube.compare_optimizers
  PYTHONPATH=. .venv_cpu/bin/python -u -m examples.baseline.cube.optimizer_frozen_study
  PYTHONPATH=. .venv_cpu/bin/python -u -m examples.baseline.cube.report_optimizers

This run installed only Fides into runs/cube_optimizer_comparison_20261115/python_deps.
To reuse that installation, prepend that folder to PYTHONPATH, followed by ':.'
All output is in runs/cube_optimizer_comparison_20261115. Completed jobs are
checkpointed and skipped on restart. Do not change tolerances when resuming an
existing result directory; use a fresh run directory for a different setting.

References
  Bui-Thanh, Willcox and Ghattas (2008), doi:10.1137/070694855, section 3.4.
  Urban, Volkwein and Zeeb (2014), doi:10.1007/978-3-319-02090-7_5, section 5.
  Fides, https://github.com/fides-dev/fides ; Coleman-Li (1994, 1996).

Counter note: h_actions counts objective H-times-vector evaluations only; it does
not count internal Hessian products/factorizations of the Newton or SQP subproblem.
Use the saved time and separate function/derivative counts for cost comparisons.
