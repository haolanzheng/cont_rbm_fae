Current squared-residual comparison
===================================
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.thermal_block.run_squared_residual
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.thermal_block.report_squared_residual

The current default uses the baseline's matrix-norm normalization rule:
J = r^T X^{-1} r / ell_n, ell_n=||B_n^T X^{-1} B_n||_F,
B_n=[f,-A_1 v_1,...,-A_d v_n]. H_n=B_n^T X^-1 B_n is the fixed Gram
matrix in the exact nonlinear representation J_raw=theta_n^T H_n theta_n.
It is not the parameter-coordinate quadratic matrix of the linear baseline.
The normalizer is recomputed on enrichment and fixed throughout each search.
eta=0.6/sqrt(1+||gradient J||^2), followed by coordinate clipping. Exactly
T steps; return the best visited point. No log, backtracking or early stop.
T=80,160,320,640, eight seeds, d=14, N=24, ten existing settings.
Output: runs/thermal_block_squared_gram_20261103.

THERMAL_NORMALIZATION=source runs the preliminary fixed-source-normalizer
version, ell=f^T X^-1 f, in runs/thermal_block_squared_residual_20261103.
This preliminary version did not use the baseline matrix-norm rule and must
not be called identical normalization. All previous outputs are retained.

SquaredResidualRB implements the squared objective and analytic derivatives.
Historical log scripts below remain to reproduce old results. The comparison
changes objective transform, normalization and line search, and therefore
does not isolate the causal effect of the logarithm alone.

Historical log-residual experiments
==================================
Thermal-block benchmark on a cube
================================
PDE: -div(a(x;z) grad u)=1 in (0,1)^2, u=0 on the boundary.
Partition into 2 x 2 or 2 x 7 rectangular blocks. On block j,
a_j=5.5+4.5 z_j, z in [-1,1]^d. The parameter map is affine;
the solution map is generally nonlinear. No perturbation approximation is used.
Reference: pyMOR thermal-block benchmark,
https://docs.pymor.org/latest/getting_started.html

Discretization: conforming P1 FEM on a block-aligned 28 x 28 square grid,
split into triangles; 729 interior unknowns. Unit forcing is integrated
exactly. X=sum_j A_j is the unit-diffusion stiffness matrix. All coefficients
lie in [1,10], hence X <= A(z) <= 10X in Loewner order.

Reduced model: V^T X V=I, K(z)c=V^T f, K=V^T A(z)V.
Residual r=f-A(z)Vc. Search maximizes J=r^T X^{-1}r.
The bound ||u-u_n||_X <= sqrt(J) follows from A(z)>=X.
The inner GD optimizer uses log(J/(f^T X^{-1}f)) to handle its changing scale,
with a numerical floor at 1e-28 of the reference value. Projected normalized
steps: eta=0.6/sqrt(1+||gradient(log J)||^2), at most 80 iterations, Armijo
coefficient 1e-4, at most 12 halvings per step. One independent uniform start
per greedy round. Return the best visited point. All methods start from the
same snapshot at z=0 and enrich up to N=24.

Methods: GD; fixed nested uniform pools of size 256,4096,10000; a fixed mixed
pool with 2048 interior uniform samples and 2048 random vertices. For d=4
also uniform meshes with h=1 and h=1/2 in z coordinates. Grid runs are
deterministic; other methods have eight seeds, 1000--1007.

Independent shared tests: 512 scrambled Sobol interior samples, 256 face
samples (one coordinate fixed), 256 edge samples (all but one fixed), every
one of the 2^d vertices, plus the center. Thus 1041 entries for d=4 and 17409
for d=14. Unlike an even quadratic objective, opposite vertices cannot be
identified. Duplicated boundary entries do not affect the maximum. No test
point is used for training or optimizer tuning.

Primary error: maximum absolute reference-energy norm over this fixed set.
Secondary: maximum relative error in the same norm. Neither is a certified
continuous-domain supremum. The first metric favors GD over uniform pools in
the tested d=14 setting; the second does not show the same advantage.

Numerics: residuals evaluated using a QR factor of the whitened residual
basis, avoiding cancellation in the expanded Gram quadratic. Truth errors
are evaluated directly as e^T X e. Analytic parameter derivatives are checked
against centered finite differences. The pool cache and single-point formula
are checked against a full residual solve. Training timing includes the
initial snapshot, residual-cache construction, search, truth snapshots and
basis enrichment; excludes common PDE assembly and test truth/evaluation.
Runs use float64 CPU with one BLAS thread. Old timing data before removal of
unused Gram caches are preserved; retiming exactly reproduces all selected
parameters. No slides are generated or changed.

Analytic structure
------------------
Let B_j=V^T A_j V. Then sum_j B_j=I, B_j>=0, and I<=K(z)<=10I.
The reduced coefficients c(z)=K(z)^{-1}f_n are rational and smooth on an open
neighborhood of the cube; dc/dz_j=-4.5 K(z)^{-1}B_j c(z).
For example ||dc/dz_j||_2 <= 4.5 ||f_n||_2. There is no inverse singularity on
the parameter domain. Higher derivatives follow by repeated differentiation.
With theta=(1, kappa_1 c_1,...,kappa_d c_1,...,kappa_d c_n), and
C=[f,-A_1 v_1,...,-A_d v_1,...,-A_d v_n],
r=C theta, J=theta^T (C^T X^{-1} C) theta. The Gram matrix is fixed PSD at a
given greedy round. These identities give a concrete nonperturbative entry
point for analysis, but do not by themselves prove a probability advantage.

Commands (repository root):
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.thermal_block.run --seeds 1000 1001 1002 1003 1004 1005 1006 1007
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.thermal_block.report
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.thermal_block.refinement

Output: runs/thermal_block_cube_20261103.
Refinement audit: lift seed1000 spaces to a nested 56 x 56 grid and recompute
Galerkin solutions on 164 shared witness parameters. This is not full
retraining or a mesh-convergence proof. The HF energy differences (roughly
9% median) indicate that the 28-grid is a pilot discretization; comparative
error ordering on this witness set is preserved after refinement.

GD iteration-budget comparison
------------------------------
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.thermal_block.step_budget
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.thermal_block.audit_step_budget
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.thermal_block.report_step_budget

The budget sweep retrains all eight greedy sequences up to N=24 for
T=160,320,640, retaining the original T=80 and fixed-pool references.
Step size, starting points, stopping tests, mesh and test parameters remain
unchanged. Results are in runs/thermal_block_step_budget_20261103.
Final maximum test errors are distinguished from the residual objective
attained on a fixed reduced space in the separate inner-search audit.
A larger search budget need not improve the final reduced space monotonically.

Unscaled squared-residual comparison
------------------------------------
THERMAL_NORMALIZATION=none PYTHONPATH=src:. .venv_cpu/bin/python -m examples.thermal_block.run_squared_residual
THERMAL_NORMALIZATION=none PYTHONPATH=src:. .venv_cpu/bin/python -m examples.thermal_block.report_squared_residual
PYTHONPATH=src:. .venv_cpu/bin/python -m examples.thermal_block.compare_normalizations

The none option sets ell_n=1 exactly: J=r^T X^-1 r. It changes only the
objective scaling. The baseline step-size rule eta=.6/sqrt(1+||gradient J||^2)
and direct clipping remain unchanged. No log or backtracking. Runs are saved
in runs/thermal_block_squared_unscaled_20261103, preserving all old data.
The comparison script checks identical per-round initial points across all
budgets and normalization choices and reports errors and training seconds.
