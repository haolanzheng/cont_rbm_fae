PIECEWISE CONSTANT FORCING: CUBE BASELINE SCREEN (2026-11-11)

PDE: -u''(x;z) = sum_j z_j 1_{((j-1)/d,j/d)}(x), x in (0,1),
zero Dirichlet boundary data, z in [-1,1]^d.

Energy norm: ||u'||_L2. No spatial discretization is used. The exact energy
Gram matrix follows by integrating the Dirichlet Green kernel over two cells.
The residual dual norm equals the Galerkin energy error. B is any square root
of the exact Gram matrix, and Q has orthonormal columns in energy coordinates.

SEARCH SETTINGS
CPU float64; one BLAS thread; seeds 1000--1003. All spaces start at V_0={0}.
Fixed uniform random pools of size 256 and 10000; the smaller is nested.
GD uses one uniform starting point per round and T=80 or 320 updates:
  H = B.T (I-Q Q.T) B; ell = ||H||_F;
  g = 2 H z / ell;
  z_next = clip(z + 0.6 g/sqrt(1+||g||_2^2), -1, 1).
The best visited point is used. The two GD budgets share their initial points.
This update is gradient descent on -J, equivalently ascent on J.

ERROR REPORTING
The common test set combines 3000 independent uniform interior points,
3000 independent random vertices, +/-1, and feasible vertices found from
96 independent starts for each compared space using 100 PSD sign iterations.
Every method is tested on the same union. Test values are LOWER bounds on
true worst error. The separate bound sqrt(d*lambda_max(H)) is a global UPPER
bound. Comparing GD's upper with the pool's lower gives a stronger check
than comparing test errors alone. These analytic formulas are evaluated
in floating point; this is not an interval-arithmetic certificate.

All displayed normalized errors are divided by sup_z ||u(z)|| = 1/sqrt(12).
They are NOT pointwise relative errors or ratios to a Kolmogorov width.
The exact uniform mean squared error equals trace(H)/3.
Same-space enrichment comparisons are also stored; no universal implication
from larger J to a larger one-step decrease of worst error is claimed.

RUN (from repository root, using the existing CPU environment)
  .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing/run_piecewise_forcing.py 256
  .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing/plot_results.py
  .venv_cpu/bin/python examples/baseline/cube_candidates/piecewise_forcing/audit_results.py

Default output: runs/alignment_baseline_20261111/piecewise_forcing.
The default experiment overwrites matching result files when explicitly rerun;
use --output for a separate run. Original completed logs are preserved.
