Cube baseline candidates -- 2026-11-04
======================================

Purpose
-------
Find an analytically accessible cube-domain PDE for which continuous search
improves the reduced-space error, rather than only the selected indicator.
All models use exact modal/slope representations. No FEM high-fidelity solve
is performed. Code belongs here; all results and logs are in
runs/cube_candidate_screen_20261104. Existing slides and TeX were not edited.

Screened families
-----------------
1. Hadamard-rotated linear forcing: no consistent full-RB error advantage
   over a large uniform pool. Do not infer Kolmogorov width from modal error.
2. Paired-layer coefficient diffusion: some improvement, but not stable
   across contrasts and random seeds. Not selected.
3. Exponential modal forcing: no full-RB advantage in the screened range.
4. Odd rational modal forcing: GD captures pure modes, but mixed pool spaces
   can be equally good or better. Not selected under the geometric spectrum.
5. Common rational amplitude: positive full-RB result; simplest candidate.
6. Rational ridge amplitudes with ordinary forcing background: positive
   full-RB result, several localized parameter regions; more constructed.

Selected candidate: common rational amplitude
--------------------------------------------
Let psi_j(x)=sqrt(2) sin(j pi x)/(j pi), 0<x<1. These functions are
orthonormal in the reference H^1_0 energy inner product. For z in [-1,1]^d,

  a(z)=1+delta-mean(z),       b_j=rho^(j-1)/sqrt(d),
  f_z=sum_j b_j z_j (-psi_j''),
  -a(z) u''=f_z,             u(0)=u(1)=0,
  u(x;z)=sum_j b_j z_j psi_j(x)/a(z).

The coefficient is positive, delta <= a(z) <= 2+delta. Equivalently one can
keep the Laplacian fixed and use the nonlinear forcing f_z/a(z).

The reported error is ||u-u_n||_{H^1_0}, not the parameter-dependent a-energy
norm. For the scalar-coefficient PDE it equals ||r_n||_{H^-1}/a(z). For the
equivalent fixed Laplacian it is the ordinary dual residual norm. Thus the
search criterion equals the error being evaluated; no residual/error mismatch
is introduced.

With B=diag(b_j), Q an energy-orthonormal reduced basis,

  H_n=B(I-QQ^T)B,
  J_n(z)=z^T H_n z/a(z)^2,
  grad J_n=2 H_n z/a(z)^2 + 2 J_n(z) 1/[d a(z)].

The actual update is unchanged in form:
  ell_n=||H_n||_F,
  g_t=grad J_n(z_t)/ell_n,
  z_(t+1)=clip(z_t+alpha*g_t/sqrt(1+||g_t||^2),-1,1).
Keep the best point among t=0,...,T. Each greedy round starts independently
from one uniform point. All methods start with the empty reduced space.

Main follow-up settings and results
----------------------------------
rho=.9, delta=.01 (coefficient contrast 201), alpha=.6, T=80,
d=16,32,64, N=d/2, eight seeds 1000,...,1007. CPU float64, single BLAS
thread. Uniform pools are fixed across greedy rounds and nested across M.
Medians of final maximum reference-energy error:

 d    N       GD          pool256      pool4096     pool10000
16    8    1.823237     17.399382       9.178311      8.088341
32   16     .885931      8.471145       5.718994      4.283999
64   32     .197019      1.759958       1.168632      1.014094

Pool10000/GD error ratios: 4.44, 4.84, 5.15.
At delta=.02 (contrast 101), corresponding TEST-error ratios are
2.43, 2.25, 3.15. Do not call all delta=.02 d64 values certified global errors.

Training times to N=d/2 (medians in milliseconds, testing excluded):
 d       GD          pool10000
16       6.59          2.89
32      13.43          8.62
64      29.80         29.86
These are short local CPU timings; they do not establish a universal speed
advantage. The clear observed improvement is in reduced-space error.

Validation and global maximum check
----------------------------------
Gradient finite-difference relative error: 1.96e-10. Basis orthogonality
defects are about 1e-15. Tests are common across methods, containing Sobol
interior points, vertices, and a common union of independent multistart
searches on EVERY final space. The latter did not increase the maxima.

There is also a stronger analytic check. If z=sum_v lambda_v v is a convex
combination of cube vertices, then
  u(z)=sum_v [lambda_v a(v)/a(z)] u(v).
The weights are nonnegative and sum to one. The projected residual norm is
convex in u, so its global maximum occurs at a vertex, for every fixed Q.

d16: every one of the 65536 vertices is evaluated.
d32/64: every vertex with k<=3 negative coordinates is evaluated. Put
  A=(I-QQ^T)B, e=||A1||, s=||A||_2.
For every vertex with k negative coordinates,
  ||A v||/a(v) <= (e+2 sqrt(k) s)/(delta+2k/d).
At delta=.01 this bound for all k>=4 lies below the evaluated maximum in
ALL 192 final spaces (two rho choices, three d, four methods, eight seeds).
The largest upper-bound/evaluated-maximum ratio is .889745. The checks use
float64 with a roundoff allowance, not outward-rounded interval arithmetic.
This supplies a substantial margin supporting the global maxima in the main
table. See collective_vertex_certificates.json.

Why the new geometry matters
---------------------------
At z=1, u=B1/delta is a large solution in a small corner region. In all main
pool10000 runs the worst remaining error is precisely at this corner. GD
captures this direction accurately: corner error / maximum error is about
zero at d16/32 and at most .008 at d64. Many d64 runs do not select the corner
exactly, so exact corner acquisition must not be asserted for every run.

If B1 lies in the reduced space, the corner error vanishes. Every other
vertex has a(z)>=2/d, so its error is bounded by d||B1||/2 independently of
delta. Conversely, for a fixed finite uniform-interior pool and N<d, almost
surely none of the finitely many N-snapshot spans contains B1. The minimum
distance c_pool from B1 to these spans is positive. Every such pool space
therefore has maximum error >=c_pool/delta. This explains why decreasing
delta can separate full-RB errors. It is not a finite-T convergence theorem
for GD, and does not estimate c_pool quantitatively.

After exact capture, H_n 1=0. Feasible common translations then satisfy
  J_n(z+c1)=J_n(z) [a(z)/(a(z)-c)]^2.
This is useful structure for further probability analysis; no complete
finite-T q_T bound or exact Kolmogorov width is claimed in these experiments.

The mechanism is principally one rare high-amplitude direction. This is
appropriate for a diagnostic baseline, but is not yet a result for arbitrary
nonlinear parameterizations or several unrelated difficult regions.

Other positive candidate
------------------------
The ridge-background model has fixed Poisson operator, d+4 orthonormal
solution modes, ordinary coefficients beta*z/sqrt(d), and four additional
coefficients h(s_j)=eps/(1+eps-s_j), s_j=v_j^T z/d for Hadamard sign rows v_j.
Settings: beta=.05, eps=.1, initial true center snapshot (counts as N=1),
N=8, T=80, alpha=.6, eight seeds, dimensions16/32/64. Its normalizer is the
Frobenius norm of the residual-Jacobian Gram at z=0, NOT the common-amplitude
model's lifted H_n normalizer. See its own settings for exact definitions.

Median common-test errors at N=8:
 d       GD          pool10000
16      .09216         .25958
32      .09528         .38513
64      .09453         .51162

A global GD error upper bound is obtained from the 16 vertices of the box
eps/(2+eps)<=h_j<=1, plus the ordinary background norm beta. Across eight
seeds its largest values are .13960,.16018,.13889, respectively; the smallest
pool10000 tested errors are .23386,.36615,.49107. Thus missed test maxima
cannot erase this separation. At N4 some seeds fail to capture a rare mode;
by N8 the tested seeds recover. Larger beta can remove the advantage.
This candidate is more constructed but offers several rare parameter regions.

Reproduction (from repository root)
----------------------------------
PYTHONPATH=. .venv_cpu/bin/python -m examples.baseline.cube_candidates.run_collective
PYTHONPATH=. .venv_cpu/bin/python -m examples.baseline.cube_candidates.run_collective_refine
PYTHONPATH=. .venv_cpu/bin/python -m examples.baseline.cube_candidates.certify_collective
PYTHONPATH=. .venv_cpu/bin/python -m examples.baseline.cube_candidates.report_collective

All negative screens and their logs are retained. No existing baseline
results, slides, or manuscript files have been overwritten.
