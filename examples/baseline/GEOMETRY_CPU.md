# Sphere and ball CPU baselines

The two parameter domains define **separate studies**. Each is compared with
random sampling on its own domain and initialization law. The PDE, FEM truth
model, indicator and greedy enrichment are retained from `continuous_fem.py`.

## Fixed problem and protocol

On D=(0,1)^2 solve

\[
-\nabla\cdot(a\nabla u(z))=\sum_{j=1}^d z_j f_j,\qquad u|_{\partial D}=0,
\quad a=1+0.25\sin(2\pi x)\sin(2\pi y).
\]

The mesh-independent manufactured sources have continuum energy-orthogonal
solutions scaled by rho^(j-1), rho=0.78. The conforming triangular P3 truth
space uses 40 interior vertices per coordinate and 14,884 free DoFs.
Dimensions are d=8,16,32,64; enrichment stops at n=d/2; seeds are 1000,...,1007.

For a fixed reduced space V_n, the squared residual dual norm is J_n(z)=z^T H_n z,
with H_n positive semidefinite. The residual dual norm uses the unit-Laplacian
stiffness X, whereas RB orthonormality and true error use the diffusion
stiffness A. These state-space metrics do not specify the parameter metric.

Let ell=||H_n||_F. Both optimizers use the normalized scalar objective J_n/ell;
this normalization preserves the relative target set and avoids changing
inner dynamics merely because the residual scale shrinks with greedy rank.
The target is J_n >= gamma^2 lambda_1(H_n), gamma=0.9.

### Round sphere

Z=S^(d-1), with the round metric induced by the Euclidean tangent inner product.
Starts and pools are uniform in surface area, obtained from normalized
Gaussian vectors. The Riemannian derivative and update are

\[
DJ(z)=\frac{2}{\ell}\bigl(H_nz-(z^TH_nz)z\bigr),\quad
v=\frac{0.2\,DJ(z)}{\sqrt{1+\|DJ(z)\|_2^2}},\quad
z^+=\cos\|v\|_2\,z+\frac{\sin\|v\|_2}{\|v\|_2}v.
\]

The sine quotient is interpreted continuously at zero. Only floating-point
norm drift is corrected. This is the exponential update from the later
manifold theorem, rather than the historical normalized Euclidean update.
The sphere theorem and its analytical bound use this exponential update;
the probability audit uses the same map.

### Closed ball

Z={z in R^d: ||z||_2 <= 1}, with the Euclidean metric. A sample is R U, with
U uniform on the sphere and R=V^(1/d), V uniform on [0,1] independent of U.
Consequently P[R <= r]=r^d: the law is uniform in volume, not uniform in radius.

\[
\eta(z)=\frac{\alpha}{\sqrt{1+\|2H_nz/\ell\|_2^2}},\quad \alpha=0.2,
\qquad P_Z(z)=\frac{z}{\max\{1,\|z\|_2\}},\quad
z^+=P_Z(z+2\eta(z)H_nz/\ell).
\]

Both domains use the same step-size rule and alpha=0.2, with their respective
gradients. Alpha is the scale parameter; eta(z) multiplies the gradient.
The ball uses direct projection, without relaxation. The code retains fixed
steps and an optional omega argument for reproducing archived runs.

Current results are reproduced by
`PYTHONPATH=src:. .venv_cpu/bin/python -u -m examples.baseline.refresh_normalized_projection`
and saved in `runs/normalized_projection_20261020/`. Both GD studies were
rerun; unchanged pool, grid and oracle histories were reused. The same
independent test sets were reevaluated, and all methods in the wall-time table
were timed again on the same local CPU with one BLAS thread. Earlier results
remain in their original directories.

## Selectors and timing

- Exact top indicator eigenvector: an oracle reference, timed separately.
- 100 inner updates: best parameter among iterations 0,...,100.
- Fixed random pools of 256,4096,10000 points: sampled once per seed and reused.
- Fresh random pool of 101 points: redrawn at each greedy round; the number of
  indicator matrix actions matches the 101 evaluations of one 100-update search.
  Equal action counts are not an assertion of equal computational work.

All selectors, assembly, solves and audits use float64 NumPy/SciPy on the same
local CPU. BLAS and OpenMP threads are fixed to one. No JAX/PyTorch backend is
imported by this runner. Method timings include matrix formation, pool setup,
selection and enrichment; exact quality/error audits are excluded. Shared FEM
assembly and residual preprocessing are reported separately. Timings must not
be compared directly with older JAX or remote-device timings.

## Exact coefficient representation of the FEM model

Let S=A^(-1)B, G0=S^T A S and H0=B^T X^(-1)B. Every snapshot belongs to range(S),
so V_n=S Q_n with Q_n^T G0 Q_n=I. With P_n=I-Q_n Q_n^T G0,

\[
R_n=B P_n,\qquad H_n=P_n^T H_0 P_n,\qquad G_n=P_n^T G_0 P_n.
\]

The largest true HF energy error is sqrt(lambda_1(G_n)). These identities are
checked against the full assembled FEM residual and error. This reduces CPU
cost without replacing the diffusion PDE by a diagonal surrogate. H0 is
computed by actual sparse X solves before the method runs.

## Independent probability audits

At reference ranks n=0,d/4,d/2 on the exact-indicator greedy history, hold H_n
fixed and draw 4096 independent starts in the current rerun (the original
audit used 8192). All optimization methods within that
domain use the same frozen objective and the same batch. These are independent
search trajectories, not a pool from which one winning trajectory is selected.
At T=0,1,2,5,10,25,50,100 record best-so-far threshold attainment and exact
pointwise 95% binomial intervals.

For a fresh iid random pool of size M on a frozen objective, success is
1-(1-p)^M, where p is the one-draw success probability. This conditional iid
identity is not used for a fixed pool reused after data-dependent enrichments.
Zero direct hits in an audit does not establish zero probability.

### Ball discrete analytical lower bound

Write J(z)=z^T H z/ell, sigma=lambda_1/ell, and fix 0<gamma<1. Before
J reaches gamma^2 sigma, the PSD inequality H^2 <= lambda_1 H implies

\[
\|DJ(z)\|^2\le4\sigma J(z)<4\gamma^2\sigma^2,
\qquad \eta(z)\ge h=\frac{\alpha}{\sqrt{1+4\gamma^2\sigma^2}}.
\]

For the actual eta=eta(z), the trial point satisfies

\[
\|z+2\eta Hz/\ell\|^2<1+4\gamma^2\eta\sigma(1+\eta\sigma).
\]

The absolute top coefficient therefore grows by at least

\[
f(\eta)=\frac{1+2\eta\sigma}
{\sqrt{1+4\gamma^2\eta\sigma(1+\eta\sigma)}}\ge f(h)=c>1.
\]

The inequality follows because f is increasing. Consequently

\[
q_T\ge\mathbb P\{\mathrm{Beta}(1/2,(d+1)/2)\ge\gamma^2c^{-2T}\}.
\]

The constants are computed from H and the step rule, without fitting measured
attainment frequencies. A kernel and repeated top eigenvalues are allowed.
The old fixed-matrix-power formula does not describe the new variable-step
update. The corresponding product has factors I+2 eta(z_k) H/ell, which depend
on the trajectory.

For either domain, when lambda_2/lambda_1<gamma^2, direct success requires the
squared top coefficient to be at least
(gamma^2-lambda_2/lambda_1)/(1-lambda_2/lambda_1). Its Beta law has second
parameter (d-1)/2 on the sphere, (d+1)/2 in the ball, giving a one-draw upper
bound and, for a fresh pool, a success upper bound. These bounds and the
binomial intervals are saved even when no direct hit is observed.

## Reproduction

Use a native local Python with the project's `fem` dependencies. The local
Mac rerun has a dedicated `.venv_cpu` that uses the native `jaxnn` NumPy/SciPy
packages and the existing pure-Python scikit-fem 12.0.2 package. The historical
`.venv` contains a Linux Python executable and is not the local CPU environment.

```bash
PYTHONPATH=src:. .venv_cpu/bin/python -m pytest -q tests/test_geometry_cpu.py
PYTHONPATH=src:. .venv_cpu/bin/python -u -m examples.baseline.run_geometry_cpu --output runs/geometry_cpu_NEW
```

The output directory must be new. It contains the exact configuration,
environment, source hashes, FEM coefficient matrices, frozen objectives,
per-seed histories, costs, probability certificates, CSV summaries, and
separate sphere/ball plots. The 2026-10-12 rerun is in
`runs/geometry_cpu_20261012/`.

## File locations

- Experiment and plotting code: `examples/baseline/`.
- Current sphere/ball results: `runs/normalized_projection_20261020/`;
  the original console output is `run.log` in that directory.
- Earlier fixed-step ball run: `runs/direct_projection_20261020/run.log`.
- Earlier baseline report data: `runs/report_baseline/`.
- Paper figures and TeX: `paper/`; LaTeX `.log` files are compilation logs.

To retain console output when running the current experiment:

```bash
mkdir -p runs/normalized_projection_20261020
PYTHONPATH=src:. .venv_cpu/bin/python -u -m examples.baseline.refresh_normalized_projection 2>&1 | tee -a runs/normalized_projection_20261020/run.log
```

The two short-slide dimension plots are reproduced from saved results by
`examples/baseline/plot_dimension_bounds.py` and
`examples/baseline/plot_empirical_dimension.py`.
