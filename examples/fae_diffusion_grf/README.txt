Matérn coefficient fields: a diffusion test case for FAE inputs
============================================================

Problem
-------
On Omega=(0,1)^2, solve

    -div(a(x) grad u(x)) = 1,       u = 0 on the boundary.

The function-valued input is a, not u. Generate a centered, stationary,
unit-variance Gaussian field g with Matérn covariance

    C(r) = 2^(1-nu)/Gamma(nu) * q^nu * K_nu(q),
    q = sqrt(2*nu)*r/length,        C(0)=1,

and set

    a(x) = exp(log(10)*tanh(g(x))).

Thus 0.1 < a(x) < 10. This smooth scalar transformation bounds the
conductivity; it does not smooth the field in space. The field g is Gaussian;
the transformed coefficient a is not Gaussian. No sample is individually
centered or normalized. The length parameter uses the covariance convention
above and is not a cutoff distance.

Selected parameters: nu=1, length=0.2, Var[g(x)]=1.
The selection retains visible small-scale variation while avoiding the most
finely oscillating candidate. This is an input-generation/FOM pilot; it is
not evidence of an FAE or continuous-greedy advantage.

Generation and discretization
-----------------------------
Circulant embedding samples the covariance at grid nodes without low-rank KL
truncation. Padding is increased if the circulant covariance has significant
negative eigenvalues. The selected embedding has no clipped eigenvalues.
The physical domain is cropped from the embedding; no periodic boundary
condition is imposed on the PDE or enforced between opposite input edges.

P1 finite elements use a uniform triangulation with each square split along
the bottom-left to top-right diagonal. The coefficient used in the FOM is
the P1 interpolant of its nodal values, so its element average is integrated
exactly in the stiffness matrix. Calculations use float64 on one CPU thread;
saved field arrays use float32.

Reproduce, from the continuous_greedy project root:

    python examples/fae_diffusion_grf/run.py \
        --output runs/fae_diffusion_grf_20261122_reproduction

Dependencies: numpy, scipy, matplotlib, threadpoolctl.
No GPU, neural-network framework, or external FEM package is needed.
Command-line arguments can change sample counts and the selected nu/length.

Data and logs
-------------
The initial run is in runs/fae_diffusion_grf_20261122/.
train.npz, validation.npz, and test.npz contain 256, 64, and 64 independent
samples, respectively, on 129x129 nodes (128 intervals per coordinate).
Each file contains:

    g:     Gaussian fields, shape [sample, x2 index, x1 index]
    a:     bounded coefficient fields, same shape
    u:     FOM solutions, same shape
    x:     node coordinates, shared by both coordinate axes
    seed:  seed used for this split

settings.json records parameters and seeds. run.log preserves progress.
screening.json records the parameter comparison. checks.json records the
manufactured-solution check, linear residuals, and mesh refinement results.
PDF and PNG figures show input samples and POD comparisons.
The split names reserve disjoint data for later training and assessment;
no FAE has yet been trained on these data.

Parameter screening
-------------------
Each candidate uses 96 independent samples on a 65x65-node grid. The same
random seed couples the candidate comparisons. The table reports the number
of POD modes retaining 99% of the empirical squared norm, equivalent to a
10% relative aggregate projection error:

    nu   length   centered a, L2      u, L2      u, H1_0 seminorm
    0.5   0.1          90              16              82
    0.5   0.2          85              12              68
    0.5   0.3          80              10              54
    1.0   0.2          75              13              56
    1.5   0.2          66              12              49
    2.5   0.3          34               8              24

Coefficient samples are centered for the coefficient POD. Solution PODs
are uncentered. Their metric is the fixed reference H1_0 seminorm, not a
sample-dependent a-energy norm. These are in-sample POD diagnostics, not
Kolmogorov widths, worst-case RB errors, or held-out reconstruction errors.
The larger ranks are particularly limited by the 96-sample screening size.

Resolution checks
-----------------
For constant a=2 and exact u=sin(pi*x1)sin(pi*x2), the discrete L2 error
relative to the nodal interpolant decreases by a factor 3.96 under halving
the mesh size; the largest relative algebraic residual is 3.3e-14.

Four additional coupled random fields were solved at 64, 128, and 256
intervals per coordinate. The coarse solutions were prolonged by their
actual P1 interpolation. The 128-to-256 relative differences are:

    L2:       0.35%--0.60%
    a-energy: 3.84%--5.06%

The a-energy comparison uses the fine-grid coefficient. These differences
measure mesh sensitivity; they are not rigorous bounds on the continuum
error. The present 128-interval grid is suitable for an initial input/FAE
pilot. Formal RB comparisons at small energy errors require further FOM
refinement. FAE reconstruction and any greedy-search advantage remain to
be tested separately.
