Paired-mode cube reduced-basis baseline
=====================================

PDE: -u''(x;z) = d^(-1/2) sum_j z_j f_j(x), x in (0,1),
u(0;z)=u(1;z)=0, z in [-1,1]^d.
psi_j(x)=sqrt(2)/(j*pi)*sin(j*pi*x), f_j=-psi_j''.
The exact solution is u=d^(-1/2) sum_j z_j psi_j.
The psi_j are orthonormal in the energy inner product integral u'v'.
All algebra is computed in these exact energy coordinates, without FEM.

The model matrix at reduced basis Q is H=(I-QQ^T)/d. J=z^T H z
is simultaneously the squared energy projection error and the exact
squared dual residual. Width: d_N=sqrt((d-N)/d). The upper bound is
given by retaining N coordinate modes; the lower bound follows by
averaging squared errors over uniform signs and using rank N.

For d=2m, repeated Hadamard rows give a prescribed valid exact-greedy
prefix of length m. Each selected snapshot is orthogonal to the prior
space and has energy norm one, attaining the global residual maximum.
Its span is the pair-average space. At this space,
J=sum_i(z_(2i-1)-z_(2i))^2/(2d), maximum J=1.
This structure is not assumed for arbitrary later greedy spaces.

Three distinct measurements are saved:
1. Fixed-space search: 16384 independent uniform starts at the paired
   prefix, with the exact original projected-gradient update. Empirical
   probabilities are compared with the binomial theoretical lower bound.
2. Paired-prefix RB continuation: each method starts from the same
   prescribed m-dimensional prefix and enriches to N=3d/4.
3. Empty-start RB: each method performs ordinary greedy enrichment from
   N=0 to N=d/2. No paired structure is imposed on this experiment.

GD uses alpha=0.6, T=80, g=2Hz/||H||_F,
eta=alpha/sqrt(1+||g||_2^2), z_next=clip(z+eta*g,-1,1),
and returns the best visited point. There is one initial point per
greedy round. Pools have sizes 256,4096,10000 and are nested for each
seed; each pool is fixed throughout the corresponding RB training.
All methods use CPU float64 with one BLAS thread.

Test sets are shared across methods/seeds and generated before training.
d<=16 uses all cube vertices up to antipodal symmetry. Larger d uses
65536 uniform vertices plus up to 65536 paired-opposite vertices, covering
the known high-error configurations at the paired prefix. These are
sampled worst errors for larger d, not certified global maxima.
The width in E_N/d_N is exact; the test maximum E_N is empirical when
the vertex set is not exhaustive.

Training wall time includes candidate generation, matrix construction,
search and enrichment. It excludes test evaluation and output writes.
The prescribed prefix is common and excluded from continuation timing.
The T+1 probability comparison is a query-count convention, not a claim
of exactly equal wall time for gradient and objective evaluations.

Run from the repository root:
  PYTHONPATH=. .venv_cpu/bin/python -m examples.baseline.cube_paired.run
  PYTHONPATH=. .venv_cpu/bin/python -m examples.baseline.cube_paired.report

Outputs: runs/cube_paired_rb_20261104. Existing runs and slides are untouched.
