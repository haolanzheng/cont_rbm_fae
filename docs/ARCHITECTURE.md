# Code architecture

The shared package contains six modules: `data`, `domain`, `hf`, `rom`,
`sampling`, and `fae`. PDE-specific definitions and experiment drivers belong
in `examples/`. Importing the top-level package does not initialize JAX or
PyTorch, and the shared package does not import examples.

## Responsibilities

- `domain` defines feasible parameter sets; `sampling` defines initialization
  distributions and parameter-selection algorithms.
- `hf` defines shared high-fidelity conventions. Assembly, PDE solvers, and
  boundary conditions belong to each example.
- `rom` provides weighted orthogonalization and the outer greedy iteration.
  Each example supplies its residual, error indicator, and derivatives.
- The greedy loop distinguishes iteration count, reduced dimension, and
  high-fidelity solve count. The stopping policy for linearly dependent
  enrichment vectors is explicit.
- `data` provides reusable Gaussian random field assembly. Covariance
  parameters and the physical interpretation of a field are example-specific.
- `fae` does not import PDE solvers. The representation interface propagates
  derivatives from the decoded physical field to the latent space.

## Examples and shared components

The spectral baseline separates the high-fidelity model, reduced model,
parameter search, evaluation, plotting, and experiment driver. It uses shared
basis construction and greedy routines. Adam and power-iteration searches
return the best evaluated iterate, including the initial and final points.

Thermal-fin separates geometry, assembly, reduced models, sampling, data, and
evaluation. Offline matrices and Gram updates use SciPy/NumPy; online indicator
values, automatic differentiation, and Adam use JAX. NumPy references and
incremental pool caches remain available for comparison.

The screening and elasticity examples use shared host optimizers. Riemannian
L-BFGS is implemented in `contgreedy.sampling`, rather than imported from another
example's driver. The BGK shared library is loaded only when needed; importing
the module does not require a precompiled library.

## Numerical backends

The private `_jax.py` module enables float64 for numerical kernels that require
it. JAX handles online dense calculations that benefit from JIT compilation or
automatic differentiation. SciPy and C++ solvers retain explicit interfaces.
Not every historical example has been rewritten in JAX or split into separate
high-fidelity and reduced-model modules.

FAE code includes an array-based JAX implementation in `model.py` and
`training.py`, and native Flax components in `core.py`, `uno.py`, and `decoder.py`.
Example code assembles the appropriate components and training procedure.
The diffusion-coefficient example also provides an equivalent PyTorch model for
local MPS training. Its shared configuration and alignment checks are documented
in `examples/fae_diffusion_grf/README_FAE.txt`.

## Verification and reproducibility

`tests/fixtures/original_small_runs.json` stores small reference runs from before
the numerical refactoring. The historical spectral Adam fixture uses the final
iterate; the revised best-iterate policy is checked against an independent
NumPy implementation in `tests/test_best_iterate.py`.

FAE checks cover model outputs, derivatives, optimizer updates, and checkpoint
recovery. Temporary small-data tests check interrupted versus uninterrupted
training without modifying production data or checkpoints.

New experiment entry points require a new output directory. Historical commands
retain their original caching conventions; see `EXPERIMENTS.md`. Timings that
include JIT compilation should not be interpreted as steady-state timings or
compared directly with older NumPy measurements.
