# Experiments

Run commands from the repository root. Use a new output directory for each run.

## Piecewise constant sources

`examples/piecewise_source/` solves Poisson on the unit square with `m × m`
source cells and parameters in `[-1,1]^(m²)`. The objective is the squared
energy error divided by the Frobenius norm of its Gram matrix.

```bash
python run.py source --m 2 4 --output runs/source_new
```

The numerical core uses GD with `T=80`, `alpha=0.6`, and pools of
256, 4096 and 10000 points. The writeup compares maximum energy errors on
4096 interior points and 4096 vertices shared by all methods; four seeds.
The exploratory test union in `run.py` is larger. The fixed-test evaluation
and presentation exports are kept locally in `local_archive/slides_tools/`.

Retained studies:

- `dimension_study.py`, `square_dimension_study.py`: stored trajectories across dimensions.
- `training.py`: GD, fixed-pool and mesh training.
- `equal_time_pool.py`, `compare_pool_size.py`: training time and pool size.
- `full_rank_decay.py`: errors through `N=d-1`.
- `boundary_updates.py`: clipping, active face and adaptive normal updates.

These follow-up scripts use the recorded files in
`runs/piecewise_forcing_2d_20261112/`; those files are local, not supplied by Git.
Run scripts directly, for example:

```bash
python examples/piecewise_source/boundary_updates.py --help
```

## Sphere and ball

`examples/baseline/` contains the spectral diffusion model, its HF and reduced
models, and probability-bound calculations in `geometry_cpu.py`.

```bash
python run.py sphere --output runs/sphere_new
python run.py ball --output runs/ball_new
```

Each domain has a `config.json`; use `--config PATH` for another setting.
The defaults retain `rho=0.78`, `alpha=0.6`, `T=40`, the energy-dual residual,
and the fixed-pool score cache. HF models are loaded from the local cache or
assembled when absent. The scripts use CPU float64 and one BLAS thread.

`common/run_energy_residual.py` is the retained batch runner for the writeup
settings, including dimensions through 128. It writes under
`runs/energy_residual_20261120/` and resumes existing results. A fresh clone
rebuilds the HF models; historical test samples are reused when available.
`common/benchmark_cached_pool.py` measures training separately from testing
and diagnostics. Presentation and writeup exports are local-only tools.
The example code does not import those tools.
