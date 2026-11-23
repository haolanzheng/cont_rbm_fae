# Spectral diffusion: sphere and ball

The two parameter domains share the same diffusion model and Galerkin ROM.
The sphere follows the tangent gradient along a great circle; the ball uses
a normalized gradient step followed by radial projection.

```bash
python run.py sphere --output runs/sphere_new
python run.py ball --output runs/ball_new
```

Run from the repository root. Domain settings are in `sphere/config.json`
and `ball/config.json`. See [the experiment notes](../../docs/EXPERIMENTS.md)
for the writeup batch and cached-pool timing commands.

`geometry_cpu.py` contains the updates and probability bounds.
`continuous_fem.py` and `consistent_fem.py` provide the HF model.
`common/` contains the runners, fixed-pool cache and shared numerical helpers.
The earlier standalone cube experiment is archived locally.
