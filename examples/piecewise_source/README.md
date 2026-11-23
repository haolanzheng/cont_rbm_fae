# Poisson with piecewise constant sources

On the unit square, solve `-Delta u = sum_j z_j 1_(D_j)` with zero Dirichlet
boundary conditions. The source cells form an `m × m` partition and
`z in [-1,1]^(m²)`. Sine expansions determine the energy Gram matrix.

`run.py` provides model construction, projected GD and basis enrichment.
`boundary_updates.py` compares clipping, active-face and adaptive normal
updates. Search and evaluation use the same energy norm.

```bash
python run.py source --m 2 4 --output runs/source_new
```

See [the experiment notes](../../docs/EXPERIMENTS.md) for the experiment commands.
Presentation scripts are kept locally in `local_archive/slides_tools/`. Historical result-directory names are
unchanged. Several follow-up scripts require those local results.
