# Continuous greedy RBM and FAE

Working code for gradient-based greedy selection in reduced basis methods.

| Example | Location |
|---|---|
| Poisson with piecewise constant sources | `examples/piecewise_source/` |
| Spectral diffusion: sphere and ball | `examples/baseline/` |
| FAE for diffusion coefficient fields (in progress) | `examples/fae_diffusion_grf/` |

```bash
python -m pip install -e '.[fem,test]'
python run.py list
python -m pytest tests -q
```

For FAE training, install `.[fae-torch]` or `.[fae-jax]` in a separate environment.
The JAX extra pins the versions used for CPU alignment checks; GPU JAX must
also match the server's CUDA setup.

- [Experiments](docs/EXPERIMENTS.md): commands and retained settings.
- [FAE](docs/FAE.md): data, initialization and backend instructions.

`src/contgreedy/` contains shared numerical and FAE components. Exploratory
examples and older utilities are kept locally in `local_archive/` and excluded
from Git. Presentation scripts are in `local_archive/slides_tools/`. Data, run
logs, checkpoints, manuscripts and figures are also excluded.
