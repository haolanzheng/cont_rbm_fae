# Spectral elliptic benchmark

Read in order: `continuous_fem.py` (mesh-independent sources and P3 FE truth),
`rom.py` (residual and exact
energy-error forms), `sampling.py` (JAX selection), `run.py` (greedy),
`evaluate.py`, `plot.py`. `hf.py` is the legacy grid-calibrated FD model.

From the repository root:

```bash
python run.py spectral --config examples/baseline/configs/smoke.json --output runs/spectral_smoke
python run.py spectral --config examples/baseline/configs/paper_d32_separated.json --output runs/spectral_d32
python run.py spectral --config examples/baseline/configs/continuous_fe_smoke.json --output runs/spectral_fe_smoke
OPENBLAS_NUM_THREADS=1 python examples/baseline/check_continuous_fe.py
OPENBLAS_NUM_THREADS=1 python examples/baseline/build_report_data.py --hf-solver continuous_fe --grid 40 --element-order 3
```

The legacy seven paper configurations recover grid, dimension, rank, spectrum, seed
counts and candidate budgets from the delivered JSON records. The current
best-iterate return policy follows the manuscript; archived final-iterate Adam
trajectories are not reproductions of this corrected policy. Expensive landscape audits
remain available through the historical `experiment` and `make_summary` commands.
The summary uses 30,000 spectral starts for q; direct p audits used 50,000 or
100,000 samples. These are different estimators.

Adam and power return the earliest maximizer over all feasible iterates 0..T,
including initialization and the last update, as required by eq:q-def.
Adam uses T+1 value/gradient evaluations for T updates. Other settings are
unchanged (beta2=.999, epsilon=1e-8, lr=.05, tangent gradient and radial retraction).
The objective is frozen within each greedy round; histories from different ROM
spaces are never compared. Historical Adam returned its final sphere iterate;
its archived results are preserved and must be rerun for the corrected policy.
An exact-indicator baseline may use the audited eigenvector; other selectors
do not use truth error. CPU sparse assembly is retained.

The new continuous-source P3 rerun is in
`runs/report_baseline/fe_p3_grid40_results.json`, with a separate report PDF.
The old FD JSON and figures are historical observations, not FE results.
New elapsed times must not be compared directly with historical CPU times.

For the separate round-sphere and closed-ball CPU reruns, read
[`GEOMETRY_CPU.md`](GEOMETRY_CPU.md). `geometry_cpu.py` implements the sphere
exponential update, relaxed ball projection, and exact FEM coefficient ROM;
`run_geometry_cpu.py` runs both studies with matching device and thread settings,
each with its own initialization law and rerun random pools. Historical selectors
and results are retained.
