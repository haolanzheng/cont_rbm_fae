# Thermal fin

Read `domain.py` → `hf.py` → `rom.py` → `sampling.py` → `run.py`.
`data.py` builds the common audit set; `evaluate.py` evaluates physical errors
and local-optimality diagnostics. `fin_local_study.py` is a compatibility facade.

```bash
python run.py thermal-fin --config examples/thermal_fin/configs/smoke.json --output runs/fin_smoke
python run.py thermal-fin --config examples/thermal_fin/configs/historical_d11.json --output runs/fin_d11
```

The historical suite uses d/N=11/50,21/80,33/100; 32 cells per unit length;
10,017 P1 nodes; 16 regions; 49 raw affine terms; physical-baseline X.
The relative indicator is alpha-free and its logarithm is optimized.
Adam uses beta1=.9, beta2=.99, epsilon=1e-8, lr=.05, 100 updates, best iterate.
Conductivity and Bi have logarithmic charts; geometry has a linear chart.

The main TeX's later rank-40→50, 2048-point continuation is a **different**
experiment. Its dedicated historical driver is missing. The 2000/4000-point
suite retained here must not be described as reproducing that continuation.

JAX float64 evaluates the online residual, autodiff gradient and Adam trajectory;
SciPy handles sparse truth solves and local L-BFGS-B/SLSQP. The independent
NumPy reference and incremental pool cache remain for verification and legacy
baselines. Timing includes JIT compilation. Small regression tests are not
evidence that all 81 full-rank historical runs have been reproduced.
