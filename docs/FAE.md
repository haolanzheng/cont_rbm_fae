# FAE experiment

The retained example reconstructs bounded Matérn diffusion coefficient fields.
It is a reconstruction experiment; coupling to greedy search is not yet tested.

- Input: `a = exp(log(10) tanh(g))`, Matérn `nu=1`, length `0.2`.
- Grid: 129 × 129; train/validation/test: 2048/256/256 fields.
- Model: UNO encoder, orthogonal integral pooling, coordinate MLP decoder.
- Shared configuration: `examples/fae_diffusion_grf/configs/fae_128.json`.
- Latent dimension 128; batch 32; float32.

```bash
python run.py fae-data --output runs/fae_new/data
JAX_PLATFORMS=cpu python run.py fae-init \
  --data runs/fae_new/data --output runs/fae_new/initial.npz
python run.py fae-torch --data runs/fae_new/data \
  --initial runs/fae_new/initial.npz --output runs/fae_new/torch --device mps
```

The initialization command requires the JAX environment. Both backends use
the same data, configuration and initial weight archive. The server launcher
runs only inside an already allocated GPU session; it does not submit a job.

See [README_FAE.txt](../examples/fae_diffusion_grf/README_FAE.txt) for the
JAX command, resume behavior and numerical alignment checks. Keep all datasets,
logs and checkpoints under `runs/`. The previous local training was stopped;
this cleanup does not start another run.
