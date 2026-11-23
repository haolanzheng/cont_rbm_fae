# FAE components

`core.py` defines the encoder/decoder interfaces and autoencoder.
`uno.py` implements the UNO encoder; `decoder.py` contains coordinate encodings
and residual MLP decoders. Training and checkpoint utilities are separate.

The active diffusion-coefficient example is in
[`examples/fae_diffusion_grf/`](../../../examples/fae_diffusion_grf/).
Its JAX implementation uses these shared components; its PyTorch implementation
uses the same model configuration and weight layout.

The array-based JAX utilities in `model.py` and `training.py` remain available
for older checkpoints. They are not the current diffusion FAE architecture.
