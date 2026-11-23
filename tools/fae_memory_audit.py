"""Compile-only memory audit; never execute the OOM step."""
import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
from contgreedy.fae.model import init_model
from contgreedy.fae.training import init_adam, make_corrupted_step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("Run on the same GH200 backend")
    config = json.loads(Path(
        "runs/fae128_raw_clean_noisy_iid005_n2048_b64_e300_64716885/config.json"
    ).read_text())
    params = jax.jit(lambda key: init_model(key, config["model"]))(jax.random.PRNGKey(1))
    state = init_adam(params)
    abstract_state = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), state)
    rows = []
    for n, b in ((128, 128), (128, 64), (100, 128), (100, 16)):
        start = time.perf_counter()
        field = jax.ShapeDtypeStruct((b, n, n), jnp.float32)
        step = make_corrupted_step(config)
        compiled = step.lower(abstract_state, field, field, field,
                              jax.ShapeDtypeStruct((n*n,), jnp.int32),
                              jax.ShapeDtypeStruct((), jnp.float32)).compile()
        memory = compiled.memory_analysis()
        row = dict(resolution=n, batch=b, compile_seconds=time.perf_counter()-start,
                   single_decoder_activation_GiB=b*n*n*256*4/2**30)
        for name in ("argument_size_in_bytes", "output_size_in_bytes",
                     "temp_size_in_bytes", "alias_size_in_bytes"):
            row[name] = int(getattr(memory, name))
        row["planned_peak_GiB"] = (
            memory.argument_size_in_bytes + memory.output_size_in_bytes +
            memory.temp_size_in_bytes - memory.alias_size_in_bytes
        ) / 2**30
        rows.append(row)
        print("MEMORY", json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(jax=jax.__version__, device=str(jax.devices()[0]),
                                               rows=rows), indent=2))
        del compiled, step
        jax.clear_caches()
if __name__ == "__main__":
    main()
