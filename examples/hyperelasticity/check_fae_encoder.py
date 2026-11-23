"""Audit the Darcy encoder control and run four real full-batch Adam updates."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from contgreedy.fae.encoder import fit_input_normalization
from contgreedy.fae.model import init_model, parameter_count
from contgreedy.fae.training import atomic_json, init_adam, learning_rate, make_step

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "fae128_pool_orthogonal_w32_d256_b6_n2048_b64_e300.json"


def control_audit(config):
    baseline = json.loads((ROOT / "examples/hyperelasticity/configs" / BASELINE).read_text())
    assert config["training"] == baseline["training"], "Training controls changed"
    assert config["data_directory"] == baseline["data_directory"]
    for key, value in baseline["model"].items():
        if key not in ("width", "modes", "pool_channels"):
            assert config["model"][key] == value, key
    assert config["model"]["encoder_kind"] == "darcy_unet"
    seed = jax.random.PRNGKey(config["training"]["seed"])
    initialize = lambda c: jax.jit(lambda k: init_model(k, c["model"]))(seed)
    old, new = initialize(baseline), initialize(config)
    shared = [name for name in old if name.startswith(("decoder.", "position_encoding."))]
    for name in shared:
        np.testing.assert_array_equal(old[name], new[name], err_msg=name)
    count = parameter_count(new)
    assert count == 1_903_368
    assert count <= config["training"]["max_real_parameters"]
    assert learning_rate(1, config["training"]) == 1e-3
    assert learning_rate(9600, config["training"]) == 1e-5
    return {
        "real_parameters": count,
        "decoder_parameters": sum(v.size for k, v in new.items() if k.startswith("decoder.")),
        "unchanged_initialization_arrays": len(shared),
        "decoder_and_position_initialization_exact": True,
        "training_controls_exact": True,
        "updates": 9600,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU preflight requires CUDA")
    jax.config.update("jax_default_matmul_precision", "highest")
    config = json.loads(args.config.read_text())
    report = control_audit(config)
    data = ROOT / "data" / config["data_directory"]
    metadata = json.loads((data / "metadata.json").read_text())
    fields = np.load(data / "fields.npy", mmap_mode="r", allow_pickle=False)
    ids = np.arange(metadata["splits"]["train"][0], config["training"]["train_size"])
    assert len(ids) == 2048 and np.max(ids) < metadata["splits"]["validation"][0]
    config = copy.deepcopy(config)
    config["model"]["encoder_input_normalization"] = fit_input_normalization(
        fields, ids, metadata["grid_resolution"], metadata["mean"], metadata["std"],
        config["model"],
    )
    seed = jax.random.PRNGKey(config["training"]["seed"] + 1)
    params = jax.jit(lambda k: init_model(k, config["model"]))(seed)
    state = init_adam(params)
    batch = np.asarray(fields[ids[:config["training"]["batch_size"]]], dtype=np.float32)
    batch = ((batch - metadata["mean"]) / metadata["std"]).reshape(-1, 80, 80)
    step_fn = make_step(config)
    steps = []
    for i in range(4):
        state, values = step_fn(state, batch, np.float32(learning_rate(i + 1, config["training"])))
        values = jax.device_get(values)
        assert bool(values["finite"]), f"Nonfinite GPU preflight step {i + 1}"
        steps.append({key: float(value) for key, value in values.items() if key != "finite"})
        print("PREFLIGHT_STEP", i + 1, steps[-1], flush=True)
    report.update(
        status="passed",
        platform=jax.default_backend(),
        devices=[str(device) for device in jax.devices()],
        full_batch_shape=list(batch.shape),
        steps=steps,
        encoder_input_normalization=config["model"]["encoder_input_normalization"],
        sources_sha256={
            name: hashlib.sha256((ROOT / "src/contgreedy/fae" / name).read_bytes()).hexdigest()
            for name in ("model.py", "encoder.py", "training.py")
        },
        scope="Implementation and finite-update audit; not a trained accuracy result.",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    print("PREFLIGHT_PASSED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
