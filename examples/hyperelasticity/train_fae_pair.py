"""Train the smooth GRF FAE using the native JAX/Flax FAE implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from examples.hyperelasticity.fae import (
    FunctionalLoader, RelativeL2Metric, TrapezoidL2, FAE_SOURCE,
    build_autoencoder, resample_fields,
)
from contgreedy.fae.checkpoint import save_model_results, write_json
from contgreedy.fae.losses import make_fae_loss
from contgreedy.fae.trainer import Trainer

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("Training requires an allocated GPU")
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    t = config["training"]
    source = ROOT / "data" / config["source_data_directory"]
    metadata = json.loads((source / "metadata.json").read_text())
    source_fields = np.load(source / "fields.npy", mmap_mode="r")
    source_n = metadata["grid_resolution"]
    n = config["resolution"]
    fields = resample_fields(source_fields, n)
    splits = {name: np.arange(*value) for name, value in metadata["splits"].items()}
    ids = splits["train"][:t["train_size"]]
    mean, std = float(fields[ids].mean()), float(fields[ids].std())
    standardized = (fields - mean) / std
    reference = (np.asarray(source_fields)[..., None] - mean) / std
    train_loader = FunctionalLoader(
        standardized, standardized, ids, batch_size=t["batch_size"],
        microbatch_size=t["microbatch_size"], noise_std=t["noise_std"],
        seed=t["seed"], paired=True,
    )
    validation = FunctionalLoader(standardized, reference, splits["validation"],
                                 batch_size=t["evaluation_batch_size"])
    ae = build_autoencoder(config)
    metric = RelativeL2Metric(ae, mean, std)
    accumulation_steps = 2 * t["batch_size"] // t["microbatch_size"]
    trainer = Trainer(
        ae, make_fae_loss(ae, TrapezoidL2(), beta=t["latent_penalty"]),
        train_loader, validation, [metric], accumulation_steps=accumulation_steps,
        gradient_clip=t["gradient_clip"], run_directory=run,
    )
    updates_per_epoch = len(train_loader) // accumulation_steps
    total_updates = t["epochs"] * updates_per_epoch
    key = jax.random.PRNGKey(t["seed"])
    initial = trainer.initialize(
        key, learning_rate=t["learning_rate"],
        minimum_learning_rate=t["minimum_learning_rate"],
        total_updates=total_updates,
    )
    count = sum(v.size for v in jax.tree_util.tree_leaves(initial.params))
    if count > t["max_real_parameters"]:
        raise ValueError(f"Parameter budget exceeded: {count}")
    decoder_kind = config["model"].get("decoder_kind", "concat")
    provenance = dict(
        framework="contgreedy.fae Trainer/Autoencoder/FAE loss/UNO/decoder ablation",
        jax=jax.__version__, devices=[str(d) for d in jax.devices()],
        parameter_count=count, resolution=n, source_resolution=source_n,
        data_sha256=hashlib.sha256((source / "fields.npy").read_bytes()).hexdigest(),
        normalization=dict(mean=mean, std=std, fitted_split="train"),
        splits=metadata["splits"],
        resampling="endpoint-aligned bilinear; same sample identities",
        validation="input 100x100, original 128x128 target" if n == 100 else "resampled input, original target",
        decoder_kind=decoder_kind,
        model_change=f"controlled decoder ablation: {decoder_kind}; no decoder LayerNorm",
        sources_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(FAE_SOURCE.glob("*.py"))
        },
    )
    provenance["sources_sha256"][str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fae_path = Path(__file__).resolve().with_name("fae.py")
    provenance["sources_sha256"][str(fae_path.relative_to(ROOT))] = hashlib.sha256(fae_path.read_bytes()).hexdigest()
    write_json(run / "config.json", config)
    write_json(run / "provenance.json", provenance)
    print("PROVENANCE", json.dumps(provenance), flush=True)
    # Compile and time repeated microbatches before committing to training.
    step_fn = trainer.train_step
    iterator = iter(train_loader)
    batch = next(iterator)
    compiled = step_fn.lower(initial, batch).compile()
    memory = compiled.memory_analysis()
    report = dict(argument_bytes=memory.argument_size_in_bytes,
                  temporary_bytes=memory.temp_size_in_bytes,
                  output_bytes=memory.output_size_in_bytes)
    report["compiled_peak_GiB"] = (memory.argument_size_in_bytes + memory.temp_size_in_bytes
                                    + memory.output_size_in_bytes - memory.alias_size_in_bytes)/2**30
    temporary_state = initial
    # Prime both initial-None and empty-dict batch_stats signatures.
    loss, temporary_state = step_fn(temporary_state, batch)
    loss, temporary_state = step_fn(temporary_state, batch)
    jax.block_until_ready((loss, temporary_state))
    start = time.perf_counter()
    for _ in range(trainer.accumulation_steps):
        loss, temporary_state = step_fn(temporary_state, batch)
    jax.block_until_ready((loss, temporary_state))
    report["seconds_per_effective_update"] = time.perf_counter() - start
    report["estimated_training_seconds"] = report["seconds_per_effective_update"] * total_updates
    report["finite_loss"] = bool(jnp.isfinite(loss))
    report["device_memory"] = jax.devices()[0].memory_stats()
    if not report["finite_loss"]:
        raise FloatingPointError("GPU preflight produced nonfinite loss")
    write_json(run / "preflight.json", report)
    print("PREFLIGHT", json.dumps(report), flush=True)
    del temporary_state, compiled, iterator, step_fn
    train_loader.epoch = 0
    if args.preflight_only:
        write_json(run / "status.json", dict(status="preflight_completed"))
        return
    results = trainer.fit(
        initial, epochs=t["epochs"],
        evaluation_interval=t["evaluation_interval_epochs"],
    )
    chosen = results["state"]
    test_source = FunctionalLoader(standardized, reference, splits["test"],
                                   batch_size=t["evaluation_batch_size"])
    metric(chosen, key, test_source)
    test128 = metric.last
    test_grid = FunctionalLoader(standardized, standardized, splits["test"],
                                 batch_size=t["evaluation_batch_size"])
    metric(chosen, key, test_grid)
    summary = dict(status="completed", epoch=t["epochs"], real_parameters=count,
                   decoder_kind=decoder_kind,
                   selected_epoch=results["best_epoch"],
                   best_validation_relative_L2=results["best_metric"],
                   completed_updates=total_updates,
                   test_reference=test128, test_training_grid=metric.last,
                   reference_resolution=source_n, training_resolution=n)
    save_model_results(results, str(run), metadata=provenance)
    write_json(run / "metrics.json", summary)
    write_json(run / "status.json", summary)
    print("FINAL_STATUS", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
