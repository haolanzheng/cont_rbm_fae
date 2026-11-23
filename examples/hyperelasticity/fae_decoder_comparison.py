"""Paired no-LayerNorm comparison of Psi(x) and FiLM Psi(x; z) decoders."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

from contgreedy.fae.model import encode
from contgreedy.fae.training import atomic_json, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def effective_rank(codes):
    centered = np.asarray(codes, dtype=np.float64) - np.mean(codes, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    fraction = variance / variance.sum()
    cumulative = np.cumsum(fraction)
    return {
        "samples": len(centered),
        "participation_rank": float(variance.sum() ** 2 / np.sum(variance**2)),
        "entropy_rank": float(
            np.exp(-np.sum(fraction * np.log(np.maximum(fraction, 1e-30))))
        ),
        "components_for_variance": {
            str(level): int(np.searchsorted(cumulative, level) + 1)
            for level in (0.9, 0.95, 0.99)
        },
    }


def validation_latent_rank(run_directory, config):
    data = ROOT / "data" / config["data_directory"]
    metadata = json.loads((data / "metadata.json").read_text())
    fields = np.load(data / "fields.npy", mmap_mode="r")
    ids = np.arange(*metadata["splits"]["validation"])
    state, _ = load_checkpoint(Path(run_directory) / "best.npz", config)
    model_config = config["model"]
    encoder = jax.jit(lambda params, values: encode(params, values, model_config))
    codes = []
    for first in range(0, len(ids), config["training"]["batch_size"]):
        raw = np.asarray(
            fields[ids[first : first + config["training"]["batch_size"]]],
            dtype=np.float32,
        ).reshape(-1, metadata["grid_resolution"], metadata["grid_resolution"])
        standardized = (raw - metadata["mean"]) / metadata["std"]
        codes.append(np.asarray(jax.device_get(encoder(state["params"], standardized))))
    return effective_rank(np.concatenate(codes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    names = ("fae64_jax_pod_n4096", "fae64_jax_pod_film_n4096")
    paths = [
        ROOT / "examples/hyperelasticity/configs" / f"{name}.json" for name in names
    ]
    sources = [
        Path(__file__),
        ROOT / "src/contgreedy/fae/model.py",
        ROOT / "src/contgreedy/fae/training.py",
        ROOT / "jobs/fae_decoder_comparison.sbatch",
    ]
    manifest = {
        "status": "running",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hypothesis": "Compare fixed Psi(x) with raw-latent FiLM Psi(x;z); neither decoder uses LayerNorm",
        "controlled": [
            "data and held-out splits",
            "encoder and pooling",
            "latent dimension and basis count",
            "initial shared parameters",
            "loss, seed, optimizer, learning-rate schedule, and update count",
        ],
        "configurations": {
            name: json.loads(path.read_text()) for name, path in zip(names, paths)
        },
        "sources_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "stages": [],
    }
    atomic_json(run / "experiment.json", manifest)
    try:
        for name in names:
            config = manifest["configurations"][name]
            snapshot = run / f"{name}.json"
            atomic_json(snapshot, config)
            destination = run / name
            manifest["active_stage"] = name
            atomic_json(run / "experiment.json", manifest)
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "examples.hyperelasticity.train_fae",
                    "--config",
                    str(snapshot),
                    "--run-dir",
                    str(destination),
                ],
                cwd=ROOT,
                check=True,
            )
            metrics = json.loads((destination / "metrics.json").read_text())
            history = json.loads((destination / "history.json").read_text())
            stage = {
                "name": name,
                "run_directory": str(destination),
                "real_parameters": metrics["real_parameters"],
                "selected_epoch": metrics["selected_epoch"],
                "best_validation_mean": metrics["best_validation_relative_L2"],
                "final_validation_mean": history[-1]["validation"]["relative_L2"][
                    "mean"
                ],
                "test": metrics["test"],
                "validation_latent_rank": validation_latent_rank(destination, config),
            }
            manifest["stages"].append(stage)
            atomic_json(run / "experiment.json", manifest)
        manifest["status"] = "completed"
        manifest.pop("active_stage", None)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = repr(error)
        raise
    finally:
        atomic_json(run / "experiment.json", manifest)


if __name__ == "__main__":
    main()
