"""Controlled 50-epoch comparison of three function-space pooling mechanisms."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

from contgreedy.fae.model import init_model, parameter_count
from contgreedy.fae.training import atomic_json
from examples.hyperelasticity.fae_decoder_comparison import validation_latent_rank

ROOT = Path(__file__).resolve().parents[2]
NAMES = (
    "fae64_pool_orthogonal_e50",
    "fae64_pool_attention_e50",
    "fae64_pool_nonlinear_integral_e50",
)
POOLING_KEYS = {
    "pooling_kind",
    "orthogonal_epsilon",
    "attention_queries",
    "attention_value_dimension",
    "attention_key_dimension",
}


def initialization_audit(configurations):
    first = configurations[NAMES[0]]
    if first["training"]["epochs"] != 50:
        raise ValueError("This comparison is fixed at 50 epochs")
    common_model = {
        key: value for key, value in first["model"].items() if key not in POOLING_KEYS
    }
    for name in NAMES[1:]:
        config = configurations[name]
        candidate = {
            key: value
            for key, value in config["model"].items()
            if key not in POOLING_KEYS
        }
        if candidate != common_model or config["training"] != first["training"]:
            raise ValueError(f"Non-pooling controls differ in {name}")
        if config["data_directory"] != first["data_directory"]:
            raise ValueError(f"Dataset differs in {name}")

    parameters = {
        name: init_model(
            jax.random.PRNGKey(configurations[name]["training"]["seed"]),
            configurations[name]["model"],
        )
        for name in NAMES
    }
    shared = set.intersection(*(set(values) for values in parameters.values()))
    shared = sorted(name for name in shared if not name.startswith("pool."))
    for parameter_name in shared:
        reference = np.asarray(parameters[NAMES[0]][parameter_name])
        for name in NAMES[1:]:
            np.testing.assert_array_equal(
                reference,
                np.asarray(parameters[name][parameter_name]),
                err_msg=f"shared initialization differs: {parameter_name}",
            )
    return {
        "shared_initialization_exact": True,
        "shared_parameter_arrays": len(shared),
        "real_parameters": {
            name: parameter_count(parameters[name]) for name in NAMES
        },
        "pool_parameters": {
            name: sum(
                value.size
                for key, value in parameters[name].items()
                if key.startswith("pool.")
            )
            for name in NAMES
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    paths = [
        ROOT / "examples/hyperelasticity/configs" / f"{name}.json"
        for name in NAMES
    ]
    configurations = {
        name: json.loads(path.read_text()) for name, path in zip(NAMES, paths)
    }
    sources = [
        Path(__file__),
        ROOT / "src/contgreedy/fae/model.py",
        ROOT / "src/contgreedy/fae/training.py",
        ROOT / "jobs/fae_pooling_comparison.sbatch",
        *paths,
    ]
    manifest = {
        "status": "running",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "question": (
            "At a strict 64-scalar bottleneck, compare orthogonal learned "
            "integrals, quadrature-aware cross-attention, and nonlinear integrals"
        ),
        "controlled": [
            "4096 training fields and identical validation/test splits",
            "UNO encoder and nonlinear coordinate decoder architecture",
            "exact shared-parameter initialization",
            "64-scalar latent bottleneck",
            "seed, batches, Adam, loss, and 50-epoch cosine schedule",
        ],
        "pooling_definitions": {
            "orthogonal_integral": "weighted Gram-whitened signed coordinate kernels",
            "quadrature_attention": "8 queries x 8 values with log quadrature weights",
            "nonlinear_integral": "pointwise MLP of coordinates and UNO features, then quadrature",
        },
        "configurations": configurations,
        "initialization_audit": initialization_audit(configurations),
        "sources_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "stages": [],
    }
    atomic_json(run / "experiment.json", manifest)
    try:
        for name in NAMES:
            config = configurations[name]
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
            manifest["stages"].append(
                {
                    "name": name,
                    "pooling_kind": config["model"]["pooling_kind"],
                    "run_directory": str(destination),
                    "real_parameters": metrics["real_parameters"],
                    "selected_epoch": metrics["selected_epoch"],
                    "best_validation_mean": metrics["best_validation_relative_L2"],
                    "final_validation_mean": history[-1]["validation"]["relative_L2"][
                        "mean"
                    ],
                    "training_elapsed_seconds": history[-1]["elapsed_seconds"],
                    "test": metrics["test"],
                    "validation_latent_rank": validation_latent_rank(
                        destination, config
                    ),
                }
            )
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
