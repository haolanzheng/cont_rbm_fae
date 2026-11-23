"""Paired nonlinear coordinate-decoder comparison at latent dimension 64."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from contgreedy.fae.training import atomic_json
from examples.hyperelasticity.fae_decoder_comparison import validation_latent_rank

ROOT = Path(__file__).resolve().parents[2]
FIXED_POD_EXPERIMENT = ROOT / "docs/fae_decoder_ablation_64631091.json"
POD_DIAGNOSTIC = ROOT / "docs/fae_pod_diagnostic_4096.json"


def fixed_pod_comparator():
    experiment = json.loads(FIXED_POD_EXPERIMENT.read_text())
    stage = next(
        row for row in experiment["stages"] if row["name"] == "fae64_jax_pod_n4096"
    )
    pod = json.loads(POD_DIAGNOSTIC.read_text())
    rank64 = next(row for row in pod["results"] if row["rank"] == 64)
    return {
        "fixed_pod_64": stage,
        "training_only_pod_64": rank64,
        "sources_sha256": {
            str(FIXED_POD_EXPERIMENT.relative_to(ROOT)): hashlib.sha256(
                FIXED_POD_EXPERIMENT.read_bytes()
            ).hexdigest(),
            str(POD_DIAGNOSTIC.relative_to(ROOT)): hashlib.sha256(
                POD_DIAGNOSTIC.read_bytes()
            ).hexdigest(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    names = (
        "fae64_jax_conditional_mlp_n4096",
        "fae64_jax_modulated_siren_n4096",
    )
    paths = [
        ROOT / "examples/hyperelasticity/configs" / f"{name}.json" for name in names
    ]
    sources = [
        Path(__file__),
        ROOT / "examples/hyperelasticity/fae_decoder_comparison.py",
        ROOT / "src/contgreedy/fae/model.py",
        ROOT / "src/contgreedy/fae/training.py",
        ROOT / "jobs/fae_nonlinear_decoder_comparison.sbatch",
    ]
    manifest = {
        "status": "running",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hypothesis": (
            "A nonlinear conditional coordinate field can outperform the fixed "
            "64-dimensional POD-like decoder on the same encoder and latent budget"
        ),
        "controlled": [
            "data and held-out splits",
            "encoder, pooling, and their initialization",
            "latent dimension",
            "loss, seed, optimizer, learning-rate schedule, and update count",
        ],
        "comparators": fixed_pod_comparator(),
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
                "training_elapsed_seconds": history[-1]["elapsed_seconds"],
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
