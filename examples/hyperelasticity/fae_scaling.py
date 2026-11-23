"""Predeclared, validation-only data-then-capacity experiment on one GH200."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from contgreedy.fae.training import atomic_json

ROOT = Path(__file__).resolve().parents[2]
# Attained by training-only weighted rank-128 POD, NOT a theoretical bound.
POD_VALIDATION_REFERENCE = 0.10209266321115457


def needs_capacity(validation_mean, reference=POD_VALIDATION_REFERENCE):
    if not 0 <= validation_mean < float("inf"):
        raise ValueError("Validation mean must be finite and nonnegative")
    return validation_mean > reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    names = ["fae128_jax_w32_n4096_steps6400", "fae128_jax_w32_n4096",
             "fae128_jax_w48_n4096"]
    config_paths = [ROOT / "examples/hyperelasticity/configs" / (n + ".json") for n in names]
    sources = [Path(__file__), ROOT / "examples/hyperelasticity/expand_data.py",
               ROOT / "src/contgreedy/fae/model.py", ROOT / "src/contgreedy/fae/training.py",
               ROOT / "jobs/fae_scaling.sbatch"]
    manifest = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "decision_rule": "Increase capacity iff width-32 full-budget best validation mean exceeds the training-only POD-128 reference; never use test metrics for this decision",
        "pod_validation_reference": POD_VALIDATION_REFERENCE,
        "configurations": {n: json.loads(p.read_text()) for n, p in zip(names, config_paths)},
        "sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "status": "running", "stages": [],
        "caveats": ["One seed: exploratory evidence, not statistical significance",
                    "Matched-update comparison uses final-checkpoint validation: best-checkpoint evaluation cadence differs",
                    "200-epoch comparison changes both dataset size and total updates"],
    }
    atomic_json(run / "experiment.json", manifest)
    try:
        for index, (name, source) in enumerate(zip(names, config_paths)):
            if index == 2:
                score = manifest["stages"][-1]["best_validation_mean"]
                manifest["capacity_decision"] = {"validation_mean": score,
                    "reference": POD_VALIDATION_REFERENCE, "train_larger_model": needs_capacity(score)}
                atomic_json(run / "experiment.json", manifest)
                if not needs_capacity(score):
                    print("Data expansion beat POD reference; capacity stage skipped.", flush=True)
                    break
            # Snapshot configs before execution so queued edits cannot change this protocol.
            snapshot = run / (name + ".json")
            atomic_json(snapshot, manifest["configurations"][name])
            destination = run / name
            manifest["active_stage"] = name
            atomic_json(run / "experiment.json", manifest)
            print(f"START {name}", flush=True)
            subprocess.run([sys.executable, "-u", "-m", "examples.hyperelasticity.train_fae",
                            "--config", str(snapshot), "--run-dir", str(destination)],
                           cwd=ROOT, check=True)
            metrics = json.loads((destination / "metrics.json").read_text())
            history = json.loads((destination / "history.json").read_text())
            manifest["stages"].append({"name": name, "run_directory": str(destination),
                "best_validation_mean": metrics["best_validation_relative_L2"],
                "final_validation_mean": history[-1]["validation"]["relative_L2"]["mean"],
                "epochs": manifest["configurations"][name]["training"]["epochs"]})
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
