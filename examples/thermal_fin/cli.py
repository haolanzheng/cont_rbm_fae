"""Run declared fin settings, with explicit paper-suite versus smoke audit design."""

import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from contgreedy.data import load_config, atomic_json
from .hf import Fin
from .data import make_test
from .run import run


def run_config(config, output):
    expected_policy = dict(
        learning_rate=0.05, beta1=0.9, beta2=0.99, epsilon=1e-8, return_best=True
    )
    if config.get("adam_policy", expected_policy) != expected_policy:
        raise ValueError("Unsupported change to the historical fin Adam policy")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "config.json", config)
    with threadpool_limits(limits=1):
        problem = Fin(config["dimension"], config["cells"])
        if config["audit"] == "historical_boundary_enriched":
            make_test(problem, output)
        elif config["audit"] == "smoke_random":
            points = np.random.default_rng(config["audit_seed"]).uniform(
                -1, 1, (config["audit_count"], problem.d)
            )
            states = np.stack([problem.fom(z) for z in points])
            norms = np.einsum("bi,bi->b", states, (problem.X @ states.T).T)
            np.save(output / "test_u.npy", states)
            # Legacy evaluation reads truth states row-wise.
            np.savez(output / "test_meta.npz", z=points, norms=norms)
        else:
            raise ValueError("Unknown audit design")
        for seed in config["seeds"]:
            for method in config["methods"]:
                run(problem, method, seed, config["rank"], output)
    atomic_json(
        output / "completion.json",
        {
            "status": "completed",
            "backend": "JAX online residual/Adam; SciPy sparse HF",
            "scope": config["scope"],
            "timing": "Includes JIT; no steady-state speed claim",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs/historical_d11.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_config(load_config(args.config), args.output)


if __name__ == "__main__":
    main()
