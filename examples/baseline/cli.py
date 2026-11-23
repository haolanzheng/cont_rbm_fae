"""Run a declared spectral protocol into a new output directory."""

import argparse
from pathlib import Path

from contgreedy.data import atomic_json, load_config

from .evaluate import aggregate
from .hf import Problem
from .run import run_greedy


def run_config(config, output):
    expected_policy = {
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "return_best": True,
    }
    if config.get("adam_policy", expected_policy) != expected_policy:
        raise ValueError("Spectral Adam requires the manuscript best-iterate policy")
    config = {**config, "adam_policy": expected_policy}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "config.json", config)
    p = config["problem"]
    hf_solver = p.get("hf_solver", "legacy_fd")
    if hf_solver == "continuous_fe":
        from .continuous_fem import FEMProblem

        problem = FEMProblem.build(
            p["grid"],
            p["dimension"],
            p["contrast"],
            p["spectrum_ratio"],
            quadrature_order=p.get("quadrature_order"),
            element_order=p.get("element_order", 3),
        )
    elif hf_solver == "legacy_fd":
        problem = Problem.build(
            p["grid"],
            p["dimension"],
            p["contrast"],
            p["spectrum_ratio"],
            p["load_mode"],
        )
    else:
        raise ValueError(f"Unknown spectral HF solver: {hf_solver}")
    runs = []
    for method in config["methods"]:
        if method["name"] == "exact_indicator":
            runs.append(run_greedy(problem, "exact_indicator", config["rank"], 7103))
    for seed in config["seeds"]:
        for method in config["methods"]:
            if method["name"] == "exact_indicator":
                continue
            runs.append(
                run_greedy(
                    problem,
                    method["name"],
                    config["rank"],
                    seed,
                    steps=method.get("steps", 25),
                    pool_size=method.get("pool_size", 256),
                    adam_lr=method.get("learning_rate", 0.05),
                )
            )
    result = {
        "config": config,
        "backend": f"jax-float64; scipy sparse {hf_solver} HF",
        "timing": "elapsed times include JIT compilation; not steady-state speed claims",
        "hf_diagnostics": {
            "dofs": problem.A.shape[0],
            "discrete_singular_values": problem.singular_values.tolist(),
            **(
                {
                    "max_relative_energy_projection_error": problem.projection_error,
                    "quadrature_order": problem.quadrature_order,
                    "element_order": problem.element_order,
                }
                if hf_solver == "continuous_fe"
                else {}
            ),
        },
        "runs": runs,
        "aggregate": aggregate(runs),
    }
    atomic_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs/paper_d32_separated.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; existing results are never reused implicitly",
    )
    args = parser.parse_args()
    run_config(load_config(args.config), args.output)


if __name__ == "__main__":
    main()
