"""Rerun the sphere and ball baselines as two separate local CPU studies.

From the repository root, with scikit-fem installed in a native environment:
  python -m examples.baseline.run_geometry_cpu --output runs/geometry_cpu_YYYYMMDD

The runner creates a new directory, pins numerical libraries to one CPU thread,
and records raw seeds, frozen objectives, costs, and environment information.
"""

from __future__ import annotations

import os

# Set before importing NumPy/SciPy; threadpoolctl also enforces this at runtime.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/contgreedy_matplotlib")

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg as la
from scipy.stats import beta
from threadpoolctl import threadpool_info, threadpool_limits

from .continuous_fem import FEMProblem
from .geometry_cpu import (
    CoordinateROM, ball_probability_lower, direct_probability_upper,
    pool_search, search, sym, uniform_parameters,
)


def confidence_interval(hits, count):
    """Two-sided 95% exact binomial interval, including zero observed hits."""
    lower = 0.0 if hits == 0 else float(beta.ppf(0.025, hits, count - hits + 1))
    upper = 1.0 if hits == count else float(beta.ppf(0.975, hits + 1, count - hits))
    return [lower, upper]


def pool_probability(p, count):
    return 1.0 if p >= 1.0 else float(-np.expm1(count * np.log1p(-p)))


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def method_names(domain, pool_sizes):
    optimizers = ["exp_rgd100"] if domain == "sphere" else ["relaxed_pg100", "hard_pg100"]
    return ["exact_indicator"] + optimizers + [f"fixed_pool{m}" for m in pool_sizes] + ["fresh_pool101"]


def run_greedy(G0, H0, domain, method, seed, config, rom_kwargs=None):
    d = len(G0)
    rom = CoordinateROM(G0, H0, **(rom_kwargs or {}))
    rng = np.random.default_rng(seed)
    pool_rng = np.random.default_rng(seed + 1_000_003)
    costs = {name: 0.0 for name in ("matrix_seconds", "selector_seconds", "enrichment_seconds", "pool_setup_seconds")}
    pool = None
    if method.startswith("fixed_pool"):
        begin = time.perf_counter()
        pool = uniform_parameters(pool_rng, int(method.removeprefix("fixed_pool")), d, domain)
        costs["pool_setup_seconds"] = time.perf_counter() - begin
    errors = [float(np.sqrt(max(la.eigvalsh(G0)[-1], 0.0)))]
    rounds = []
    h_actions = 0
    for n in range(d // 2):
        begin = time.perf_counter()
        H, _ = rom.matrices(include_error=False)
        costs["matrix_seconds"] += time.perf_counter() - begin
        begin = time.perf_counter()
        if method == "exact_indicator":
            _, vectors = la.eigh(H, subset_by_index=[d - 1, d - 1])
            point = vectors[:, 0]
            data = {"h_actions_per_start": None}
        elif method in ("exp_rgd100", "relaxed_pg100", "hard_pg100", "normalized_pg100", "exp_rgd20", "normalized_pg20", "exp_rgd40", "normalized_pg40"):
            initial = uniform_parameters(rng, 1, d, domain)[0]
            point, data = search(H, initial, steps=config["updates"], domain=domain,
                                 alpha=config["sphere_alpha"],
                                 omega=1.0 if method in ("hard_pg100", "normalized_pg100", "normalized_pg20", "normalized_pg40") else config["ball_omega"],
                                 ball_step_rule="normalized" if method in ("normalized_pg100", "normalized_pg20", "normalized_pg40") else "fixed", diagnostics=False)
        else:
            if method.startswith("fresh_pool"):
                pool = uniform_parameters(pool_rng, config["updates"] + 1, d, domain)
            point, data = pool_search(H, pool)
        costs["selector_seconds"] += time.perf_counter() - begin
        if data["h_actions_per_start"] is not None:
            h_actions += data["h_actions_per_start"]
        # These exact audits are excluded from selector timing and are not used
        # by the optimization or pool selectors to choose their parameter.
        top = max(float(la.eigvalsh(H)[-1]), 0.0)
        quality = float(np.sqrt(max(float(point @ H @ point), 0.0) / top))
        begin = time.perf_counter()
        novelty = rom.append(point)
        costs["enrichment_seconds"] += time.perf_counter() - begin
        _, G = rom.matrices()
        errors.append(float(np.sqrt(max(float(la.eigvalsh(G)[-1]), 0.0))))
        rounds.append({"n": n, "indicator_quality": quality, "parameter": point.tolist(),
                       "snapshot_novelty": novelty, **data})
    return {"domain": domain, "method": method, "dimension": d, "seed": seed,
            "errors": errors, "rounds": rounds, "costs": costs,
            "timed_total_seconds": sum(costs.values()),
            "h_actions_total": None if method == "exact_indicator" else h_actions,
            "a_orthogonality_defect": rom.orthogonality_defect()}


def frozen_audits(G0, H0, domain, config, output):
    """All starts/methods see the same exact-indicator reference space."""
    d = len(G0)
    rom = CoordinateROM(G0, H0)
    ranks = {0, d // 4, d // 2}
    records = []
    for n in range(d // 2 + 1):
        H, _ = rom.matrices(include_error=False)
        if n in ranks:
            np.savez_compressed(output / f"frozen_{domain}_d{d}_n{n}.npz", H=H, Q=rom.Q)
            eigenvalues = la.eigvalsh(H)
            threshold = config["gamma"]**2 * float(eigenvalues[-1])
            audit_seed = config["audit_seed"] + 100*d + n
            starts = uniform_parameters(np.random.default_rng(audit_seed), config["audit_starts"], d, domain)
            single_hits = int(np.count_nonzero(np.sum(starts * (starts @ H.T), axis=1) >= threshold))
            p_hat = single_hits / len(starts)
            p_ci = confidence_interval(single_hits, len(starts))
            p_upper = direct_probability_upper(H, config["gamma"], domain)
            modes = [("exp_rgd100", 0.5)] if domain == "sphere" else [("relaxed_pg100", config["ball_omega"]), ("hard_pg100", 1.0)]
            trajectories = {}
            for method, omega in modes:
                _, data = search(H, starts, config["updates"], domain, config["sphere_alpha"], omega,
                                 checkpoints=config["checkpoints"], threshold=threshold)
                for t, entry in data["snapshots"].items():
                    entry["q_hat"] = entry["hits"] / len(starts)
                    entry["q_ci95"] = confidence_interval(entry["hits"], len(starts))
                    if domain == "ball":
                        entry["q_analytic_lower"] = float(ball_probability_lower(H, config["gamma"], int(t), omega))
                trajectories[method] = data
            pools = {}
            for m in [config["updates"] + 1] + config["pool_sizes"]:
                pools[str(m)] = {
                    "fresh_iid_probability_estimate": pool_probability(p_hat, m),
                    "fresh_iid_probability_ci95": [pool_probability(p, m) for p in p_ci],
                    "fresh_iid_probability_analytic_upper": pool_probability(p_upper, m),
                }
            records.append({"domain": domain, "dimension": d, "rank": n, "audit_seed": audit_seed,
                            "audit_starts": len(starts), "gamma": config["gamma"],
                            "lambda1": float(eigenvalues[-1]), "lambda2_ratio": float(eigenvalues[-2]/eigenvalues[-1]),
                            "single_hit_count": single_hits, "single_hit_probability": p_hat,
                            "single_hit_ci95": p_ci, "single_hit_analytic_upper": p_upper,
                            "fresh_iid_pools": pools, "trajectories": trajectories})
            print(f"  frozen {domain} d={d} n={n}: direct {single_hits}/{len(starts)}, "
                  + ", ".join(f"{m} q100={v['snapshots'][str(config['updates'])]['q_hat']:.4f}" for m, v in trajectories.items()), flush=True)
        if n < d // 2:
            _, vectors = la.eigh(H, subset_by_index=[d - 1, d - 1])
            rom.append(vectors[:, 0])
    return records


LABELS = {"exact_indicator": "Exact indicator", "exp_rgd100": "Sphere Exp RGD (100)",
          "relaxed_pg100": "Relaxed PG (100), omega=0.5", "hard_pg100": "Hard PG (100), omega=1",
          "fixed_pool256": "Fixed pool 256", "fixed_pool4096": "Fixed pool 4096",
          "fixed_pool10000": "Fixed pool 10000", "fresh_pool101": "Fresh pool 101"}
COLORS = {"exact_indicator": "#333333", "exp_rgd100": "#005ca9", "relaxed_pg100": "#005ca9",
          "hard_pg100": "#763bac", "fixed_pool256": "#d67a13", "fixed_pool4096": "#2d934b",
          "fixed_pool10000": "#bb4145", "fresh_pool101": "#888888"}


def create_outputs(output, config, all_runs, audits, problems):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "font.family": "DejaVu Sans"})
    summary = []
    for domain in ("sphere", "ball"):
        dimensions = config["dimensions"]
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
        for d, ax in zip(dimensions, axes.flat):
            width = np.array(problems[str(d)]["hf_widths"])
            x = np.arange(d // 2 + 1)
            ax.semilogy(x, width[:len(x)], "k--", lw=1.4, label="HF width")
            for method in method_names(domain, config["pool_sizes"]):
                runs = [r for r in all_runs if r["domain"] == domain and r["dimension"] == d and r["method"] == method]
                curves = np.array([r["errors"] for r in runs])
                median = np.median(curves, axis=0)
                color = COLORS.get(method)
                ax.semilogy(x, median, color=color, lw=1.6, ls=":" if method in ("hard_pg100", "fresh_pool101") else "-", label=LABELS.get(method, method))
                if len(runs) > 1:
                    ax.fill_between(x, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.1, linewidth=0)
                qualities = np.array([r["rounds"][-1]["indicator_quality"] for r in runs])
                summary.append({"domain": domain, "dimension": d, "method": method, "seeds": len(runs),
                                "final_error_median": float(median[-1]), "final_error_over_width_median": float(median[-1]/width[d//2]),
                                "final_quality_min": float(qualities.min()), "final_quality_median": float(np.median(qualities)),
                                "timed_seconds_median": float(np.median([r["timed_total_seconds"] for r in runs])),
                                "selector_seconds_median": float(np.median([r["costs"]["selector_seconds"] for r in runs]))})
            ax.set_title(f"d = {d}")
            ax.set_xlabel("Reduced basis dimension n")
            ax.set_ylabel("Largest HF energy error")
            ax.grid(alpha=0.2)
        axes.flat[0].legend(fontsize=8, loc="lower left")
        fig.suptitle(f"{'Round sphere' if domain == 'sphere' else 'Closed Euclidean ball'} baseline: local CPU, separate study\nMedian and min/max over {len(config['seeds'])} seeds; optimization returns best iterate")
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{domain}_greedy.{extension}")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
        for d, ax in zip(dimensions, axes.flat):
            record = next(r for r in audits if r["domain"] == domain and r["dimension"] == d and r["rank"] == d//2)
            for method, data in record["trajectories"].items():
                entries = list(data["snapshots"].items())
                times = [int(t) for t, _ in entries]
                q = [s["q_hat"] for _, s in entries]
                ax.plot(times, q, "o-", ms=3, color=COLORS[method], label=LABELS[method])
                ax.fill_between(times, [s["q_ci95"][0] for _, s in entries], [s["q_ci95"][1] for _, s in entries], color=COLORS[method], alpha=0.16)
                if domain == "ball":
                    ax.plot(times, [s["q_analytic_lower"] for _, s in entries], "--", color=COLORS[method], label=f"Analytic lower, omega={config['ball_omega'] if method == 'relaxed_pg100' else 1:g}")
            upper = record["fresh_iid_pools"]["10000"]["fresh_iid_probability_analytic_upper"] if "10000" in record["fresh_iid_pools"] else record["fresh_iid_pools"][str(config["pool_sizes"][-1])]["fresh_iid_probability_analytic_upper"]
            ax.axhline(upper, color=COLORS["fixed_pool10000"], ls=":", label="Fresh iid pool 10000: upper bound")
            ax.set(title=f"d={d}, frozen reference V_{d//2}", xlabel="Inner updates T", ylabel="Threshold success probability", ylim=(-0.025, 1.025))
            ax.grid(alpha=0.2)
        axes.flat[0].legend(fontsize=8, loc="lower right")
        fig.suptitle(f"{domain.capitalize()}: q_T at gamma={config['gamma']}; {config['audit_starts']} independent starts\nExact binomial 95% intervals; same frozen objective for all searches")
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{domain}_success.{extension}")
        plt.close(fig)
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# Local CPU geometry baselines", "", "Sphere and ball are reported as separate experiments. Both retain the same manufactured linear diffusion PDE and P3 FEM truth model.", "", "## Protocol", "",
             f"Dimensions: {config['dimensions']}; grid={config['grid']}; seeds={config['seeds']}; rho={config['rho']}; contrast={config['contrast']}.",
             f"Updates={config['updates']}; sphere alpha={config['sphere_alpha']}; ball omega={config['ball_omega']}, with omega=1 as a companion. Objective: J/||H||_F.",
             "Sphere uses its round metric, tangent derivative, and exponential map. Ball uses the Euclidean metric and relaxed nearest-point projection. Ball radii are U^(1/d), so initialization and pools are uniform in volume.",
             "Every method uses float64 NumPy/SciPy on the same local CPU with one BLAS thread. Common FEM construction and H0=B.T X^(-1) B preprocessing are timed separately. Dense eigensolvers are an oracle reference. Selector costs exclude exact-quality and exact-error audits.",
             "The ROM representation V=S Q uses exact small matrices from the assembled FEM PDE; it is not a diagonal substitute. The fixed pools are reused along their own greedy history. Fresh pool 101 is redrawn each round. All searches include initialization in their best-iterate return.",
             "", "## Final greedy errors", "", "| Domain | d | Method | Median error / HF width | Median seconds |", "|---|---:|---|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['domain']} | {row['dimension']} | {row['method']} | {row['final_error_over_width_median']:.4g} | {row['timed_seconds_median']:.4g} |")
    lines += ["", "## Probability audits", "", "These are independent starts on frozen exact-indicator reference spaces at n=0,d/4,d/2. They are not the random pools, and are not drawn from the different greedy histories of the compared methods. q_T counts threshold attainment by any iterate 0..T. Binomial intervals are pointwise, not simultaneous bands.",
              "For a fresh iid pool of size M at a frozen space, success is 1-(1-p)^M. This identity is not asserted conditionally for the reused fixed pool. Zero observed direct hits does not establish p=0; exact confidence intervals and spectral upper bounds are retained in the JSON.",
              "The ball analytical lower bound uses c=1-omega+omega*a/b, a=1+lambda1/||H||_F, b=sqrt(1+gamma^2*(a^2-1)); q_T >= P[Beta(1/2,(d+1)/2) >= gamma^2*c^(-2T)]. It applies to the discrete relaxed update, including PSD matrices with a kernel. The historical normalized-Euclidean sphere bound is not assigned to the new exponential update.",
              "For omega<1 an initially interior ball iterate remains interior in exact arithmetic at every finite step. Projection activity refers to the unconstrained trial; reported boundary contact uses a floating-point tolerance and is not a theorem of finite identification.",
              "", "## Files", "", "- `results.json`: raw greedy histories and timing data.", "- `probability_audits.json`: direct sampling, optimization q_T, confidence intervals and analytical bounds.", "- `fem_d*.npz`, `frozen_*.npz`: assembled matrices and frozen reference spaces for reproduction.", "- `sphere_greedy.*`, `ball_greedy.*`: separate greedy curves.", "- `sphere_success.*`, `ball_success.*`: separate success-probability plots.", "- `config.json`, `environment.json`, `validation.json`: protocol, CPU/dependency details and numerical checks."]
    (output / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--audit-starts", type=int, default=8192)
    args = parser.parse_args()
    if len(args.dimensions) > 4 or any(d < 4 or d % 4 for d in args.dimensions):
        parser.error("Provide at most four dimensions, each a positive multiple of four")
    if args.seed_count < 1 or args.audit_starts < 1 or args.grid < 1:
        parser.error("Seed count, audit start count, and grid must be positive")
    config = {"dimensions": args.dimensions, "grid": args.grid, "element_order": 3,
              "rho": 0.78, "contrast": 0.25, "seeds": list(range(1000, 1000 + args.seed_count)),
              "updates": 100, "sphere_alpha": 0.2, "ball_omega": 0.5,
              "pool_sizes": [256, 4096, 10000], "gamma": 0.9, "audit_seed": 20261012,
              "audit_starts": args.audit_starts, "checkpoints": [0, 1, 2, 5, 10, 25, 50, 100],
              "device": "local_cpu", "dtype": "float64", "blas_threads": 1,
              "sphere_metric": "unit round sphere, induced Euclidean tangent metric",
              "ball_metric": "Euclidean", "sphere_law": "uniform surface",
              "ball_law": "uniform volume", "objective": "J=z.T H z, search uses J/||H||_F"}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    with threadpool_limits(limits=1):
        import skfem
        environment = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                       "executable": sys.executable, "platform": platform.platform(), "machine": platform.machine(),
                       "host": platform.node(), "logical_cpu_count": os.cpu_count(), "numpy": np.__version__,
                       "scipy": scipy.__version__, "scikit_fem": skfem.__version__, "scikit_fem_path": skfem.__file__,
                       "backend": "NumPy/SciPy only; no JAX/PyTorch import", "threadpools": threadpool_info(),
                       "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(__file__).with_name('geometry_cpu.py'), Path(__file__).with_name('continuous_fem.py'))}}
        write_json(output / "environment.json", environment)
        all_runs, audits, problems = [], [], {}
        begin_all = time.perf_counter()
        for d in args.dimensions:
            begin = time.perf_counter()
            problem = FEMProblem.build(args.grid, d, contrast=config["contrast"], spectrum_ratio=config["rho"], element_order=3)
            construction = time.perf_counter() - begin
            begin = time.perf_counter()
            H0 = sym(problem.B.T @ problem.solve_X(problem.B))
            G0 = problem.solution_gram
            preprocessing = time.perf_counter() - begin
            problems[str(d)] = {"dofs": problem.dofs, "element_order": problem.element_order,
                                "quadrature_order": problem.quadrature_order, "projection_error": problem.projection_error,
                                "hf_widths": problem.singular_values.tolist(), "fem_construction_seconds": construction,
                                "shared_residual_preprocessing_seconds": preprocessing}
            np.savez_compressed(output / f"fem_d{d}.npz", G0=G0, H0=H0, hf_widths=problem.singular_values,
                                source_transform=problem.source_transform, modes=problem.modes)
            print(f"FEM d={d}, dofs={problem.dofs}, construction={construction:.2f}s, shared preprocessing={preprocessing:.2f}s", flush=True)
            for domain in ("sphere", "ball"):
                for method in method_names(domain, config["pool_sizes"]):
                    runs = [run_greedy(G0, H0, domain, method, seed, config) for seed in config["seeds"]]
                    all_runs.extend(runs)
                    print(f"  {domain} {method}: {len(runs)} seeds; final error/width="
                          f"{np.median([r['errors'][-1] for r in runs])/problem.singular_values[d//2]:.4f}; "
                          f"median timed={np.median([r['timed_total_seconds'] for r in runs]):.3f}s", flush=True)
                    write_json(output / "results.json", {"config": config, "problems": problems, "runs": all_runs})
                audits.extend(frozen_audits(G0, H0, domain, config, output))
                write_json(output / "probability_audits.json", audits)
            del problem
        max_defect = max(r["a_orthogonality_defect"] for r in all_runs)
        max_feasibility = max(s["max_feasibility_defect"] for r in audits for s in r["trajectories"].values())
        minimum_scaled_increment = min(s["minimum_objective_increment"]/r["lambda1"] for r in audits for s in r["trajectories"].values())
        validation = {"completed_greedy_runs": len(all_runs), "frozen_objectives": len(audits),
                      "max_a_orthogonality_defect": max_defect, "max_feasibility_defect": max_feasibility,
                      "minimum_objective_increment_over_lambda1": minimum_scaled_increment,
                      "blas_one_thread": all(p["num_threads"] == 1 for p in threadpool_info()),
                      "gpu_backends_loaded": [name for name in ("jax", "torch", "cupy") if name in sys.modules],
                      "all_finite": all(np.isfinite(r["errors"]).all() for r in all_runs),
                      "actual_elapsed_seconds_before_plots": time.perf_counter() - begin_all}
        if (max_defect > 1e-8 or max_feasibility > 1e-12 or not validation["all_finite"]
                or not validation["blas_one_thread"] or validation["gpu_backends_loaded"]):
            raise RuntimeError(f"Numerical validation failed: {validation}")
        write_json(output / "validation.json", validation)
        create_outputs(output, config, all_runs, audits, problems)
        environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
        environment["wall_seconds_including_plots"] = time.perf_counter() - begin_all
        write_json(output / "environment.json", environment)
        print(f"Complete: {output}, elapsed={environment['wall_seconds_including_plots']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
