"""Cube baseline using the unchanged assembled diffusion family; CPU only.

Run: .venv_cpu/bin/python -m examples.baseline.run_cube_cpu
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/continuous_cube_matplotlib")
import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import beta
from threadpoolctl import threadpool_limits, threadpool_info
from .geometry_cpu import CoordinateROM, pool_search
from .cube_geometry_cpu import vertices, values, width_bounds, cube_search


def write(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def interval(hits, count):
    return [0.0 if hits == 0 else float(beta.ppf(.025, hits, count-hits+1)),
            1.0 if hits == count else float(beta.ppf(.975, hits+1, count-hits))]


def pool_probability(p, count):
    return 1.0 if p >= 1 else float(-np.expm1(count*np.log1p(-p)))


def run_greedy(G0, H0, vertex, method, seed, config):
    d = len(G0)
    rng = np.random.default_rng(seed)
    pool_rng = np.random.default_rng(seed+1_000_003)
    rom = CoordinateROM(G0, H0)
    fixed = pool_rng.uniform(-1, 1, (int(method.removeprefix("fixed_pool")), d)) if method.startswith("fixed_pool") else None
    errors = [float(np.sqrt(np.max(values(G0, vertex))))]
    rows = []
    selector_seconds = 0.0
    for n in range(d//2):
        H, _ = rom.matrices()
        begin = time.perf_counter()
        if method in ("grid_vertex_reduction", "exact_vertex_reference"):
            point = vertex[int(np.argmax(values(H, vertex)))].copy()
        elif method in ("relaxed_pg100", "hard_pg100"):
            point, _ = cube_search(H, rng.uniform(-1, 1, d), config["updates"],
                                   1.0 if method == "hard_pg100" else config["omega"])
        else:
            point, _ = pool_search(H, fixed)
        selector_seconds += time.perf_counter()-begin
        maximum = float(np.max(values(H, vertex)))
        quality = float(np.sqrt(max(float(point @ H @ point), 0)/maximum))
        rom.append(point)
        _, G = rom.matrices()
        errors.append(float(np.sqrt(max(float(np.max(values(G, vertex))), 0))))
        rows.append({"n": n, "parameter": point.tolist(), "indicator_quality": quality})
    return {"domain": "cube", "dimension": d, "method": method, "seed": seed,
            "errors": errors, "rounds": rows, "selector_seconds": selector_seconds,
            "a_orthogonality_defect": rom.orthogonality_defect()}


def frozen(G0, H0, vertex, config, output):
    d = len(G0)
    rom = CoordinateROM(G0, H0)
    records = []
    for n in range(d//2+1):
        H, _ = rom.matrices()
        maximum = float(np.max(values(H, vertex)))
        if n in {0, d//4, d//2}:
            np.savez_compressed(output/f"frozen_cube_d{d}_n{n}.npz", H=H, Q=rom.Q)
            seed = config["audit_seed"]+100*d+n
            starts = np.random.default_rng(seed).uniform(-1, 1, (config["audit_starts"], d))
            threshold = config["gamma"]**2*maximum
            hits = int(np.sum(values(H, starts) >= threshold))
            ci = interval(hits, len(starts))
            trajectories = {}
            for method, omega in (("relaxed_pg100", config["omega"]), ("hard_pg100", 1.0)):
                _, data = cube_search(H, starts, config["updates"], omega, config["checkpoints"], threshold)
                for item in data["snapshots"].values():
                    item["q_ci95"] = interval(item["hits"], len(starts))
                trajectories[method] = data
            records.append({"domain": "cube", "dimension": d, "rank": n,
                            "reference": "exact vertex-indicator greedy space",
                            "audit_seed": seed, "audit_starts": len(starts),
                            "gamma": config["gamma"], "J_max": maximum,
                            "single_hit_count": hits, "single_hit_probability": hits/len(starts),
                            "single_hit_ci95": ci, "trajectories": trajectories,
                            "fresh_iid_pools": {str(m): {
                                "probability_estimate": pool_probability(hits/len(starts), m),
                                "probability_ci95": [pool_probability(p, m) for p in ci]}
                                for m in [101]+config["pool_sizes"]}})
            print(f"frozen d={d} n={n}: q0={hits/len(starts):.5f}, q100={trajectories['relaxed_pg100']['snapshots']['100']['q_hat']:.5f}", flush=True)
        if n < d//2:
            rom.append(vertex[int(np.argmax(values(H, vertex)))])
    return records


def plots(output, config, runs, problems, audits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    methods = ["relaxed_pg100", "grid_vertex_reduction"]+[f"fixed_pool{m}" for m in config["pool_sizes"]]
    labels = {"relaxed_pg100": "Projected GD, 100 steps", "grid_vertex_reduction": "Grid h=1, 1/2, 1/4 (identical)"}
    colors = ["#265B7D", "#242C33", "#B48442", "#55806C", "#9B6267"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    summary = []
    for ax, d in zip(axes.flat, config["dimensions"]):
        bounds = problems[str(d)]["width_bounds"]
        lower = np.array(bounds["lower"][:d//2+1])
        upper = np.array(bounds["upper"][:d//2+1])
        for method, color in zip(methods, colors):
            if method == "grid_vertex_reduction" and d > 6:
                continue
            selected = [r for r in runs if r["dimension"]==d and r["method"]==method]
            errors = np.array([r["errors"] for r in selected])
            median = np.median(errors, axis=0)
            ax.plot(range(len(median)), median/lower, color=color, label=labels.get(method, method.replace("fixed_pool", "Random pool ")))
            ax.fill_between(range(len(median)), errors.min(axis=0)/upper, errors.max(axis=0)/lower, color=color, alpha=.13)
            summary.append({"dimension": d, "method": method, "final_ratio_lower": float(median[-1]/upper[-1]),
                            "final_ratio_upper": float(median[-1]/lower[-1]), "seeds": len(selected)})
        ax.axhline(1, color="gray", ls=":", lw=.8)
        ax.set(title=f"d={d}", xlabel="Reduced dimension N", ylabel=r"$E_N/d_N$ (certified interval)")
        ax.grid(alpha=.18)
    axes.flat[-1].axis("off")
    handles, labels_ = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels_, loc="center", frameon=False, fontsize=10)
    fig.suptitle("Cube parameter baseline: unchanged linear diffusion PDE\nMedian; shading includes seed range and width-certification interval")
    for ext in ("png", "pdf"):
        fig.savefig(output/f"cube_greedy_ratio.{ext}", dpi=180)
    plt.close(fig)
    write(output/"summary.json", summary)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for d in config["dimensions"]:
        record = next(r for r in audits if r["dimension"]==d and r["rank"]==d//2)
        entries = record["trajectories"]["relaxed_pg100"]["snapshots"]
        ax.plot([int(t) for t in entries], [v["q_hat"] for v in entries.values()], marker=".", label=f"d={d}")
    ax.set(xlabel="Updates T", ylabel="Empirical success probability", ylim=(-.02, 1.02), title="Cube: frozen exact-vertex reference spaces, N=d/2")
    ax.legend(ncol=3)
    ax.grid(alpha=.2)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"cube_success.{ext}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/cube_cpu_20261015"))
    parser.add_argument("--source", type=Path, default=Path("runs/geometry_cpu_20261012/fem_d16.npz"))
    args = parser.parse_args()
    config = {"dimensions": [4, 6, 8, 10, 12], "seeds": list(range(1000,1008)),
              "updates": 100, "omega": .5, "gamma": .9, "pool_sizes": [256,4096,10000],
              "grid_h": [1,.5,.25], "audit_seed": 20261015, "audit_starts": 16384,
              "checkpoints": [0,1,2,3,5,10,15,20,30,50,75,100], "device": "local_cpu",
              "dtype": "float64", "blas_threads": 1, "parameter_domain": "[-1,1]^d",
              "initialization": "uniform volume", "source_matrices": str(args.source.resolve()),
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "source_note": "Leading principal matrices of the existing d=16 P3 FEM family; identical PDE, mesh, quadrature and first d loads. No recalibration.",
              "grid_note": "Tensor grids include all cube vertices. PSD quadratic maxima occur at vertices, so all three h give exactly the same selector. Nominal counts are not actual evaluation counts."}
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/"config.json", config)
    source = np.load(args.source)
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                       "numpy": np.__version__, "scipy": scipy.__version__, "threadpools": threadpool_info(),
                       "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(__file__).with_name("cube_geometry_cpu.py"))}}
        write(args.output/"environment.json", environment)
        runs, audits, problems, grid_checks = [], [], {}, []
        for d in config["dimensions"]:
            G0, H0 = source["G0"][:d,:d], source["H0"][:d,:d]
            vertex = vertices(d)
            lower, upper = width_bounds(G0, vertex)
            problems[str(d)] = {"width_bounds": {"lower": lower.tolist(), "upper": upper.tolist()},
                               "width_max_relative_gap": float(np.max(upper/lower-1)),
                               "grid_sizes": {str(h): int(round(2/h)+1)**d for h in config["grid_h"]},
                               "evaluated_vertices": len(vertex)}
            np.savez_compressed(args.output/f"fem_cube_d{d}.npz", G0=G0, H0=H0, width_lower=lower, width_upper=upper)
            methods = ["grid_vertex_reduction" if d <= 6 else "exact_vertex_reference", "relaxed_pg100", "hard_pg100"]+[f"fixed_pool{m}" for m in config["pool_sizes"]]
            for method in methods:
                seeds = config["seeds"] if method not in ("grid_vertex_reduction", "exact_vertex_reference") else [config["seeds"][0]]
                for seed in seeds:
                    runs.append(run_greedy(G0,H0,vertex,method,seed,config))
                print(f"d={d} {method}: completed", flush=True)
            audits.extend(frozen(G0,H0,vertex,config,args.output))
            if d==4:
                rom = CoordinateROM(G0,H0)
                for n in range(3):
                    H,_ = rom.matrices()
                    vmax = float(np.max(values(H,vertex)))
                    for h in config["grid_h"]:
                        grid = np.array(list(itertools.product(np.linspace(-1,1,round(2/h)+1),repeat=d)))
                        grid_checks.append({"d":d,"n":n,"h":h,"actual_grid_points":len(grid),
                                            "max_difference":abs(float(np.max(values(H,grid)))-vmax)})
                    if n<2:
                        rom.append(vertex[int(np.argmax(values(H,vertex)))])
            write(args.output/"results.json", {"config":config,"problems":problems,"runs":runs})
            write(args.output/"probability_audits.json",audits)
        validation = {"grid_vertex_checks":grid_checks,"max_orthogonality_defect":max(r["a_orthogonality_defect"] for r in runs),
                      "max_width_relative_gap":max(p["width_max_relative_gap"] for p in problems.values()),
                      "minimum_objective_increment":min(t["minimum_increment"] for a in audits for t in a["trajectories"].values()),
                      "blas_one_thread":all(p["num_threads"]==1 for p in threadpool_info()),
                      "gpu_backends_loaded":[n for n in ("jax","torch","cupy") if n in sys.modules],
                      "elapsed_before_plots":time.perf_counter()-start}
        write(args.output/"validation.json",validation)
        plots(args.output,config,runs,problems,audits)
        environment["elapsed_total_seconds"] = time.perf_counter()-start
        write(args.output/"environment.json",environment)
        (args.output/"README.md").write_text("# Cube CPU baseline\n\nThe unchanged manufactured diffusion FEM matrices are inherited from the d=16 sphere/ball run. Cube maxima are audited on every vertex. Kolmogorov widths are certified between a tail-trace lower bound and an SVD-space vertex-error upper bound; plots retain the interval.\n\nUniform grids h=1,1/2,1/4 contain every vertex and therefore have identical quadratic maximizers. Their full nominal counts are recorded, but their reported selector time uses exact vertex reduction. The d=4 grids were also explicitly enumerated at three greedy ranks.\n\nFixed random pools are reused along each method's own greedy sequence. Probability audits use independent uniform starts at the same frozen exact-vertex reference spaces. Fresh iid pool probabilities derive from the single-draw estimate and its exact binomial interval; these are not conditional success probabilities of reused greedy pools. Analytical probability certificates are supplied separately.\n")
        print(f"Complete {args.output}; {environment['elapsed_total_seconds']:.1f}s",flush=True)


if __name__ == "__main__":
    main()
