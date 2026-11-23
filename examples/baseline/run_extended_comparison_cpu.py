"""Extended CPU sphere/ball comparisons, without changing the original runner.

Uses the assembled diffusion FEM coefficient matrices, deterministic tensor
grids, and independent frozen-space probability experiments.  Run with:
  PYTHONPATH=src:. .venv_cpu/bin/python -u -m examples.baseline.run_extended_comparison_cpu
"""
from __future__ import annotations

import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/contgreedy_matplotlib")

import argparse
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
from scipy.special import betaincc
from threadpoolctl import threadpool_info, threadpool_limits

from .continuous_fem import FEMProblem
from .geometry_cpu import (CoordinateROM, ball_probability_lower, ball_normalized_probability_lower,
                           direct_probability_upper, pool_search, search, sym,
                           uniform_parameters)
from .run_geometry_cpu import confidence_interval, pool_probability, write_json


def tensor_candidates(d, h, domain):
    """Actual Cartesian tensor grid; sphere directions are its radial image."""
    intervals = round(2 / h)
    axis = np.linspace(-1.0, 1.0, intervals + 1)
    raw = np.stack(np.meshgrid(*([axis] * d), indexing="ij"), axis=-1).reshape(-1, d)
    radius = np.linalg.norm(raw, axis=1)
    if domain == "ball":
        candidates = raw[radius <= 1.0 + 1e-14]
    else:
        keep = radius > 0
        normalized = raw[keep] / radius[keep, None]
        # Preserve original float64 coordinates; rounding only identifies
        # rays that differ by a positive scalar within this exact rational grid.
        _, indices = np.unique(np.round(normalized, 13), axis=0, return_index=True)
        candidates = normalized[np.sort(indices)]
    return candidates, {"h": h, "tensor_count_before_filtering": len(raw),
                         "actual_candidate_count": len(candidates),
                         "axis_count": len(axis), "dimension": d,
                         "domain": domain,
                         "construction": ("Cartesian tensor grid restricted to closed unit ball"
                                          if domain == "ball" else
                                          "Radial image of nonzero Cartesian tensor grid; duplicate rays removed; not surface-area-uniform")}


def selector_run(G0, H0, domain, method, seed, updates, max_rank, pool=None, omega=0.5, ball_step_rule="fixed", rom_kwargs=None, alpha=0.2):
    rom = CoordinateROM(G0, H0, **(rom_kwargs or {}))
    rng = np.random.default_rng(seed)
    if method.startswith("pool"):
        pool = uniform_parameters(np.random.default_rng(seed + 1_000_003),
                                  int(method[4:]), len(G0), domain)
    errors = [float(np.sqrt(max(la.eigvalsh(G0)[-1], 0)))]
    rounds = []
    selector_seconds = 0.0
    for n in range(max_rank):
        H, _ = rom.matrices(include_error=False)
        begin = time.perf_counter()
        if method == "oracle":
            _, vectors = la.eigh(H, subset_by_index=[len(H) - 1, len(H) - 1])
            point = vectors[:, 0]
            actions = None
        elif method == "gd":
            point, data = search(H, uniform_parameters(rng, 1, len(H), domain)[0],
                                 steps=updates, domain=domain, alpha=alpha, omega=omega, ball_step_rule=ball_step_rule, diagnostics=False)
            actions = data["h_actions_per_start"]
        else:
            point, data = pool_search(H, pool)
            actions = len(pool)
        selector_seconds += time.perf_counter() - begin
        top = max(float(la.eigvalsh(H)[-1]), 0)
        quality = float(np.sqrt(max(point @ H @ point, 0) / top))
        novelty = rom.append(point)
        _, G = rom.matrices()
        errors.append(float(np.sqrt(max(la.eigvalsh(G)[-1], 0))))
        rounds.append({"n": n, "parameter": point.tolist(), "quality": quality,
                       "novelty": novelty, "indicator_evaluations": actions})
    return {"domain": domain, "dimension": len(G0), "method": method,
            "seed": seed, "errors": errors, "rounds": rounds,
            "selector_seconds": selector_seconds,
            "a_orthogonality_defect": rom.orthogonality_defect()}


def sphere_probability_lower(H, gamma, times, alpha=0.2):
    """Exp-map bound using r <= alpha*sigma/sqrt(1+sigma**2).

    Requires 1-alpha**2*sigma**2/(6*(1+sigma**2))-alpha*gamma**2*sigma > 0.
    Uses sin(r)/r >= 1-r**2/6 and cos(r) >= 1-r**2/2,
    retaining the sharper angular bound supplied by the normalized step.
    """
    top = float(la.eigvalsh(H)[-1])
    sigma = top / float(np.linalg.norm(H, "fro"))
    factor = 1.0 - alpha**2 * sigma**2 / (6.0 * (1.0 + sigma**2)) - alpha * gamma**2 * sigma
    if factor <= 0:
        raise ValueError("Sphere certificate requires a positive multiplier bound")
    c = 1 + 2 * alpha * sigma * (1 - gamma**2) / np.sqrt(1 + sigma**2) * factor
    cutoff = gamma**2 * c ** (-2 * np.asarray(times))
    return betaincc(0.5, (len(H) - 1) / 2, cutoff), {"sigma": sigma, "c": c}


def frozen_space(G0, H0, n, rom_kwargs=None):
    rom = CoordinateROM(G0, H0, **(rom_kwargs or {}))
    for _ in range(n):
        H, _ = rom.matrices(include_error=False)
        _, vectors = la.eigh(H, subset_by_index=[len(H) - 1, len(H) - 1])
        rom.append(vectors[:, 0])
    H, _ = rom.matrices(include_error=False)
    return H, rom.Q


def batch_probability(H, domain, count, steps, seed, gamma, output, rank, ball_omegas=(0.5, 1.0), ball_step_rule="fixed", alpha=0.2):
    """Independent search trajectories and independent iid pools on one H.

    Orthogonal eigencoordinates make each matrix action elementwise without
    changing the map, domain, or initialization law.  The saved eigensystem
    ties this computation to the actual assembled FEM objective.
    """
    d = len(H)
    eigenvalues, vectors = la.eigh(H)
    scale = float(np.linalg.norm(eigenvalues))
    threshold = gamma**2 * eigenvalues[-1]
    Hdiag = np.diag(eigenvalues)
    rng = np.random.default_rng(seed)
    starts = uniform_parameters(rng, count, d, domain)
    modes = [("exp_rgd", None)] if domain == "sphere" else [("normalized_pg" if ball_step_rule == "normalized" else "hard_pg" if w == 1.0 else "relaxed_pg", w) for w in ball_omegas]
    methods = {}
    raw_hits = {}
    early_iterates = {}
    max_defect = 0.0
    minimum_increment = np.inf
    for name, omega in modes:
        z = starts.copy()
        hits = np.full(count, -1, dtype=np.int16)
        previous = None
        snapshots = []
        for t in range(steps + 1):
            if t == min(steps, 7):
                early_iterates[name] = z[:16].copy()
            hz = z * eigenvalues
            values = np.sum(z * hz, axis=1)
            hits[(hits < 0) & (values >= threshold)] = t
            if previous is not None:
                minimum_increment = min(minimum_increment, float(np.min(values - previous) / eigenvalues[-1]))
            previous = values
            nhit = int(np.count_nonzero(hits >= 0))
            snapshots.append({"T": t, "hits": nhit, "q_hat": nhit / count,
                              "ci95": confidence_interval(nhit, count)})
            radii = np.linalg.norm(z, axis=1)
            defect = np.abs(radii - 1) if domain == "sphere" else np.maximum(radii - 1, 0)
            max_defect = max(max_defect, float(np.max(defect)))
            if t == steps:
                break
            if domain == "sphere":
                derivative = 2 * (hz - values[:, None] * z) / scale
                tangent = alpha * derivative / np.sqrt(1 + np.sum(derivative**2, axis=1, keepdims=True))
                length = np.linalg.norm(tangent, axis=1, keepdims=True)
                z = np.cos(length) * z + np.sinc(length / np.pi) * tangent
                z /= np.linalg.norm(z, axis=1, keepdims=True)
            else:
                derivative = 2*hz/scale
                eta = (alpha/np.sqrt(1+np.sum(derivative**2,axis=1,keepdims=True))
                       if ball_step_rule == "normalized" else 0.5)
                trial = z + eta*derivative
                z = (1 - omega) * z + omega * trial / np.maximum(1, np.linalg.norm(trial, axis=1, keepdims=True))
        times = np.arange(steps + 1)
        if domain == "sphere":
            # The coefficient proof applies whenever its bracket is positive;
            # alpha=1, gamma=.9 satisfies this since sigma<=1.
            lower, certificate = sphere_probability_lower(Hdiag, gamma, times, alpha)
        elif ball_step_rule == "normalized":
            assert omega == 1.0
            lower, certificate = ball_normalized_probability_lower(Hdiag, gamma, times, alpha)
        else:
            lower = ball_probability_lower(Hdiag, gamma, times, omega)
            certificate = {"omega": omega}
        for row, bound in zip(snapshots, lower):
            row["analytic_lower"] = None if bound is None else float(bound)
        methods[name] = {"snapshots": snapshots, "certificate": certificate}
        raw_hits[name] = hits
    # Each row is one independent iid pool, nested as its size grows.  Sharing
    # its first candidate with a GD start creates a paired experiment, not a
    # selection of the best GD trajectory among the 4096 starts.
    pool_rng = np.random.default_rng(seed + 7_000_019)
    pool_hits = np.full(count, -1, dtype=np.int16)
    pool_rows = []
    direct_upper = direct_probability_upper(Hdiag, gamma, domain)
    for t in range(steps + 1):
        point = starts if t == 0 else uniform_parameters(pool_rng, count, d, domain)
        values = np.sum(point**2 * eigenvalues, axis=1)
        pool_hits[(pool_hits < 0) & (values >= threshold)] = t
        nhit = int(np.count_nonzero(pool_hits >= 0))
        pool_rows.append({"T": t, "M": t + 1, "hits": nhit, "q_hat": nhit / count,
                          "ci95": confidence_interval(nhit, count),
                          "analytic_upper": pool_probability(direct_upper, t + 1)})
    np.savez_compressed(output / f"frozen_{domain}_d{d}_n{rank}.npz", H=H,
                        eigenvalues=eigenvalues, eigenvectors=vectors,
                        starts_in_eigencoordinates=starts, pool_hit_time=pool_hits,
                        **{f"{name}_hit_time": values for name, values in raw_hits.items()})
    # Verify the efficient implementation against the unchanged general code
    # on the same actual H, including the orthogonal coordinate transform.
    validation_errors = {}
    for name, omega in modes:
        reference, _ = search(H, starts[:16] @ vectors.T, steps=min(steps, 7),
                              domain=domain, alpha=alpha, omega=omega or 0.5, ball_step_rule=ball_step_rule)
        diagonal, _ = search(Hdiag, starts[:16], steps=min(steps, 7),
                             domain=domain, alpha=alpha, omega=omega or 0.5, ball_step_rule=ball_step_rule)
        validation_errors[name] = float(max(np.max(np.abs(reference - diagonal @ vectors.T)),
                                            np.max(np.abs(reference - early_iterates[name] @ vectors.T))))
        assert validation_errors[name] < 1e-10
    return {"domain": domain, "dimension": d, "rank": rank, "seed": seed,
            "starts": count, "gamma": gamma, "lambda1": float(eigenvalues[-1]),
            "lambda2_ratio": float(eigenvalues[-2] / eigenvalues[-1]),
            "negative_eigenvalue_relative": float(min(eigenvalues[0], 0) / eigenvalues[-1]),
            "methods": methods, "iid_pool": pool_rows,
            "direct_analytic_upper": direct_upper,
            "max_feasibility_defect": max_defect,
            "minimum_increment_over_lambda1": minimum_increment,
            "coordinate_update_validation": validation_errors}


def make_plots(output, old, low_runs, low_widths, grid_info, probability):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 190})
    navy, gold = "#17365D", "#B18B3F"
    styles = {"oracle": ("#383D43", "-", "Exact indicator"),
              "gd": (navy, "-", "Continuous search (100 updates)"),
              "pool256": ("#D6BA7D", "-", "Random pool 256"),
              "pool4096": (gold, "-", "Random pool 4096"),
              "pool10000": ("#806226", "-", "Random pool 10000"),
              "grid1": ("#6B7F91", ":", "Tensor grid h=1"),
              "grid0.5": ("#6B7F91", "--", "Tensor grid h=1/2"),
              "grid0.25": ("#6B7F91", "-.", "Tensor grid h=1/4")}
    def save(fig, name):
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"{name}.{suffix}", bbox_inches="tight")
        plt.close(fig)
    def draw_runs(ax, rows, widths, methods):
        for method in methods:
            selected = [r for r in rows if r["method"] == method]
            if not selected:
                continue
            curves = np.array([r["errors"] for r in selected]) / widths[:len(selected[0]["errors"])][None, :]
            x = np.arange(curves.shape[1])
            color, ls, label = styles[method]
            ax.plot(x, np.median(curves, axis=0), color=color, ls=ls, lw=1.7, label=label)
            if len(curves) > 1:
                ax.fill_between(x, curves.min(0), curves.max(0), color=color, alpha=0.08, linewidth=0)
        ax.set_yscale("log")
        ax.grid(alpha=0.16)
        ax.set_xlabel("Reduced dimension N")
    mapped = []
    mapping = {"exact_indicator": "oracle", "exp_rgd100": "gd", "relaxed_pg100": "gd",
               "fixed_pool256": "pool256", "fixed_pool4096": "pool4096", "fixed_pool10000": "pool10000"}
    for row in old["runs"]:
        if row["method"] in mapping:
            mapped.append({**row, "method": mapping[row["method"]]})
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.4), sharey=True)
    for row, domain in enumerate(("sphere", "ball")):
        for col, d in enumerate((8, 16, 32, 64)):
            ax = axes[row, col]
            selected = [r for r in mapped if r["domain"] == domain and r["dimension"] == d]
            draw_runs(ax, selected, np.array(old["problems"][str(d)]["hf_widths"]),
                      ["oracle", "gd", "pool256", "pool4096", "pool10000"])
            ax.set_title(f"{'Sphere' if domain == 'sphere' else 'Ball'}, d={d}")
            if col == 0:
                ax.set_ylabel(r"$E_N/d_N$")
    fig.suptitle("Diffusion FEM: greedy error relative to the exact HF width\nEight seeds; median and range; original 2026-10-12 data", fontsize=13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    save(fig, "greedy_ratio_combined")

    for orientation in ("rotated", "original"):
        fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.8), sharey=True)
        for row, domain in enumerate(("sphere", "ball")):
            for col, d in enumerate((3, 4, 6)):
                selected = [r for r in low_runs if r["domain"] == domain and r["dimension"] == d and r["orientation"] == orientation]
                methods = ["oracle", "gd", "pool256", "pool4096", "pool10000", "grid1", "grid0.5", "grid0.25"] if orientation == "rotated" else ["oracle", "grid1", "grid0.5", "grid0.25"]
                ax = axes[row, col]
                draw_runs(ax, selected, np.array(low_widths[str(d)]), methods)
                counts = [g["actual_candidate_count"] for g in grid_info if g["domain"] == domain and g["dimension"] == d]
                ax.set_title(f"{domain.capitalize()}, d={d}\nGrid counts: {', '.join(f'{c:,}' for c in counts)}", fontsize=10)
                if col == 0:
                    ax.set_ylabel(r"$E_N/d_N$")
        suffix = "Fixed orthogonal parameter rotation, applied to all selectors" if orientation == "rotated" else "Original coordinates: axis alignment is retained as a control"
        fig.suptitle("Deterministic tensor grids, random pools, and continuous search\n" + suffix, fontsize=12)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5)
        fig.tight_layout(rect=(0, 0.10, 1, 0.91))
        save(fig, f"grid_comparison_{orientation}")

    for rank_kind in ("zero", "quarter", "half"):
        fig, axes = plt.subplots(2, 5, figsize=(17.5, 7.2), sharex=True, sharey=True)
        for row, domain in enumerate(("sphere", "ball")):
            for col, d in enumerate((8, 16, 32, 64, 128)):
                rank = 0 if rank_kind == "zero" else d // (4 if rank_kind == "quarter" else 2)
                record = next(r for r in probability if r["domain"] == domain and r["dimension"] == d and r["rank"] == rank)
                ax = axes[row, col]
                method = "exp_rgd" if domain == "sphere" else "relaxed_pg"
                gd = record["methods"][method]["snapshots"]
                pool = record["iid_pool"]
                x = [r["T"] for r in gd]
                ax.plot(x, [r["q_hat"] for r in gd], color=navy, lw=1.8, label="GD: observed")
                ax.fill_between(x, [r["ci95"][0] for r in gd], [r["ci95"][1] for r in gd], color=navy, alpha=0.12)
                ax.plot(x, [r["analytic_lower"] for r in gd], color=navy, ls="--", lw=1.5, label="GD: analytic lower")
                ax.plot(x, [r["q_hat"] for r in pool], color=gold, lw=1.7, label="iid pool: observed")
                ax.fill_between(x, [r["ci95"][0] for r in pool], [r["ci95"][1] for r in pool], color=gold, alpha=0.14)
                ax.plot(x, [r["analytic_upper"] for r in pool], color=gold, ls=":", lw=1.8, label="iid pool: analytic upper")
                ax.set_title(f"{domain.capitalize()}, d={d}, n={rank}", fontsize=10)
                ax.set_ylim(-0.025, 1.025)
                ax.grid(alpha=0.16)
                if row == 1:
                    ax.set_xlabel("GD updates T; pool M=T+1")
                if col == 0:
                    ax.set_ylabel("Threshold success probability")
        fig.suptitle(r"Frozen-space success at $\gamma=0.9$: independent trajectories and iid pools" + "\n4096 replicates; pointwise 95% intervals; equal indicator counts do not imply equal runtime", fontsize=12)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10)
        fig.tight_layout(rect=(0, 0.07, 1, 0.91))
        save(fig, f"success_curves_n_{rank_kind}")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    dimension_rows = []
    for ax, domain in zip(axes, ("sphere", "ball")):
        records = sorted((r for r in probability if r["domain"] == domain and
                          r["rank"] == r["dimension"] // 4), key=lambda r: r["dimension"])
        dimensions = [r["dimension"] for r in records]
        method = "exp_rgd" if domain == "sphere" else "relaxed_pg"
        observed = [r["methods"][method]["snapshots"][-1]["q_hat"] for r in records]
        lower = [r["methods"][method]["snapshots"][-1]["analytic_lower"] for r in records]
        ax.semilogy(dimensions, observed, "o-", color=navy, ms=4, label="GD T=100: observed")
        ax.semilogy(dimensions, lower, "--", color=navy, lw=1.8, label="GD T=100: analytic lower")
        for M, color in ((256, "#D6BA7D"), (4096, gold), (10000, "#806226")):
            upper = [pool_probability(r["direct_analytic_upper"], M) for r in records]
            ax.semilogy(dimensions, upper, "o:", color=color, ms=3, label=f"iid pool M={M}: upper")
            dimension_rows.extend({"domain": domain, "dimension": r["dimension"],
                                   "rank": r["rank"], "M": M, "pool_analytic_upper": p}
                                  for r, p in zip(records, upper))
        ax.set_title(domain.capitalize())
        ax.set_xlabel("Parameter dimension d")
        ax.set_xticks(dimensions)
        ax.set_ylim(1e-20, 3)
        ax.grid(alpha=0.16)
    axes[0].set_ylabel("Success probability (log scale)")
    fig.suptitle("Frozen reference n=d/4: dimension dependence\nFixed pool sizes shown separately; these are not equal-budget comparisons", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.15, 1, 0.88))
    save(fig, "success_vs_dimension_n_quarter")
    write_json(output / "pool_size_dimension_bounds.json", dimension_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/geometry_extended_cpu_20261015"))
    parser.add_argument("--audit-starts", type=int, default=4096)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path("runs/geometry_cpu_20261012")
    old = json.loads((source / "results.json").read_text())
    config = {"source_data": str(source.resolve()), "greedy_dimensions_reused": [8, 16, 32, 64],
              "grid_dimensions": [3, 4, 6], "grid_h": [1.0, 0.5, 0.25],
              "probability_dimensions": [8, 16, 32, 64, 128], "ranks": ["0", "d/4", "d/2"],
              "seeds": list(range(1000, 1008)), "updates": 100, "sphere_alpha": 0.2,
              "ball_omega": 0.5, "ball_hard_companion_omega": 1.0,
              "pool_sizes": [256, 4096, 10000], "audit_starts": args.audit_starts,
              "gamma": 0.9, "rotation_seed": 20261015,
              "dtype": "float64", "backend": "local CPU NumPy/SciPy", "threads": 1,
              "pool_probability_budget": "M=T+1; indicator evaluation count only, not runtime equivalence",
              "grid_rank_limit": "d-1", "fem_grid": 40, "fem_order": 3,
              "fem_rho": 0.78, "fem_contrast": 0.25}
    write_json(output / "config.json", config)
    begin_all = time.perf_counter()
    with threadpool_limits(limits=1):
        import skfem
        env = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
               "executable": sys.executable, "numpy": np.__version__, "scipy": scipy.__version__,
               "scikit_fem": skfem.__version__, "platform": platform.platform(),
               "threadpools": threadpool_info(),
               "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), Path(__file__).with_name("geometry_cpu.py"),
                                            Path(__file__).with_name("run_geometry_cpu.py"),
                                            source / "results.json")}}
        write_json(output / "environment.json", env)
        matrices = {d: dict(np.load(source / f"fem_d{d}.npz")) for d in (8, 16, 32, 64)}
        print("Building same diffusion P3 FEM at d=128", flush=True)
        begin = time.perf_counter()
        fem = FEMProblem.build(40, 128, contrast=0.25, spectrum_ratio=0.78, element_order=3)
        H128 = sym(fem.B.T @ fem.solve_X(fem.B))
        matrices[128] = {"G0": fem.solution_gram, "H0": H128, "hf_widths": fem.singular_values}
        np.savez_compressed(output / "fem_d128.npz", **matrices[128], source_transform=fem.source_transform, modes=fem.modes)
        write_json(output / "fem_d128_metadata.json", {"dofs": fem.dofs, "projection_error": fem.projection_error,
                   "quadrature_order": fem.quadrature_order, "construction_and_residual_seconds": time.perf_counter() - begin})
        print(f"FEM ready: {fem.dofs} DoFs; {time.perf_counter()-begin:.1f}s", flush=True)
        del fem

        low_runs, low_widths, grids = [], {}, []
        for d in config["grid_dimensions"]:
            G0 = matrices[8]["G0"][:d, :d]
            H0 = matrices[8]["H0"][:d, :d]
            low_widths[str(d)] = np.sqrt(np.maximum(la.eigvalsh(G0)[::-1], 0)).tolist()
            O, R = la.qr(np.random.default_rng(config["rotation_seed"] + d).normal(size=(d, d)))
            O *= np.sign(np.diag(R))[None, :]
            np.savez_compressed(output / f"low_d{d}.npz", G0=G0, H0=H0, rotation=O,
                                rotated_G0=O.T @ G0 @ O, rotated_H0=O.T @ H0 @ O)
            for domain in ("sphere", "ball"):
                for orientation in ("rotated", "original"):
                    G, H = (O.T @ G0 @ O, O.T @ H0 @ O) if orientation == "rotated" else (G0, H0)
                    methods = ["oracle", "gd", "pool256", "pool4096", "pool10000"] if orientation == "rotated" else ["oracle"]
                    for method in methods:
                        seeds = config["seeds"] if method != "oracle" else [config["seeds"][0]]
                        for seed in seeds:
                            record = selector_run(G, H, domain, method, seed, 100, d - 1)
                            low_runs.append({**record, "orientation": orientation})
                    for h in config["grid_h"]:
                        points, metadata = tensor_candidates(d, h, domain)
                        if orientation == "rotated":
                            grids.append(metadata)
                        record = selector_run(G, H, domain, f"grid{h:g}", 0, 100, d - 1, points)
                        low_runs.append({**record, "orientation": orientation,
                                         "grid_h": h, "candidate_count": len(points)})
                        print(f"Grid {orientation} {domain} d={d}, h={h}: {len(points):,} candidates", flush=True)
                write_json(output / "low_dimension_results.json", {"widths": low_widths, "grid_counts": grids, "runs": low_runs})
        probability = []
        for d in config["probability_dimensions"]:
            for rank in (0, d // 4, d // 2):
                H, Q = frozen_space(matrices[d]["G0"], matrices[d]["H0"], rank)
                np.savez_compressed(output / f"reference_d{d}_n{rank}.npz", H=H, Q=Q)
                for domain in ("sphere", "ball"):
                    record = batch_probability(H, domain, args.audit_starts, 100,
                                                20261015 + 100 * d + rank, 0.9, output, rank)
                    probability.append(record)
                    method = "exp_rgd" if domain == "sphere" else "relaxed_pg"
                    last = record["methods"][method]["snapshots"][-1]
                    pool = record["iid_pool"][-1]
                    print(f"Probability {domain} d={d} n={rank}: q100={last['q_hat']:.4f}, "
                          f"lower={last['analytic_lower']:.4f}, pool101={pool['q_hat']:.4g}, upper={pool['analytic_upper']:.4g}", flush=True)
                    write_json(output / "probability_curves.json", probability)
        validation = {"max_orthogonality_defect": max(r["a_orthogonality_defect"] for r in low_runs),
                      "max_feasibility_defect": max(r["max_feasibility_defect"] for r in probability),
                      "max_coordinate_update_difference": max(v for r in probability for v in r["coordinate_update_validation"].values()),
                      "min_increment_over_lambda1": min(r["minimum_increment_over_lambda1"] for r in probability),
                      "minimum_relative_eigenvalue": min(r["negative_eigenvalue_relative"] for r in probability),
                      "all_blas_one_thread": all(p["num_threads"] == 1 for p in threadpool_info()),
                      "gpu_backends_loaded": [s for s in ("jax", "torch", "cupy") if s in sys.modules],
                      "low_dimension_runs": len(low_runs), "frozen_domain_objectives": len(probability)}
        if validation["max_coordinate_update_difference"] > 1e-8 or validation["max_feasibility_defect"] > 1e-12 or validation["max_orthogonality_defect"] > 1e-8 or not validation["all_blas_one_thread"] or validation["gpu_backends_loaded"]:
            raise RuntimeError(f"Validation failed: {validation}")
        write_json(output / "validation.json", validation)
        make_plots(output, old, low_runs, low_widths, grids, probability)
        env["finished_utc"] = datetime.now(timezone.utc).isoformat()
        env["elapsed_seconds"] = time.perf_counter() - begin_all
        write_json(output / "environment.json", env)
        (output / "README.txt").write_text(
            "Extended sphere/ball CPU study\n\n"
            "The high-dimensional greedy curves reuse raw 2026-10-12 results; new low-dimensional curves use exact prefixes of the same assembled FEM matrices.\n"
            "Low-dimensional main comparisons apply one saved orthogonal parameter rotation to every selector. The sphere/ball solution manifold and widths are unchanged. Original-coordinate grid controls are also saved and plotted.\n"
            "Ball grids are actual Cartesian tensor grids restricted to the closed ball. Sphere grids are the radial images of nonzero Cartesian tensor-grid points, with duplicate rays removed; they are not uniform in surface area. Actual retained counts and pre-filter tensor counts are both saved. These grids have exponential dimension cost; only d=3,4,6 is attempted.\n"
            "Probability curves hold V_n fixed on exact-indicator reference spaces, at n=0, d/4, d/2, and use 4096 independent trajectories plus 4096 independent nested iid pools. Their M=T+1 budget matches indicator evaluation counts only, not arithmetic cost or runtime. Pool and GD share each replicate's first candidate.\n"
            "The sphere analytic lower bound is the existing normalized-gradient exponential-map formula. The ball lower bound is the existing relaxed-projection formula, with omega=0.5 (plus omega=1 raw companion). iid pool upper=1-(1-p_upper)^M using the spectral one-draw bound. Intervals are pointwise exact binomial 95% intervals, not simultaneous confidence bands.\n"
            "Probability updates are performed in saved eigencoordinates of the actual FEM H_n, equivalent under orthogonal invariance. They are checked against the unchanged original update implementation on H_n. No diagonal surrogate PDE is introduced.\n")
        print(f"Complete: {output}; {env['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
