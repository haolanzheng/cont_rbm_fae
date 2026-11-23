#!/usr/bin/env python3
"""Create final figures and a self-contained Markdown report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.nonsmooth.nonsmooth_continuous_greedy import ContactModel, ObstacleModel, build_common_basis  # noqa: E402


ROOT = Path("nonsmooth_pde_bench/results_full")


def fixed_success_figure(base: dict, retuned: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), sharey=True)
    steps = [1, 5, 10, 25, 50, 100]
    for ax, name in zip(axes, ["membrane_obstacle", "elastic_contact"]):
        f0 = base["models"][name]["fixed_space"]
        f = retuned["models"][name]["fixed_space"]
        for q, color in [("0.99", "#2b6cb0"), ("0.995", "#d97706"), ("0.999", "#c53030")]:
            ax.plot(steps, [f["q_by_step"][str(T)][q] for T in steps], "o-", color=color, label=f"Adam, τ=q{q[2:]}")
            ax.axhline(f0["pool100_success_formula"][q], color=color, linestyle="--", linewidth=1.3, alpha=0.75)
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("Adam steps T")
        ax.set_ylim(0, 0.82)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("success probability")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Fixed-space rare-set success (dashed: iid pool M=100)")
    fig.tight_layout()
    fig.savefig(ROOT / "fixed_space_success_retuned.png", dpi=210)
    plt.close(fig)


def greedy_figure(retuned: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    rng = np.random.default_rng(17)
    for ax, name in zip(axes, ["membrane_obstacle", "elastic_contact"]):
        g = retuned["models"][name]["full_greedy"]
        pool = np.array([r["max_error"] for r in g["pool_summary"]["runs"]])
        adam = np.array([r["max_error"] for r in g["adam_runs"]])
        bp = ax.boxplot([pool, adam], positions=[1, 2], widths=0.52, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], ["#90cdf4", "#f6ad55"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.scatter(1 + rng.normal(0, 0.045, len(pool)), pool, s=21, color="#2b6cb0", alpha=0.8, zorder=3)
        ax.scatter(2 + rng.normal(0, 0.045, len(adam)), adam, s=21, color="#c05621", alpha=0.8, zorder=3)
        ax.set_xticks([1, 2], ["pool M=100", "Adam T=100"])
        ax.set_title(name.replace("_", " "))
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("rank-12 maximum relative K-energy error")
    fig.suptitle("Full greedy, 24 seeds and one common 2,048-point deployment set")
    fig.tight_layout()
    fig.savefig(ROOT / "full_greedy_24seeds_retuned.png", dpi=210)
    plt.close(fig)


def lr_figure(lrdata: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for ax, name in zip(axes, ["membrane_obstacle", "elastic_contact"]):
        rows = sorted((float(k), v["mean"]) for k, v in lrdata[name].items())
        x, y = zip(*rows)
        ax.plot(x, y, "o-", color="#4a5568")
        best = int(np.argmax(y))
        ax.scatter([x[best]], [y[best]], s=75, color="#c53030", zorder=4, label=f"chosen lr={x[best]:g}")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("Adam learning rate")
        ax.set_ylabel("mean best log-indicator")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.suptitle("Learning-rate selection on 40 disjoint tuning starts")
    fig.tight_layout()
    fig.savefig(ROOT / "learning_rate_audit.png", dpi=210)
    plt.close(fig)


def nonsmooth_slice_figure() -> dict:
    specifications = [
        (ObstacleModel(24), 20260919, 2, np.array([0.25, 0.0, 0.0]), "obstacle height coordinate"),
        (ContactModel(28, 8), 20261919, 0, np.array([0.0, 0.0, 0.0]), "load coordinate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7.5), sharex="col")
    summary = {}
    saved = {}
    for col, (model, seed, dim, basez, xlabel) in enumerate(specifications):
        V, _ = build_common_basis(model, 16, 7, seed + 1)
        ts = np.linspace(-0.98, 0.98, 241)
        J, grad, rom_active, fom_active = [], [], [], []
        for t in ts:
            z = basez.copy()
            z[dim] = t
            val, g, meta = model.indicator_and_grad(V, z)
            _, info = model.solve_full(z)
            J.append(val)
            grad.append(g[dim])
            rom_active.append(meta["active_count"])
            fom_active.append(info.active_count)
        J, grad = np.asarray(J), np.asarray(grad)
        rom_active, fom_active = np.asarray(rom_active), np.asarray(fom_active)
        axes[0, col].plot(ts, J, color="#2b6cb0")
        axes[0, col].set_title(model.name.replace("_", " "))
        axes[0, col].set_ylabel("log residual indicator")
        axes[0, col].grid(alpha=0.2)
        axes[1, col].step(ts, fom_active, where="mid", label="FOM active nodes", color="#2f855a")
        axes[1, col].step(ts, rom_active, where="mid", label="ROM active nodes", color="#c05621", alpha=0.85)
        axes[1, col].set_xlabel(f"normalized {xlabel}")
        axes[1, col].set_ylabel("active-set size")
        axes[1, col].grid(alpha=0.2)
        axes[1, col].legend(fontsize=8)
        summary[model.name] = {
            "fom_active_min": int(fom_active.min()), "fom_active_max": int(fom_active.max()),
            "fom_active_count_changes": int(np.sum(np.diff(fom_active) != 0)),
            "rom_active_min": int(rom_active.min()), "rom_active_max": int(rom_active.max()),
            "rom_active_count_changes": int(np.sum(np.diff(rom_active) != 0)),
            "largest_adjacent_gradient_change": float(np.max(np.abs(np.diff(grad)))),
        }
        saved[model.name + "_t"] = ts
        saved[model.name + "_J"] = J
        saved[model.name + "_grad"] = grad
        saved[model.name + "_rom_active"] = rom_active
        saved[model.name + "_fom_active"] = fom_active
    fig.suptitle("One-dimensional slices expose active-set switching")
    fig.tight_layout()
    fig.savefig(ROOT / "nonsmooth_parameter_slices.png", dpi=210)
    plt.close(fig)
    np.savez(ROOT / "nonsmooth_slice_data.npz", **saved)
    (ROOT / "nonsmooth_slice_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def report(base: dict, retuned: dict, slices: dict) -> None:
    o0 = base["models"]["membrane_obstacle"]["fixed_space"]
    c0 = base["models"]["elastic_contact"]["fixed_space"]
    o = retuned["models"]["membrane_obstacle"]
    c = retuned["models"]["elastic_contact"]
    lines = [
        "# Continuous greedy on two nonsmooth PDE benchmarks",
        "",
        "## Bottom line",
        "",
        "The contact problem is a positive example; the obstacle problem is a useful control rather than a second positive example. In both models, gradient trajectories reach the rarest indicator tail much more often than a 100-point iid pool. Only in elastic contact does that search effect survive 24 full greedy replications as a statistically supported reduction of the rank-12 maximum deployment error.",
        "",
        "## Problems and protocol",
        "",
        "The models use semismooth penalty/Nitsche-type discretizations (not an exact hard-constraint variational inequality):",
        "",
        "- **Membrane obstacle:** `K u - f(mu) + gamma (u-psi(mu))_+ = 0` on a 24x24 interior grid (576 unknowns). The three parameters are source amplitude, horizontal source location, and obstacle height.",
        "- **Frictionless elastic contact:** `K u - f(mu) + gamma C^T(Cu+g(mu))_- = 0` for a plane-strain Q1 cantilever with a rigid foundation (28x8 elements, 504 free unknowns, 28 contact nodes). The parameters are downward load, load location, and gap.",
        "",
        "The positive/negative-part laws make the parameter-to-solution and residual objectives piecewise smooth. This is the same active-set structure that motivates semismooth formulations of obstacle and contact problems. Related foundations include [Christof--Wachsmuth on semismooth obstacle solution operators](https://arxiv.org/abs/2112.12018), [Haasdonk--Salomon--Wohlmuth on RB methods for variational inequalities](https://epubs.siam.org/doi/10.1137/110835372), and [Niakh et al. on Nitsche RB contact mechanics](https://arxiv.org/abs/2307.11541).",
        "",
        "For a fixed K-orthonormal RB space `V_N`, the reduced state solves `V_N^T F(V_N c;mu)=0`, and selection maximizes",
        "",
        "`J_N(mu) = log( ||F(V_N c;mu)||_{K^{-1}} / ||f(mu)||_{K^{-1}} )`.",
        "",
        "No truth or projection error is used for selection or learning-rate tuning. Adam uses one start, exactly T=100 steps, exact active-region sensitivities, and `z=tanh(x)` rather than clipping. The learning rate is selected by mean best-J on 40 disjoint tuning starts: 0.55 (obstacle) and 0.8 (contact). Thresholds come from 32,768 Sobol reference points; q uses 320 independent starts. Full greedy uses rank 12, 24 independent seeds, and one common 2,048-point deployment set. Pool and Adam both spend 100 parameter evaluations per outer step, although an Adam evaluation also requires derivatives.",
        "",
        "The finite-difference gradient audit has median relative error "
        f"{o0['gradient_audit']['median_relative_error']:.2e} (obstacle) and {c0['gradient_audit']['median_relative_error']:.2e} (contact). All 16 truth snapshots used to build the common rank-7 fixed space converged.",
        "",
        "## 1. Fixed-space p-q experiment",
        "",
        "Here `p=P(J>=tau)` and the exact iid pool success is `1-(1-p)^100`. The empirical p differs slightly from the nominal quantile because of finite reference sampling.",
        "",
        "| model | tau quantile | p | pool-100 success | Adam q(T=100) | Adam 95% Wilson interval |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, block, f0 in [("obstacle", o, o0), ("contact", c, c0)]:
        f = block["fixed_space"]
        for q in ["0.99", "0.995", "0.999"]:
            ci = f["q_T100_wilson95"][q]
            lines.append(f"| {name} | {q} | {f0['p'][q]:.5f} | {f0['pool100_success_formula'][q]:.3f} | {f['q_T100'][q]:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    lines += [
        "",
        "![Fixed-space success](fixed_space_success_retuned.png)",
        "",
        "The mechanism is not uniform stochastic dominance. For contact, Adam is worse at the 0.99 threshold (0.494 versus pool 0.634), roughly tied at 0.995 (0.400 versus 0.395), but much better at 0.999 (0.400 versus 0.096). Successful trajectories fall into the basin of a very high local maximum; unsuccessful ones remain below the broader moderate tail.",
        "",
        "Most of the rare-tail gain appears quickly. At the 0.999 threshold, obstacle q is 0.134, 0.278, 0.325, and 0.334 at T=5,10,25,100; contact q is 0.331, 0.388, 0.400, and 0.400. Thus contact obtains essentially all of its T=100 tail success within 10--25 steps.",
        "",
        "## 2. Full greedy at equal basis cardinality",
        "",
        "| model / metric | pool M=100 | Adam T=100 | comparison |",
        "|---|---:|---:|---:|",
    ]
    for label, block in [("obstacle", o), ("contact", c)]:
        g = block["full_greedy"]
        ps, aa, comp = g["pool_summary"], g["adam_summary"], g["comparison"]
        lines += [
            f"| {label}: max error median | {ps['max_error_median']:.3f} | {aa['max_error_median']:.3f} | pool/Adam={comp['median_ratio_pool_over_adam']:.3f} |",
            f"| {label}: q99 error median | {ps['q99_error_median']:.3f} | {aa['q99_error_median']:.3f} | lower is better |",
            f"| {label}: mean error median | {ps['mean_error_median']:.3f} | {aa['mean_error_median']:.3f} | lower is better |",
        ]
    lines += [
        "",
        "![Full greedy](full_greedy_24seeds_retuned.png)",
        "",
        "- **Obstacle:** the max-error median changes 0.525 to 0.488, but the bootstrap 95% interval for pool/Adam is [0.937, 1.194] and Mann--Whitney p=0.370. Its q99 and mean errors are slightly worse under Adam. This is not a convincing full-greedy advantage.",
        "- **Contact:** max-error median changes 0.752 to 0.529 (factor 1.420); the bootstrap 95% ratio interval is [1.019, 1.636], Mann--Whitney p=0.0046, and the all-pairs probability that Adam beats pool is 0.740. The q99 median improves 0.719 to 0.485 and mean error 0.282 to 0.208. The fraction of runs with max error below 0.6 rises from 3/24 to 14/24.",
        "",
        "## 3. Is nonsmoothness genuinely present?",
        "",
        "The fixed-space snapshot activity ranges are 41--155 obstacle nodes and 7--14 contact nodes. Along representative one-dimensional parameter slices, the full obstacle active-set size changes "
        f"{slices['membrane_obstacle']['fom_active_count_changes']} times and the full contact set changes {slices['elastic_contact']['fom_active_count_changes']} times. The corresponding indicator curves remain continuous but acquire derivative changes at active-set switches.",
        "",
        "![Nonsmooth slices](nonsmooth_parameter_slices.png)",
        "",
        "![Learning-rate audit](learning_rate_audit.png)",
        "",
        "## What this test establishes—and what it does not",
        "",
        "1. **Established experimentally:** a continuous one-start trajectory can have much higher probability than a 100-point iid pool of reaching an extremely small high-indicator set, even when the objective is only piecewise smooth. In contact, this rare-tail advantage translates into a meaningful rank-12 RB improvement.",
        "2. **Not established:** nonsmoothness by itself is beneficial. The obstacle control has the same qualitative active-set switching but no statistically convincing full-greedy gain.",
        "3. **The sharper condition suggested by the data:** continuous search helps when (i) the useful states sit in a very thin extreme-indicator tail, (ii) a basin of one-sided gradients leads to that tail, and (iii) snapshots from that tail add contact modes that a size-100 pool frequently misses. Merely having kinks or many active-set switches is insufficient.",
        "4. **Current limitation:** contact Adam remains bimodal; its max-error IQR is [0.429, 0.747]. The continuous method improves the probability of a good greedy construction, but does not make every single-start run reliable. Learning-rate tuning is material and must remain part of the protocol.",
        "",
        "## Reproducibility files",
        "",
        "- `nonsmooth_continuous_greedy.py`: models, solvers, sensitivities, p-q test, and full greedy.",
        "- `replicate_full_greedy.py`: 24-seed replication and bootstrap comparison.",
        "- `retuned_evaluation.py`: independent evaluation after the expanded learning-rate audit.",
        "- `results.json`, `greedy_24seeds.json`, and `retuned_results.json`: raw results.",
    ]
    (ROOT / "REPORT_FINAL.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    base = json.loads((ROOT / "results.json").read_text())
    retuned = json.loads((ROOT / "retuned_results.json").read_text())
    lrdata = json.loads((ROOT / "expanded_lr_tuning.json").read_text())
    fixed_success_figure(base, retuned)
    greedy_figure(retuned)
    lr_figure(lrdata)
    slices = nonsmooth_slice_figure()
    report(base, retuned, slices)


if __name__ == "__main__":
    main()
