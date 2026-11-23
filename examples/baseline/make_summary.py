#!/usr/bin/env python3
"""Create cross-case summary tables and figures from completed experiments."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((ROOT / name / "results.json").read_text(encoding="utf-8"))


def final_ratio(data: dict, method: str, denominator: float) -> float:
    return data["aggregate"][method]["stats"][-1]["worst_median"] / denominator


def min_rho_q10(data: dict, method: str) -> float:
    rows = [x for x in data["aggregate"][method]["stats"] if x["rho_q10"] is not None]
    return min(x["rho_q10"] for x in rows)


def q_power_from_eigs(data: dict, steps: int, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    out = []
    for row in data["audit_run"]["rows"][:-1]:
        eigs = np.maximum(np.asarray(row["indicator_eigenvalues"]), 0.0)
        eigs /= eigs[-1]
        z = rng.standard_normal((samples, eigs.size))
        weights = z * z
        weights /= weights.sum(axis=1, keepdims=True)
        weights *= eigs[None, :] ** (2 * steps)
        values = (weights @ eigs) / weights.sum(axis=1)
        out.append(float(np.mean(values >= 0.9**2)))
    return out


def main() -> None:
    outdir = ROOT / "summary"
    outdir.mkdir(exist_ok=True)
    dims = [8, 16, 32, 64]
    dim_data = {d: load(f"results_d{d}_separated") for d in dims}
    cases = {
        "separated r=0.78": load("results_d32_separated"),
        "clustered r=0.97": load("results_d32_clustered"),
        "flat r=0.995": load("results_d32_flat"),
        "raw sine loads": load("results_d32_raw"),
    }

    dimension_table = []
    for d in dims:
        data = dim_data[d]
        rank = data["config"]["rank"]
        width = data["problem"]["singular_values"][rank]
        audit = data["audit_run"]["rows"][:-1]
        q10 = q_power_from_eigs(data, 10, 30000, 8800 + d)
        dimension_table.append({
            "d": d,
            "rank": rank,
            "width": width,
            "power_T10_over_width": final_ratio(data, "power_T10", width),
            "adam_T100_over_width": final_ratio(data, "adam_T100", width),
            "pool_M256_over_width": final_ratio(data, "pool_M256", width),
            "pool_M4096_over_width": final_ratio(data, "pool_M4096", width),
            "power_T10_min_rho_q10": min_rho_q10(data, "power_T10"),
            "pool_M4096_min_rho_q10": min_rho_q10(data, "pool_M4096"),
            "p_upper_median": float(np.median([x["p_high_rigorous_upper"] for x in audit])),
            "pool_M95_lower_median": float(np.median([x["pool_M95_rigorous_lower"] for x in audit])),
            "power_T10_success_min": min(q10),
        })

    case_table = []
    for name, data in cases.items():
        exact = data["aggregate"]["exact_indicator"]["stats"][-1]["worst_median"]
        audit = data["audit_run"]["rows"][:-1]
        q5 = q_power_from_eigs(data, 5, 30000, 9100)
        q10 = q_power_from_eigs(data, 10, 30000, 9200)
        case_table.append({
            "case": name,
            "exact_final_error": exact,
            "power_T2_over_exact": final_ratio(data, "power_T2", exact),
            "power_T10_over_exact": final_ratio(data, "power_T10", exact),
            "power_T100_over_exact": final_ratio(data, "power_T100", exact),
            "adam_T100_over_exact": final_ratio(data, "adam_T100", exact),
            "pool_M4096_over_exact": final_ratio(data, "pool_M4096", exact),
            "eigengap_median": float(np.median([x["relative_gap"] for x in audit])),
            "exact_PL_min_over_rounds": float(min(x["local_pl_min_normalized"] for x in audit)),
            "p_MC_median": float(np.median([x["p_high"] for x in audit])),
            "p_MC_max": float(max(x["p_high"] for x in audit)),
            "q_T5_min": min(q5),
            "q_T10_min": min(q10),
        })

    lr = json.loads((ROOT / "adam_lr_sweep.json").read_text(encoding="utf-8"))
    summary = {
        "threshold_gamma": 0.9,
        "dimension_table": dimension_table,
        "case_table": case_table,
        "adam_lr_sweep": lr,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axs = plt.subplots(2, 2, figsize=(11.8, 8.2))
    data = cases["separated r=0.78"]
    ax = axs[0, 0]
    for key, label in [
        ("exact_indicator", "exact indicator"), ("power_T10", "continuous power T=10"),
        ("adam_T100", "Adam T=100"), ("pool_M256", "fixed pool M=256"),
        ("pool_M4096", "fixed pool M=4096")]:
        s = data["aggregate"][key]["stats"]
        ax.semilogy([x["n"] for x in s], [x["worst_median"] for x in s], marker="o", ms=3, label=label)
    sv = np.asarray(data["problem"]["singular_values"])
    n = np.arange(data["config"]["rank"] + 1)
    ax.semilogy(n, sv[: len(n)], "k--", label="Kolmogorov width")
    ax.set(xlabel="RB dimension n", ylabel="worst energy error", title="Full greedy convergence, d=32")
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=7)

    ax = axs[0, 1]
    for key, label in [("power_T10_over_width", "continuous T=10"), ("adam_T100_over_width", "Adam T=100"),
                       ("pool_M256_over_width", "pool M=256"), ("pool_M4096_over_width", "pool M=4096")]:
        ax.plot(dims, [x[key] for x in dimension_table], marker="o", label=label)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set(xlabel="parameter dimension d", ylabel="final error / n-width", title="Dimension scaling at n=d/2")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=7)

    ax = axs[1, 0]
    ax.semilogy(dims, [x["p_upper_median"] for x in dimension_table], "o-", label="upper bound on p")
    ax.plot(dims, [x["power_T10_success_min"] for x in dimension_table], "s-", label="continuous q, T=10")
    ax.set(xlabel="parameter dimension d", ylabel="probability", title="High-value mass p vs attraction success q")
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=7)

    ax = axs[1, 1]
    names = list(cases)
    x = np.arange(len(names)); width = 0.2
    for j, (key, label) in enumerate([("power_T2_over_exact", "continuous T=2"),
                                      ("power_T10_over_exact", "continuous T=10"),
                                      ("pool_M4096_over_exact", "pool M=4096")]):
        ax.bar(x + (j - 1) * width, [r[key] for r in case_table], width, label=label)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xticks(x, ["separated", "clustered", "flat", "raw"], rotation=15)
    ax.set(ylabel="final error / exact-indicator greedy", title="Dependence on landscape geometry")
    ax.legend(fontsize=7); ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "regularity_convergence_summary.png", dpi=200)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(6.6, 4.2))
    lrs = [x["lr"] for x in lr["results"]]
    err = [x["final_error_median"] / lr["optimal_width_rank16"] for x in lr["results"]]
    succ = [x["success_all_rounds_rho_ge_0.9"] for x in lr["results"]]
    ax1.semilogx(lrs, err, "o-", color="tab:blue", label="final error / n-width")
    ax1.set(xlabel="Adam learning rate", ylabel="final error / n-width")
    ax1.grid(True, which="both", alpha=0.25)
    ax2 = ax1.twinx(); ax2.semilogx(lrs, succ, "s--", color="tab:orange", label="all-round success")
    ax2.set_ylabel("fraction with rho_n >= 0.9 for all n")
    lines = ax1.lines + ax2.lines
    ax1.legend(lines, [x.get_label() for x in lines], fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(outdir / "adam_lr_robustness.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
