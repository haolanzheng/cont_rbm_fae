"""Plot saved spectral runs; does not train a model on import."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def make_figures(data: dict, outdir: Path) -> None:
    agg = data["aggregate"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    preferred = [
        "exact_indicator",
        "power_T5",
        "power_T25",
        "adam_T100",
        "pool_M256",
        "pool_M1024",
        "pool_M4096",
    ]
    for key in preferred:
        if key not in agg:
            continue
        s = agg[key]["stats"]
        n = np.array([x["n"] for x in s])
        med = np.array([x["worst_median"] for x in s])
        q10 = np.array([x["worst_q10"] for x in s])
        q90 = np.array([x["worst_q90"] for x in s])
        ax.semilogy(n, med, marker="o", ms=3, label=key)
        if agg[key]["count"] > 1:
            ax.fill_between(n, q10, q90, alpha=0.12)
    sv = np.asarray(data["problem"]["singular_values"])
    n = np.arange(min(len(sv), data["config"]["rank"]) + 1)
    width = np.r_[sv[: len(n) - 1], sv[len(n) - 1] if len(n) - 1 < len(sv) else 0.0]
    if len(width) == len(n):
        ax.semilogy(n, width, "k--", lw=1.5, label="Kolmogorov width")
    ax.set_xlabel("RB dimension n")
    ax.set_ylabel("exact worst energy error (audit only)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "outer_convergence.png", dpi=180)
    plt.close(fig)

    audit = data["audit_run"]["rows"][:-1]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.7))
    nn = np.array([r["n"] for r in audit])
    axs[0].semilogy(
        nn,
        [max(r["p_high"], 0.5 / data["config"]["audit_samples"]) for r in audit],
        "o-",
        label="p: MC high-value set",
    )
    axs[0].semilogy(
        nn,
        [r["p_high_rigorous_upper"] for r in audit],
        "^-",
        label="rigorous upper bound for p",
    )
    axs[0].plot(
        nn, [r["q_empirical"] for r in audit], "s-", label="q: attraction success"
    )
    axs[0].set_xlabel("greedy step n")
    axs[0].set_ylabel("measure / success probability")
    axs[0].legend(fontsize=8)
    axs[0].grid(True, which="both", alpha=0.25)
    axs[1].plot(nn, [r["relative_gap"] for r in audit], "o-", label="relative eigengap")
    axs[1].plot(
        nn,
        [
            r["local_pl_min_normalized"]
            if r["local_pl_min_normalized"] is not None
            else np.nan
            for r in audit
        ],
        "s-",
        label="normalized local PL",
    )
    axs[1].set_xlabel("greedy step n")
    axs[1].legend(fontsize=8)
    axs[1].grid(True, alpha=0.25)
    axs[2].plot(nn, [r["c_minus"] for r in audit], "o-", label="c_-")
    axs[2].plot(nn, [r["c_plus"] for r in audit], "s-", label="c_+")
    axs[2].set_xlabel("greedy step n")
    axs[2].set_ylabel("indicator/error constants")
    axs[2].legend(fontsize=8)
    axs[2].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "assumption_audit.png", dpi=180)
    plt.close(fig)
