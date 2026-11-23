"""Exact and Monte Carlo diagnostics for the spectral benchmark."""

import math
import numpy as np
import scipy.linalg as la
from scipy.stats import beta as beta_distribution
from .sampling import normalize_rows


def estimate_high_set_probability(
    eigs: np.ndarray, gamma: float, samples: int, rng: np.random.Generator
) -> tuple[float, int]:
    hit = 0
    batch = 20000
    threshold = gamma * gamma * eigs[-1]
    for start in range(0, samples, batch):
        m = min(batch, samples - start)
        z = normalize_rows(rng.standard_normal((m, eigs.size)))
        vals = (z * z) @ eigs
        hit += int(np.count_nonzero(vals >= threshold))
    return hit / samples, hit


def landscape_audit(
    H: np.ndarray, gamma: float, starts: int, steps: int, rng: np.random.Generator
) -> dict:
    eigs = la.eigvalsh(H)
    jstar = max(float(eigs[-1]), 1e-300)
    ratios = []
    pl_values = []
    threshold_pl_values = []
    for _ in range(starts):
        z = rng.standard_normal(H.shape[0])
        z /= la.norm(z)
        best_value = float(z @ H @ z)
        for _ in range(steps):
            hz = H @ z
            val = float(z @ hz)
            best_value = max(best_value, val)
            gap = jstar - val
            threshold_gap = gamma * gamma * jstar - val
            if threshold_gap > 1e-13 * jstar:
                grad = 2.0 * (hz - val * z)
                threshold_pl_values.append(0.5 * float(grad @ grad) / threshold_gap)
            if val >= 0.8 * jstar and gap > 1e-13 * jstar:
                grad = 2.0 * (hz - val * z)
                pl_values.append(0.5 * float(grad @ grad) / gap)
            nh = la.norm(hz)
            if nh < 1e-15:
                break
            z = hz / nh
        best_value = max(best_value, float(z @ H @ z))
        ratios.append(best_value / jstar)
    ratios_arr = np.asarray(ratios)
    lam_ratio = float(eigs[-2] / jstar) if eigs.size > 1 else 0.0
    # Rotationally invariant rigorous bounds based only on lambda_1 and lambda_2.
    # z_1^2 ~ Beta(1/2,(d-1)/2).  z_1^2 >= gamma^2 is sufficient, while
    # z_1^2 >= (gamma^2-lambda_2/lambda_1)/(1-lambda_2/lambda_1) is necessary.
    sufficient = gamma * gamma
    necessary = max(0.0, (gamma * gamma - lam_ratio) / max(1.0 - lam_ratio, 1e-15))
    p_lower = float(beta_distribution.sf(sufficient, 0.5, 0.5 * (H.shape[0] - 1)))
    p_upper = float(beta_distribution.sf(necessary, 0.5, 0.5 * (H.shape[0] - 1)))
    m95_lower_bound = (
        float(math.log(0.05) / math.log1p(-p_upper))
        if 0.0 < p_upper < 1.0
        else (1.0 if p_upper >= 1.0 else math.inf)
    )
    pl_min = float(np.min(pl_values)) if pl_values else None
    pl_median = float(np.median(pl_values)) if pl_values else None
    threshold_pl_min = (
        float(np.min(threshold_pl_values)) if threshold_pl_values else None
    )
    threshold_pl_median = (
        float(np.median(threshold_pl_values)) if threshold_pl_values else None
    )
    return {
        "lambda1": float(eigs[-1]),
        "lambda2": float(eigs[-2]) if eigs.size > 1 else 0.0,
        "relative_gap": float((eigs[-1] - eigs[-2]) / jstar) if eigs.size > 1 else 1.0,
        "q_empirical": float(np.mean(ratios_arr >= gamma * gamma)),
        "rho_median": float(np.sqrt(np.median(ratios_arr))),
        "rho_min": float(np.sqrt(max(np.min(ratios_arr), 0.0))),
        "local_pl_min": pl_min,
        "local_pl_median": pl_median,
        "local_pl_min_normalized": pl_min / jstar if pl_min is not None else None,
        "local_pl_median_normalized": pl_median / jstar
        if pl_median is not None
        else None,
        "threshold_pl_min": threshold_pl_min,
        "threshold_pl_median": threshold_pl_median,
        "threshold_pl_min_normalized": threshold_pl_min / jstar
        if threshold_pl_min is not None
        else None,
        "threshold_pl_median_normalized": threshold_pl_median / jstar
        if threshold_pl_median is not None
        else None,
        "p_high_rigorous_lower": p_lower,
        "p_high_rigorous_upper": p_upper,
        "pool_M95_rigorous_lower": m95_lower_bound,
        "indicator_eigenvalues": eigs.tolist(),
    }


def aggregate(runs: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for run in runs:
        if run["method"] in {"power", "adam"}:
            key = f"{run['method']}_T{run['steps']}"
        elif "pool" in run["method"]:
            key = f"{run['method']}_M{run['pool_size']}"
        else:
            key = run["method"]
        groups.setdefault(key, []).append(run)
    out = {}
    for key, rr in groups.items():
        max_len = max(len(r["rows"]) for r in rr)
        stats = []
        for k in range(max_len):
            vals = [r["rows"][k]["worst_true"] for r in rr if k < len(r["rows"])]
            rhos = [
                r["rows"][k].get("rho")
                for r in rr
                if k < len(r["rows"]) and r["rows"][k].get("rho") is not None
            ]
            stats.append(
                {
                    "n": k,
                    "worst_median": float(np.median(vals)),
                    "worst_q10": float(np.quantile(vals, 0.1)),
                    "worst_q90": float(np.quantile(vals, 0.9)),
                    "rho_median": float(np.median(rhos)) if rhos else None,
                    "rho_q10": float(np.quantile(rhos, 0.1)) if rhos else None,
                }
            )
        out[key] = {
            "count": len(rr),
            "elapsed_median": float(np.median([r["elapsed_seconds"] for r in rr])),
            "stats": stats,
        }
    return out
