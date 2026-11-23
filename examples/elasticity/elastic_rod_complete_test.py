"""Complete pool-versus-continuous greedy test for the random elastic rod.

The physical model and discretization follow Sec. 4.1 of Dematteis, Grafke,
and Vanden-Eijnden (2018): 30 independent elastic blocks, 30 moving nodes,
T=1, dt=1e-3, zero initial data, and r(t)=a*t**1.5.  The amplitude a only
scales this linear problem, so it does not affect relative ROM errors.

Selection never evaluates the FOM.  It maximizes the ROM dynamic residual.
Only the selected point is solved with the FOM for POD enrichment.  Continuous
selection uses one start and exactly 100 Adam updates per greedy round.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from contgreedy.sampling.host import T_INNER, adam_ascent
from examples.elasticity.random_rod_greedy import RandomElasticRod, initial_basis


class PaperRod(RandomElasticRod):
    def __init__(self, amplitude=5.0):
        super().__init__(n=30, final_time=1.0, dt=1e-3, beta=1.5,
                         action_radius=np.sqrt(30.0))
        self.amplitude = float(amplitude)

    def force(self, t):
        return self.amplitude * max(t, 0.0) ** self.beta

    def prior_samples(self, rng, count):
        return rng.normal(size=(count, self.d))

    def sobol_shell(self, count, seed):
        # Deterministic low-discrepancy Gaussian directions, projected to the
        # same action contour as random-pool and continuous searches.
        u = qmc.Sobol(self.d, scramble=True, seed=seed).random(count)
        z = ndtri(np.clip(u, 1e-12, 1.0 - 1e-12))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return self.radius * z


def parameter_summary(problem, theta):
    d = problem.elasticity(np.asarray(theta))
    return {
        "theta_norm": float(np.linalg.norm(theta)),
        "D_left_mean": float(d[:15].mean()),
        "D_right_mean": float(d[15:].mean()),
        "D_min": float(d.min()),
        "D_argmin_1based": int(np.argmin(d) + 1),
    }


def train(problem, name, rounds, seed, learning_rate, pool_size=100):
    rng = np.random.default_rng(seed)
    v = initial_basis(problem)
    bases = [v.copy()]
    selected = []
    fixed = None
    if name == "fixed_random_100":
        fixed = problem.shell_samples(rng, pool_size)
    elif name == "fixed_sobol_100":
        fixed = problem.sobol_shell(pool_size, seed)
    elif name == "fixed_random_10000":
        fixed = problem.shell_samples(rng, pool_size)

    started = time.perf_counter()
    for iteration in range(rounds):
        search_started = time.perf_counter()
        if name == "continuous_adam100":
            theta0 = problem.shell_samples(rng, 1)[0]
            fg = lambda z: problem.sphere_indicator_grad(z, v)
            theta, value = adam_ascent(fg, theta0, problem.project,
                                       steps=T_INNER, lr=learning_rate)
            start_value = float(problem.indicator_batch(theta0, v)[0])
        else:
            candidates = (problem.shell_samples(rng, pool_size)
                          if name == "resampled_random_100" else fixed)
            values = problem.indicator_batch(candidates, v)
            index = int(np.argmax(values))
            theta, value = candidates[index].copy(), float(values[index])
            start_value = None
        search_seconds = time.perf_counter() - search_started

        full = problem.full_trajectory(theta)  # only after selection
        selected_error_before = problem.true_error(theta, v, full)
        v = problem.enrich(v, full, 1)
        bases.append(v.copy())
        selected.append({
            "round": iteration + 1,
            "indicator_squared": value,
            "start_indicator_squared": start_value,
            "true_error_before_enrichment": selected_error_before,
            "search_seconds": search_seconds,
            "theta": theta.tolist(),
            **parameter_summary(problem, theta),
        })
        print(name, "round", iteration + 1, "rank", v.shape[1],
              "indicator", f"{value:.6g}", "error", f"{selected_error_before:.5f}",
              "search_s", f"{search_seconds:.2f}", flush=True)
    return {
        "basis": v,
        "bases": bases,
        "selected": selected,
        "training_seconds": time.perf_counter() - started,
    }


def stiffness_action(problem, dval, u):
    """Batched tridiagonal K(theta)u in O(batch*n)."""
    du = np.empty_like(u)
    du[:, 0] = u[:, 0]
    du[:, 1:] = u[:, 1:] - u[:, :-1]
    flux = dval * du / problem.h**2
    ku = np.empty_like(u)
    ku[:, :-1] = flux[:, :-1] - flux[:, 1:]
    ku[:, -1] = flux[:, -1]
    return ku


def error_batch(problem, theta, bases):
    """Relative trajectory errors for several bases, sharing each FOM solve."""
    theta = np.asarray(theta)
    bsz = theta.shape[0]
    dval = problem.elasticity(theta)
    um = np.zeros((bsz, problem.n))
    u = np.zeros_like(um)
    unext = np.broadcast_to(
        0.5 * problem.dt**2 * problem.force(0.0) * problem.load[None, :],
        (bsz, problem.n)).copy()

    states = []
    for v in bases:
        # Discrete gradients BV, one per spring.
        bv = np.empty_like(v)
        bv[0] = v[0]
        bv[1:] = v[1:] - v[:-1]
        kr = np.einsum("bj,jr,js->brs", dval, bv, bv,
                       optimize=True) / problem.h**2
        fr = v.T @ problem.load
        qm = np.zeros((bsz, v.shape[1]))
        q = np.zeros_like(qm)
        qnext = np.broadcast_to(
            0.5 * problem.dt**2 * problem.force(0.0) * fr[None, :],
            (bsz, v.shape[1])).copy()
        states.append([v, kr, fr, qm, q, qnext, np.zeros(bsz)])

    denominator = np.zeros(bsz)
    for it in range(1, problem.nt + 1):
        um, u = u, unext
        denominator += np.einsum("bn,bn->b", u, u)
        for state in states:
            v, kr, fr, qm, q, qnext, numerator = state
            qm[:], q[:] = q, qnext
            diff = u - q @ v.T
            numerator += np.einsum("bn,bn->b", diff, diff)
            if it < problem.nt:
                accel = problem.force(problem.times[it]) * fr[None, :] - np.einsum(
                    "bij,bj->bi", kr, q, optimize=True)
                qnext[:] = 2.0 * q - qm + problem.dt**2 * accel
        if it < problem.nt:
            accel = (problem.force(problem.times[it]) * problem.load[None, :]
                     - stiffness_action(problem, dval, u))
            unext = 2.0 * u - um + problem.dt**2 * accel
    return [np.sqrt(s[-1] / np.maximum(denominator, 1e-30)) for s in states]


def errors_for_points(problem, points, bases, batch_size=128):
    chunks = [[] for _ in bases]
    for begin in range(0, len(points), batch_size):
        out = error_batch(problem, points[begin:begin + batch_size], bases)
        for target, values in zip(chunks, out):
            target.append(values)
    return [np.concatenate(x) for x in chunks]


def lr_check(problem, seed, rates):
    """Indicator-only calibration at the initial basis and one fixed start."""
    v = initial_basis(problem)
    theta0 = problem.shell_samples(np.random.default_rng(seed), 1)[0]
    out = {}
    for rate in rates:
        started = time.perf_counter()
        fg = lambda z: problem.sphere_indicator_grad(z, v)
        z, value = adam_ascent(fg, theta0, problem.project,
                               steps=T_INNER, lr=rate)
        out[str(rate)] = {
            "indicator_squared": value,
            "seconds": time.perf_counter() - started,
            **parameter_summary(problem, z),
        }
        print("lr", rate, "indicator", value, flush=True)
    return out


def audit(problem, trained, gaussian_count, shell_count, seed):
    rng = np.random.default_rng(seed)
    gaussian = problem.prior_samples(rng, gaussian_count)
    shell = problem.shell_samples(rng, shell_count)

    # The stress set is shared: every method is tested on every point chosen by
    # every method, plus one independent Adam-100 attack per final basis.
    selected = np.asarray([row["theta"] for run in trained.values()
                           for row in run["selected"]])
    attacks = []
    for run in trained.values():
        z0 = problem.shell_samples(rng, 1)[0]
        fg = lambda z, v=run["basis"]: problem.sphere_indicator_grad(z, v)
        z, _ = adam_ascent(fg, z0, problem.project,
                           steps=T_INNER, lr=0.12)
        attacks.append(z)
    attacks = np.asarray(attacks)
    points = np.vstack((gaussian, shell, selected, attacks))

    names = list(trained)
    bases = [trained[name]["basis"] for name in names]
    errors = errors_for_points(problem, points, bases)
    result = {}
    g_end = gaussian_count
    s_end = g_end + shell_count
    sel_end = s_end + len(selected)
    for name, v, err in zip(names, bases, errors):
        # Correlation uses a fixed audit subset and is diagnostic only.
        subset = points[:min(500, len(points))]
        indicator = np.sqrt(problem.indicator_batch(subset, v))
        result[name] = {
            "rank": int(v.shape[1]),
            "gaussian_mean": float(err[:g_end].mean()),
            "gaussian_median": float(np.median(err[:g_end])),
            "gaussian_q95": float(np.quantile(err[:g_end], 0.95)),
            "gaussian_q99": float(np.quantile(err[:g_end], 0.99)),
            "shell_max": float(err[g_end:s_end].max()),
            "all_selected_max": float(err[s_end:sel_end].max()),
            "shared_attack_max": float(err[sel_end:].max()),
            "stress_max": float(err[g_end:].max()),
            "indicator_error_corr_500_gaussian": float(np.corrcoef(
                indicator, err[:len(subset)])[0, 1]),
            "training_seconds": trained[name]["training_seconds"],
        }
    return result, {"gaussian": gaussian_count, "shell": shell_count,
                    "selected": len(selected), "attacks": len(attacks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--learning-rate", type=float, default=0.12)
    ap.add_argument("--gaussian-audit", type=int, default=2000)
    ap.add_argument("--shell-audit", type=int, default=2000)
    ap.add_argument("--skip-lr-check", action="store_true")
    ap.add_argument("--output", type=Path,
                    default=Path("elastic_rod_complete_result.json"))
    args = ap.parse_args()

    problem = PaperRod()
    # Recheck the optimized gradient implementation.
    v0 = initial_basis(problem)
    rng = np.random.default_rng(123)
    z = problem.shell_samples(rng, 1)[0]
    direction = rng.normal(size=problem.d); direction /= np.linalg.norm(direction)
    val, grad = problem.indicator_grad(z, v0)
    eps = 1e-6
    fd = (problem.indicator_batch(z + eps * direction, v0)[0]
          - problem.indicator_batch(z - eps * direction, v0)[0]) / (2 * eps)
    gradient_check = {"value": val, "analytic_directional": float(grad @ direction),
                      "finite_difference": float(fd),
                      "relative_difference": float(abs(grad @ direction - fd)
                                                   / max(1.0, abs(fd)))}
    if gradient_check["relative_difference"] > 2e-5:
        raise RuntimeError(gradient_check)

    calibration = ({} if args.skip_lr_check else
                   lr_check(problem, args.seed, [0.06, 0.12, 0.24]))
    methods = [
        ("fixed_random_100", 100),
        ("resampled_random_100", 100),
        ("fixed_sobol_100", 100),
        ("fixed_random_10000", 10000),
        ("continuous_adam100", 0),
    ]
    trained = {}
    for name, size in methods:
        trained[name] = train(problem, name, args.rounds, args.seed,
                              args.learning_rate, size)

    audit_result, audit_counts = audit(problem, trained, args.gaussian_audit,
                                       args.shell_audit, seed=20260916)
    serial_runs = {}
    for name, run in trained.items():
        serial_runs[name] = {
            "training_seconds": run["training_seconds"],
            "selected": run["selected"],
        }
    result = {
        "model": {
            "reference": "Dematteis--Grafke--Vanden-Eijnden (2018), Sec. 4.1, Case 2",
            "dimension_M": 30, "spatial_unknowns_N": 30,
            "final_time": 1.0, "dt": 1e-3, "time_integrator": "position form of velocity-Verlet",
            "elasticity": "D_j=theta_j/2+sqrt(theta_j^2/4+1)",
            "prior": "theta_j iid N(0,1)",
            "forcing": "r(t)=5*t^1.5 (amplitude cancels in relative errors)",
            "initial_data": "u(0)=u_t(0)=0",
            "search_constraint": "||theta||_2=sqrt(30)",
        },
        "greedy": {
            "initial_rank": 2, "enrichments": args.rounds,
            "final_rank": 2 + args.rounds,
            "indicator": "time-integrated squared full dynamic residual of ROM",
            "continuous_starts_per_round": 1,
            "continuous_adam_steps_per_round": T_INNER,
            "learning_rate": args.learning_rate,
            "fom_during_selection": False,
        },
        "gradient_check": gradient_check,
        "learning_rate_calibration": calibration,
        "audit_counts": audit_counts,
        "audit": audit_result,
        "runs": serial_runs,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print("FINAL_AUDIT")
    print(json.dumps(audit_result, indent=2), flush=True)


if __name__ == "__main__":
    main()
