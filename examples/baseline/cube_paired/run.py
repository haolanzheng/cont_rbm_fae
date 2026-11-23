"""CPU reduced-basis tests for the paired-mode cube benchmark.

Run from the repository root:
    PYTHONPATH=. .venv_cpu/bin/python -m examples.baseline.cube_paired.run

The PDE is -u''=d**(-1/2)*sum(z_j*(-psi_j'')), with zero Dirichlet
conditions and energy-orthonormal sine modes psi_j. All algebra is evaluated
in these exact energy coordinates; there is no high-fidelity discretization.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.stats import binom
from threadpoolctl import threadpool_info, threadpool_limits

from examples.baseline.cube.test_set import make_vertex_test_set
from .model import PairedCubeModel, paired_prefix, paired_search, projected_search, verify_model


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "runs" / "cube_paired_rb_20261104"


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def wilson(hits, total):
    z = 1.959963984540054
    p = hits / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return max(0., center-half), min(1., center+half)


def probability_bounds(d, alpha, T, gamma):
    m = d//2
    B = (1 + alpha/(3*np.sqrt(m)))**T
    p = (1-1/B)**2
    k = int(np.ceil(gamma*gamma*m))
    q = float(binom.sf(k-1, m, p))
    single_vertex = float(binom.sf(k-1, m, .5))
    M = T+1
    return {
        "gd_lower": q,
        "pool_upper_81": float(min(1., M*np.exp(-2*m*(gamma*gamma-1/6)**2))) if gamma*gamma>1/6 else 1.,
        "vertex_pool_exact_81": float(-np.expm1(M*np.log1p(-single_vertex))),
        "pool_size": M,
        "pair_amplification_lower": B,
        "pair_saturation_probability_lower": p,
    }


def frozen_test(d, args, out):
    rng = np.random.default_rng(2026110400+d)
    starts = rng.uniform(-1., 1., size=(args.starts, d))
    begin = time.perf_counter()
    best, stats = paired_search(starts, alpha=args.alpha, T=args.T,
                               checkpoints=tuple(sorted(set([0, 10, 20, 40, args.T]))),
                               threshold=args.gamma**2)
    elapsed = time.perf_counter()-begin
    # Independently evaluate the original PDE energy-coordinate residual.
    model = PairedCubeModel(d)
    values = model.error_squared(paired_prefix(d), best)
    initial_values = model.error_squared(paired_prefix(d), starts)
    history = stats["history"]
    if not np.allclose(values, np.asarray(stats["final_values"]), rtol=2e-12, atol=2e-12):
        raise AssertionError("Batched paired search and original energy residual disagree")
    hits = int(np.count_nonzero(values >= args.gamma**2))
    lo, hi = wilson(hits, args.starts)
    bounds = probability_bounds(d, args.alpha, args.T, args.gamma)
    row = dict(d=d, starts=args.starts, T=args.T, alpha=args.alpha, gamma=args.gamma,
               gd_hits=hits, gd_q=hits/args.starts, gd_ci_low=lo, gd_ci_high=hi,
               initial_hits=int(np.count_nonzero(initial_values >= args.gamma**2)),
               mean_final_J=float(values.mean()), min_final_J=float(values.min()),
               mean_initial_J=float(initial_values.mean()), wall_seconds=elapsed,
               min_increment=float(stats["min_increment"]),
               feasibility_defect=float(stats["feasibility_defect"]), **bounds)
    row["history"] = [dict(T=int(h["T"]), hits=int(np.count_nonzero(np.asarray(h["best_values"])>=args.gamma**2)),
                          mean_J=float(np.mean(h["best_values"]))) for h in history]
    np.savez_compressed(out/f"frozen_d{d}.npz", initial_parameters=starts,
                        returned_parameters=best, initial_J=initial_values, final_J=values)
    print(f"frozen d={d}: GD={row['gd_q']:.6f}, bound={row['gd_lower']:.6f}, pool upper={row['pool_upper_81']:.3g}", flush=True)
    return row


def train_one(d, seed, mode, method, test_coefficients, initial_test_error2, args, out):
    model = PairedCubeModel(d)
    Q = paired_prefix(d) if mode == "paired_prefix" else np.empty((d, 0))
    initial_rank = Q.shape[1]
    target = 3*d//4 if mode == "paired_prefix" else d//2
    errors2 = initial_test_error2.copy()
    rows, choices = [], []
    elapsed = 0.
    # Pool prefixes are nested for a given seed. GD draws use the same seed,
    # with one independent initialization at each enrichment.
    rng = np.random.default_rng(seed+1000003)
    begin = time.perf_counter()
    if method.startswith("pool"):
        M = int(method[4:])
        pool = rng.uniform(-1., 1., size=(M, d))
        pool_coefficients = pool/np.sqrt(d)
        pool_values = np.sum(pool_coefficients*pool_coefficients, axis=1)
        if initial_rank:
            pool_values -= np.sum((pool_coefficients@Q)**2, axis=1)
        pool_values = np.maximum(pool_values, 0.)
    elapsed += time.perf_counter()-begin
    stopped = "target rank reached"

    def record():
        N = Q.shape[1]
        err = float(np.sqrt(max(float(errors2.max()), 0.)))
        width = model.width(N)
        rows.append(dict(d=d, seed=seed, mode=mode, method=method, N=N,
                         test_worst_error=err, width=width,
                         error_ratio=err/width if width else None, rank=N))

    record()
    while Q.shape[1] < target:
        begin = time.perf_counter()
        n = Q.shape[1]
        if method == "gd":
            initial = rng.uniform(-1., 1., size=d)
            H = model.matrix(Q)
            parameter, search_stats = projected_search(H, initial, alpha=args.alpha, T=args.T,
                                                       checkpoints=(), threshold=None)
            parameter = np.asarray(parameter).reshape(d)
            value = float(parameter@H@parameter)
            grad_min_increment = float(search_stats["min_increment"])
        else:
            index = int(np.argmax(pool_values))
            parameter = pool[index].copy()
            value = float(pool_values[index])
            grad_min_increment = None
        Q_new, novelty = model.append(Q, parameter)
        if Q_new.shape[1] == Q.shape[1]:
            elapsed += time.perf_counter()-begin
            stopped = "dependent selected snapshot"
            break
        Q = Q_new
        q = Q[:, -1]
        if method.startswith("pool"):
            pool_values = np.maximum(pool_values-(pool_coefficients@q)**2, 0.)
        elapsed += time.perf_counter()-begin
        choices.append(dict(n=n, parameter=parameter, selected_J=value, novelty=novelty,
                            min_increment=grad_min_increment))
        # Evaluation is excluded from training time and uses identical points.
        errors2 = np.maximum(errors2-(test_coefficients@q)**2, 0.)
        record()

    defect = float(np.linalg.norm(Q.T@Q-np.eye(Q.shape[1]), ord=2))
    if defect > 2e-10:
        raise AssertionError(f"Loss of basis orthogonality: {defect}")
    # One direct evaluation checks accumulated test errors without rerunning
    # the complete expensive test matrix multiplication.
    sample = np.linspace(0, len(test_coefficients)-1, min(257,len(test_coefficients)), dtype=int)
    expected = np.sum(test_coefficients[sample]**2, axis=1)-np.sum((test_coefficients[sample]@Q)**2, axis=1)
    mismatch = float(np.max(np.abs(expected-errors2[sample])))
    if mismatch > 3e-11:
        raise AssertionError(f"Incremental error mismatch: {mismatch}")
    np.savez_compressed(out/f"basis_{mode}_d{d}_{method}_s{seed}.npz", Q=Q,
                        selected_parameters=np.asarray([c["parameter"] for c in choices]),
                        selected_J=np.asarray([c["selected_J"] for c in choices]))
    timing = dict(d=d, seed=seed, mode=mode, method=method, train_seconds=elapsed,
                  final_N=Q.shape[1], stopped_reason=stopped,
                  orthogonality_defect=defect, incremental_error_discrepancy=mismatch)
    return rows, timing, [dict(d=d, seed=seed, mode=mode, method=method, **c) for c in choices]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dimensions", type=int, nargs="+", default=[8,16,32,64,128])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1000,1008)))
    p.add_argument("--starts", type=int, default=16384)
    p.add_argument("--alpha", type=float, default=.6)
    p.add_argument("--T", type=int, default=80)
    p.add_argument("--gamma", type=float, default=.9)
    p.add_argument("--test-count", type=int, default=65536)
    args = p.parse_args()
    if args.T != 80:
        raise ValueError("This initial report schema uses the 81-point comparison; use T=80.")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        validation = verify_model()
        save_json(out/"validation.json", validation)
        settings = dict(dimensions=args.dimensions, seeds=args.seeds, starts=args.starts,
                        alpha=args.alpha, T=args.T, gamma=args.gamma,
                        pool_sizes=[256,4096,10000], modes=["empty_start","paired_prefix"],
                        targets={"empty_start":"N=d/2", "paired_prefix":"N=3d/4, starting at N=d/2"},
                        device="CPU", dtype="float64", platform=platform.platform(),
                        threadpools=threadpool_info(),
                        test_set="Same saved vertex set for every method and seed within each d; exhaustive for d<=16. For larger d, union of 65536 uniform cube vertices and up to 65536 vertices with opposite signs within each pair. The latter cover the known maximizers at the paired prefix; the test set is fixed before any training.",
                        test_count_limit=args.test_count,
                        test_error="maximum absolute energy error on the saved common vertex test set",
                        exact_width="sqrt((d-N)/d)",
                        PDE="-u'' = d^(-1/2) sum z_j (-psi_j''), u(0)=u(1)=0; psi_j=sqrt(2)/(j*pi) sin(j*pi*x)",
                        discretization="exact analytic energy-coordinate algebra; no FEM or HF solve",
                        timing="candidate generation, search and enrichment; excludes fixed-test evaluation and saved output",
                        probability="independent uniform starts in one fixed paired-prefix space; separate from complete RB histories",
                        scope="paired prefix is a prescribed valid exact-greedy tie sequence, not assumed for random training")
        save_json(out/"settings.json", settings)
        frozen, curves, timings, choices, test_metadata = [], [], [], [], []
        for d in args.dimensions:
            frozen.append(frozen_test(d,args,out))
            save_json(out/"frozen.json", frozen)
            Z, exhaustive = make_vertex_test_set(d, args.test_count, seed=202611040)
            general_count = len(Z)
            if not exhaustive:
                pair_signs, _ = make_vertex_test_set(d//2, args.test_count, seed=202611041)
                pair_vertices = np.empty((len(pair_signs), d))
                pair_vertices[:,0::2] = pair_signs
                pair_vertices[:,1::2] = -pair_signs
                Z = np.unique(np.vstack([Z, pair_vertices]), axis=0)
            S = Z/np.sqrt(d)
            np.savez_compressed(out/f"test_vertices_d{d}.npz", parameters=Z)
            test_metadata.append(dict(d=d, count=len(Z), exhaustive=exhaustive,
                                      uniform_cube_vertex_count=general_count,
                                      additional_paired_vertex_count=len(Z)-general_count))
            save_json(out/"test_sets.json", test_metadata)
            del Z
            paired_errors = np.maximum(1.-np.sum((S@paired_prefix(d))**2, axis=1),0.)
            for mode in ("empty_start", "paired_prefix"):
                initial_errors = paired_errors if mode=="paired_prefix" else np.ones(len(S))
                for seed in args.seeds:
                    for method in ("gd","pool256","pool4096","pool10000"):
                        r,t,c = train_one(d,seed,mode,method,S,initial_errors,args,out)
                        curves.extend(r);timings.append(t);choices.extend(c)
                        print(f"RB d={d} {mode} seed={seed} {method}: N={t['final_N']}, E={r[-1]['test_worst_error']:.6f}, E/width={r[-1]['error_ratio']:.6f}, seconds={t['train_seconds']:.4f}",flush=True)
                    save_json(out/"rb_curves.json", curves)
                    save_json(out/"timings.json", timings)
                    save_json(out/"selected_parameters.json", choices)
            del S
        print("Completed paired-mode RB tests.",flush=True)


if __name__ == "__main__":
    main()
