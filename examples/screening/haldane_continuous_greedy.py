"""Continuous versus random-pool greedy for a finite Haldane spin-1 chain.

The Hamiltonian and parameter cut follow Brehmer et al. (PRE 108, 025306):

    H(E) = sum_i S_i.S_{i+1} + E sum_i [(S_i^x)^2-(S_i^y)^2],
    E in [0, 2], D=0, open boundaries, J=1.

We work in the conserved even-magnetization-parity sector and use exact sparse
diagonalization.  Search uses only the reduced Ritz residual and its analytic
parameter derivative; truth solves are performed only at selected snapshots.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from contgreedy.sampling.host import BestObserved
from scipy import optimize, sparse
from scipy.sparse.linalg import eigsh


def orth_append(V: np.ndarray, x: np.ndarray, tol: float = 1e-11):
    q = np.asarray(x, float).copy()
    for _ in range(2):
        if V.size:
            q -= V @ (V.T @ q)
    nq = np.linalg.norm(q)
    if nq <= tol:
        return V, False
    return np.column_stack((V, q / nq)), True


@dataclass
class SearchInfo:
    objective_calls: int = 0
    iterations: int = 0
    seconds: float = 0.0


class HaldaneChain:
    def __init__(self, L: int = 8):
        self.L = int(L)
        self.domain = (0.0, 2.0)
        self._truth_cache: dict[float, tuple[float, np.ndarray, float]] = {}
        self.H0, self.H1, self.full_codes = self._assemble()
        self.n = self.H0.shape[0]
        rng = np.random.default_rng(91827 + L)
        self._eig_v0 = rng.normal(size=self.n)
        self._eig_v0 /= np.linalg.norm(self._eig_v0)

    def _assemble(self):
        L = self.L
        powers = 3 ** np.arange(L, dtype=np.int64)
        nfull = int(3**L)
        codes = np.arange(nfull, dtype=np.int64)
        digits_all = (codes[:, None] // powers[None, :]) % 3
        m_all = digits_all - 1
        keep = (np.sum(m_all, axis=1) % 2) == 0
        full_codes = codes[keep]
        digits = digits_all[keep]
        m = m_all[keep]
        n = len(full_codes)
        lookup = np.full(nfull, -1, dtype=np.int64)
        lookup[full_codes] = np.arange(n, dtype=np.int64)

        rows0: list[int] = []
        cols0: list[int] = []
        vals0: list[float] = []
        rows1: list[int] = []
        cols1: list[int] = []
        vals1: list[float] = []

        for a, code in enumerate(full_codes):
            ma = m[a]
            diag = float(np.dot(ma[:-1], ma[1:]))
            rows0.append(a); cols0.append(a); vals0.append(diag)

            # 0.5(S+S- + S-S+) has unit matrix elements for spin one.
            for i in range(L - 1):
                di, dj = int(digits[a, i]), int(digits[a, i + 1])
                if di < 2 and dj > 0:
                    new_code = int(code + powers[i] - powers[i + 1])
                    rows0.append(a); cols0.append(int(lookup[new_code])); vals0.append(1.0)
                if di > 0 and dj < 2:
                    new_code = int(code - powers[i] + powers[i + 1])
                    rows0.append(a); cols0.append(int(lookup[new_code])); vals0.append(1.0)

            # (Sx)^2-(Sy)^2 = 0.5[(S+)^2+(S-)^2], coefficient one for spin one.
            for i in range(L):
                di = int(digits[a, i])
                if di == 0:
                    new_code = int(code + 2 * powers[i])
                    rows1.append(a); cols1.append(int(lookup[new_code])); vals1.append(1.0)
                elif di == 2:
                    new_code = int(code - 2 * powers[i])
                    rows1.append(a); cols1.append(int(lookup[new_code])); vals1.append(1.0)

        H0 = sparse.coo_matrix((vals0, (rows0, cols0)), shape=(n, n)).tocsr()
        H1 = sparse.coo_matrix((vals1, (rows1, cols1)), shape=(n, n)).tocsr()
        H0.sum_duplicates(); H1.sum_duplicates()
        return H0, H1, full_codes

    def truth(self, E: float, need_gap: bool = False):
        key = round(float(E), 13)
        if key not in self._truth_cache:
            H = self.H0 + key * self.H1
            vals, vecs = eigsh(H, k=2, which="SA", v0=self._eig_v0,
                               tol=2e-11, maxiter=20000)
            order = np.argsort(vals)
            vals = vals[order]; vecs = vecs[:, order]
            u = vecs[:, 0]
            # Deterministic gauge, relevant only for reproducible saved states.
            pivot = int(np.argmax(np.abs(u)))
            if u[pivot] < 0:
                u = -u
            residual = np.linalg.norm(H @ u - vals[0] * u)
            self._truth_cache[key] = (float(vals[0]), u, float(vals[1] - vals[0]))
            if residual > 2e-8:
                raise RuntimeError(f"truth residual {residual:.3e} at E={E}")
        lam, u, gap = self._truth_cache[key]
        return (lam, u, gap) if need_gap else u

    def reduced_data(self, V: np.ndarray):
        H0V = self.H0 @ V
        H1V = self.H1 @ V
        h0 = V.T @ H0V
        h1 = V.T @ H1V
        g00 = H0V.T @ H0V
        g01 = H0V.T @ H1V + H1V.T @ H0V
        g11 = H1V.T @ H1V
        h0, h1, g00, g01, g11 = tuple(
            (A + A.T) * 0.5 for A in (h0, h1, g00, g01, g11)
        )
        # A thin offline QR yields a stable, dimension-independent online
        # residual norm.  Directly forming y'G(E)y-lambda^2 loses digits once
        # the Ritz residual is small.
        _, residual_R = np.linalg.qr(np.column_stack((H0V, H1V, V)), mode="reduced")
        return {"h0": h0, "h1": h1, "g00": g00, "g01": g01,
                "g11": g11, "residual_R": residual_R}

    @staticmethod
    def indicator_and_grad(data, E: float):
        h0, h1 = data["h0"], data["h1"]
        E = float(E)
        h = h0 + E * h1
        vals, Q = np.linalg.eigh(h)
        lam = float(vals[0]); y = Q[:, 0]
        lamp = float(y @ h1 @ y)
        if len(vals) > 1:
            den = lam - vals[1:]
            coeff = Q[:, 1:].T @ h1 @ y
            # Exact away from crossings; a crossing is a genuine nonsmooth point.
            safe = np.abs(den) > 1e-11
            yp = Q[:, 1:][:, safe] @ (coeff[safe] / den[safe]) if np.any(safe) else np.zeros_like(y)
        else:
            yp = np.zeros_like(y)
        R = data["residual_R"]
        c = np.concatenate((y, E * y, -lam * y))
        cp = np.concatenate((yp, y + E * yp, -lamp * y - lam * yp))
        rcoord = R @ c
        rpcoord = R @ cp
        raw = float(rcoord @ rcoord)
        grad = float(2.0 * rcoord @ rpcoord)
        return raw, grad

    @staticmethod
    def indicator_values(data, points: np.ndarray):
        return np.array([HaldaneChain.indicator_and_grad(data, x)[0] for x in points])



def select_continuous(model: HaldaneChain, data, x0: float, maxiter: int):
    observed = BestObserved(
        lambda x: model.indicator_and_grad(data, float(x[0]))
    )
    start = np.array([np.clip(x0, *model.domain)])

    def fg(x):
        value, grad = observed.evaluate(x)
        denom = max(value, 1e-28)
        return -np.log(denom), np.array([-grad / denom])

    t0 = time.perf_counter()
    res = optimize.minimize(
        fg, start, jac=True, method="L-BFGS-B", bounds=[model.domain],
        options={"maxiter": maxiter, "maxfun": max(80, 12 * maxiter),
                 "ftol": 1e-14, "gtol": 1e-10, "maxls": 30},
    )
    observed.ensure(start)
    observed.ensure(res.x)
    point, value = observed.result()
    elapsed = time.perf_counter() - t0
    x = float(point[0])
    return x, SearchInfo(observed.calls, int(res.nit), elapsed), {
        "start": float(start[0]), "end": x, "optimizer_end": float(res.x[0]),
        "success": bool(res.success), "message": str(res.message),
        "indicator": value, "search_return_policy": "best_feasible_evaluation",
    }

def train(model: HaldaneChain, method: str, seed: int, rank: int, pool_size: int | None = None):
    rng = np.random.default_rng(seed)
    if method.startswith("contms_"):
        _, nstarts_text, _ = method.split("_")
        nstarts = int(nstarts_text)
    else:
        nstarts = 1
    # Start j at greedy round k is keyed independently.  Hence the K-start
    # experiment is exactly nested in the (K+1)-start experiment.
    anchors = np.empty((rank, nstarts))
    for k in range(rank):
        for j in range(nstarts):
            start_rng = np.random.default_rng(np.random.SeedSequence([seed, k, j, 2026]))
            anchors[k, j] = start_rng.uniform(*model.domain)
    pool = None
    if method == "pool":
        if pool_size is None or pool_size < 3:
            raise ValueError("pool_size must be at least three")
        pool = np.r_[model.domain[0], model.domain[1],
                     rng.uniform(*model.domain, size=pool_size - 2)]

    V = np.empty((model.n, 0))
    selected: list[float] = []
    traces: list[dict] = []
    total = SearchInfo()
    for k in range(rank):
        if k == 0:
            E = 1.0
            trace = {"start": 1.0, "end": 1.0, "indicator": None, "initial_snapshot": True}
        else:
            data = model.reduced_data(V)
            if method.startswith("cont_"):
                steps = int(method.split("_")[1])
                E, info, trace = select_continuous(model, data, anchors[k, 0], steps)
            elif method.startswith("contms_"):
                _, _, steps_text = method.split("_")
                steps = int(steps_text)
                candidates = []
                info = SearchInfo()
                for x0 in anchors[k]:
                    xj, ij, tj = select_continuous(model, data, float(x0), steps)
                    candidates.append((tj["indicator"], xj, tj))
                    info.objective_calls += ij.objective_calls
                    info.iterations += ij.iterations
                    info.seconds += ij.seconds
                candidates.sort(key=lambda x: x[0], reverse=True)
                _, E, best_trace = candidates[0]
                trace = {
                    "starts": [float(x) for x in anchors[k]],
                    "end": float(E),
                    "indicator": float(candidates[0][0]),
                    "candidates": [x[2] for x in candidates],
                    "nstarts": nstarts,
                }
            elif method == "pool":
                t0 = time.perf_counter()
                values = model.indicator_values(data, pool)
                E = float(pool[int(np.argmax(values))])
                info = SearchInfo(len(pool), 0, time.perf_counter() - t0)
                trace = {"end": E, "indicator": float(np.max(values)), "pool_size": len(pool)}
            elif method == "oracle_grid":
                grid = np.linspace(*model.domain, 4001)
                t0 = time.perf_counter()
                values = model.indicator_values(data, grid)
                E = float(grid[int(np.argmax(values))])
                info = SearchInfo(len(grid), 0, time.perf_counter() - t0)
                trace = {"end": E, "indicator": float(np.max(values)), "grid_size": len(grid)}
            else:
                raise ValueError(method)
            total.objective_calls += info.objective_calls
            total.iterations += info.iterations
            total.seconds += info.seconds

        u = model.truth(E)
        V, added = orth_append(V, u)
        selected.append(E)
        trace["added"] = bool(added)
        traces.append(trace)
        if not added:
            break
    return V, np.array(selected), total, traces


def true_errors(V: np.ndarray, Utest: np.ndarray):
    overlaps = V.T @ Utest
    sq = np.maximum(0.0, 1.0 - np.sum(overlaps * overlaps, axis=0))
    return np.sqrt(sq)


def gradient_check(model: HaldaneChain):
    V = np.empty((model.n, 0))
    for E in (0.15, 0.83, 1.61):
        V, _ = orth_append(V, model.truth(E))
    data = model.reduced_data(V)
    rows = []
    for E in (0.21, 0.57, 1.17, 1.83):
        val, grad = model.indicator_and_grad(data, E)
        h = 2e-6
        vp = model.indicator_and_grad(data, E + h)[0]
        vm = model.indicator_and_grad(data, E - h)[0]
        fd = (vp - vm) / (2 * h)
        rows.append({"E": E, "value": val, "analytic": grad, "finite_difference": fd,
                     "relative_error": abs(grad - fd) / max(abs(fd), 1e-12)})
    return rows


def run(args):
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    model = HaldaneChain(args.L)
    symmetry_checks = {
        "H0_asymmetry": float(sparse.linalg.norm(model.H0 - model.H0.T)),
        "H1_asymmetry": float(sparse.linalg.norm(model.H1 - model.H1.T)),
    }
    checks = gradient_check(model)
    print(f"L={args.L} sector_dim={model.n} nnz=({model.H0.nnz},{model.H1.nnz})", flush=True)
    print("gradient check max rel", max(x["relative_error"] for x in checks), flush=True)

    test_grid = np.linspace(*model.domain, args.test_size)
    Utest = np.empty((model.n, len(test_grid)))
    gaps = np.empty(len(test_grid))
    t0 = time.perf_counter()
    prev = None
    for j, E in enumerate(test_grid):
        _, Utest[:, j], gaps[j] = model.truth(float(E), need_gap=True)
        if (j + 1) % max(1, len(test_grid) // 10) == 0:
            print(f"truth grid {j+1}/{len(test_grid)}", flush=True)
    truth_seconds = time.perf_counter() - t0

    rows = []
    methods = [("cont_8", None), ("cont_100", None)]
    methods += [(f"contms_{k}_8", None) for k in args.multistarts]
    methods += [("pool", p) for p in args.pools]
    if args.oracle:
        methods.append(("oracle_grid", None))
    for seed in args.seeds:
        for method, pool_size in methods:
            label = f"pool_{pool_size}" if method == "pool" else method
            V, chosen, info, traces = train(model, method, seed, args.rank, pool_size)
            err = true_errors(V, Utest)
            wi = int(np.argmax(err))
            row = {
                "method": label, "seed": int(seed), "rank": int(V.shape[1]),
                "max_true_projection_error": float(err[wi]),
                "mean_true_projection_error": float(np.mean(err)),
                "worst_E": float(test_grid[wi]),
                "objective_calls": int(info.objective_calls),
                "optimizer_iterations": int(info.iterations),
                "selection_seconds": float(info.seconds),
                "selected_E": chosen.tolist(), "trace": traces,
            }
            rows.append(row)
            print(label, seed, "max", row["max_true_projection_error"],
                  "worst", row["worst_E"], "calls", row["objective_calls"], flush=True)

    summary = {}
    for name in sorted({r["method"] for r in rows}):
        rr = [r for r in rows if r["method"] == name]
        x = np.array([r["max_true_projection_error"] for r in rr])
        summary[name] = {
            "values": x.tolist(), "median": float(np.median(x)),
            "mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "objective_calls_mean": float(np.mean([r["objective_calls"] for r in rr])),
            "selection_seconds_mean": float(np.mean([r["selection_seconds"] for r in rr])),
        }

    result = {
        "case": "Haldane spin-1 chain, D=0, E/J in [0,2]",
        "reference": "Brehmer et al., Phys. Rev. E 108, 025306 (2023), Eq. (21)",
        "boundary": "open", "J": 1.0, "L": args.L,
        "full_dimension": int(3**args.L), "sector": "sum Sz mod 2 = 0",
        "sector_dimension": int(model.n), "rank_budget": args.rank,
        "test_grid_size": len(test_grid), "truth_grid_seconds": truth_seconds,
        "truth_metric": "max ground-state orthogonal projection error on independent uniform grid",
        "search_metric": "squared full Ritz residual from affine reduced tensors",
        "single_start": True, "continuous_parameterization": "direct bounded E/J",
        "common_initial_snapshot": 1.0,
        "pool_protocol": "fixed per run; includes both endpoints; remaining points iid uniform",
        "minimum_test_gap": float(np.min(gaps)),
        "minimum_gap_E": float(test_grid[int(np.argmin(gaps))]),
        "symmetry_checks": symmetry_checks, "gradient_checks": checks,
        "seeds": list(args.seeds), "rows": rows, "summary": summary,
    }
    path = outdir / f"haldane_L{args.L}_rank{args.rank}.json"
    path.write_text(json.dumps(result, indent=2))
    print("saved", path, flush=True)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--rank", type=int, default=10)
    p.add_argument("--test-size", type=int, default=401)
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    p.add_argument("--pools", nargs="+", type=int, default=[10, 20, 50, 100])
    p.add_argument("--multistarts", nargs="+", type=int, default=[])
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--output", default="haldane_results")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
