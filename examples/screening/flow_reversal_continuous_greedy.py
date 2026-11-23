"""Single-start continuous greedy for a natural flow-reversal problem.

    -eps u'' + mu u' = 1 on (0,1),  u(0)=u(1)=0,  mu in [-1,1].

The truth discretization is conforming P1 Galerkin.  Search maximizes the
K-dual residual normalized by the K norm of the RB solution.  Deployment is
judged by the maximum relative K-error against the finite-element truth.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from contgreedy.sampling.host import BestObserved
from scipy import optimize, sparse
from scipy.sparse.linalg import splu


def k_append(V, u, K, tol=1e-11):
    q = np.asarray(u, float).copy()
    for _ in range(2):
        if V.size:
            q -= V @ (V.T @ (K @ q))
    nq = float(np.sqrt(max(q @ (K @ q), 0.0)))
    if nq < tol:
        return V, False
    return np.column_stack((V, q / nq)), True


class FlowReversal:
    def __init__(self, nx=1000, epsilon=5e-3):
        self.n = int(nx)
        self.epsilon = float(epsilon)
        self.domain = (-1.0, 1.0)
        h = 1.0 / (self.n + 1)
        self.x = h * np.arange(1, self.n + 1)
        self.K = sparse.diags(
            [-np.ones(self.n - 1) / h, 2 * np.ones(self.n) / h,
             -np.ones(self.n - 1) / h], [-1, 0, 1], format="csc"
        )
        self.C = sparse.diags(
            [-0.5 * np.ones(self.n - 1), 0.5 * np.ones(self.n - 1)],
            [-1, 1], format="csc"
        )
        self.f = np.full(self.n, h)
        self.Klu = splu(self.K)
        self._truth_cache = {}

    def truth(self, mu):
        key = round(float(mu), 14)
        if key not in self._truth_cache:
            A = self.epsilon * self.K + key * self.C
            self._truth_cache[key] = splu(A.tocsc()).solve(self.f)
        return self._truth_cache[key]

    def reduced_data(self, V):
        KV = self.K @ V
        CV = self.C @ V
        return {
            "V": V, "KV": KV, "CV": CV,
            "kr": V.T @ KV, "cr": V.T @ CV,
            "fr": V.T @ self.f,
        }

    def indicator_and_grad(self, data, mu):
        mu = float(mu)
        V, KV, CV = data["V"], data["KV"], data["CV"]
        Ar = self.epsilon * data["kr"] + mu * data["cr"]
        c = np.linalg.solve(Ar, data["fr"])
        cp = -np.linalg.solve(Ar, data["cr"] @ c)
        u = V @ c
        up = V @ cp
        Au = self.epsilon * (KV @ c) + mu * (CV @ c)
        Aup = self.epsilon * (KV @ cp) + mu * (CV @ cp)
        r = self.f - Au
        rp = -(CV @ c) - Aup
        zr = self.Klu.solve(r)
        q = float(r @ zr)
        qp = float(2.0 * rp @ zr)
        den = float(c @ data["kr"] @ c)
        denp = float(2.0 * cp @ data["kr"] @ c)
        value = q / max(den, 1e-30)
        grad = qp / max(den, 1e-30) - q * denp / max(den, 1e-30) ** 2
        return max(value, 0.0), grad

    def indicator_values(self, data, points):
        return np.array([self.indicator_and_grad(data, z)[0] for z in points])

    def deployment_errors(self, V, points):
        data = self.reduced_data(V)
        out = np.empty(len(points))
        for j, mu in enumerate(points):
            u = self.truth(float(mu))
            Ar = self.epsilon * data["kr"] + float(mu) * data["cr"]
            c = np.linalg.solve(Ar, data["fr"])
            e = u - V @ c
            out[j] = np.sqrt(max(e @ (self.K @ e), 0.0) / max(u @ (self.K @ u), 1e-30))
        return out



def select_continuous(model, data, x0, steps):
    observed = BestObserved(lambda x: model.indicator_and_grad(data, x[0]))
    start = np.array([np.clip(x0, *model.domain)])

    def fg(x):
        value, grad = observed.evaluate(x)
        denom = max(value, 1e-30)
        return -np.log(denom), np.array([-grad / denom])

    t0 = time.perf_counter()
    res = optimize.minimize(
        fg, start, jac=True, method="L-BFGS-B", bounds=[model.domain],
        options={"maxiter": steps, "maxfun": max(80, 12 * steps),
                 "ftol": 1e-14, "gtol": 1e-10, "maxls": 30},
    )
    observed.ensure(start)
    observed.ensure(res.x)
    point, value = observed.result()
    return float(point[0]), observed.calls, int(res.nit), time.perf_counter() - t0, {
        "start": float(start[0]), "end": float(point[0]),
        "optimizer_end": float(res.x[0]), "success": bool(res.success),
        "indicator": value, "message": str(res.message),
        "search_return_policy": "best_feasible_evaluation",
    }

def train(model, method, seed, rank, pool_size=None):
    rng = np.random.default_rng(seed)
    if method.startswith("contms_"):
        _, nstarts_text, _ = method.split("_")
        nstarts = int(nstarts_text)
    else:
        nstarts = 1
    anchors = np.empty((rank, nstarts))
    for k in range(rank):
        for j in range(nstarts):
            start_rng = np.random.default_rng(np.random.SeedSequence([seed, k, j, 2026]))
            anchors[k, j] = start_rng.uniform(*model.domain)
    pool = None
    if method == "pool":
        # Include the two boundaries and the reversal point in every pool.
        pool = np.r_[model.domain[0], 0.0, model.domain[1],
                     rng.uniform(*model.domain, size=pool_size - 3)]
    V = np.empty((model.n, 0)); selected = []; trace = []
    calls = nit = 0; seconds = 0.0
    for k in range(rank):
        if k == 0:
            mu = 0.0
            tr = {"initial_snapshot": True, "end": mu}
        else:
            data = model.reduced_data(V)
            if method.startswith("cont_"):
                mu, nc, ni, sec, tr = select_continuous(
                    model, data, anchors[k, 0], int(method.split("_")[1]))
                calls += nc; nit += ni; seconds += sec
            elif method.startswith("contms_"):
                _, _, steps_text = method.split("_")
                candidates = []
                for x0 in anchors[k]:
                    xj, nc, ni, sec, tj = select_continuous(
                        model, data, float(x0), int(steps_text))
                    calls += nc; nit += ni; seconds += sec
                    candidates.append((tj["indicator"], xj, tj))
                candidates.sort(key=lambda x: x[0], reverse=True)
                _, mu, best = candidates[0]
                tr = {"starts": anchors[k].tolist(), "end": float(mu),
                      "indicator": float(candidates[0][0]), "nstarts": nstarts,
                      "candidates": [x[2] for x in candidates]}
            elif method == "pool":
                t0 = time.perf_counter(); vals = model.indicator_values(data, pool)
                mu = float(pool[int(np.argmax(vals))])
                calls += len(pool); seconds += time.perf_counter() - t0
                tr = {"end": mu, "indicator": float(vals.max()), "pool_size": len(pool)}
            elif method == "oracle_grid":
                grid = np.linspace(*model.domain, 10001)
                t0 = time.perf_counter(); vals = model.indicator_values(data, grid)
                mu = float(grid[int(np.argmax(vals))])
                calls += len(grid); seconds += time.perf_counter() - t0
                tr = {"end": mu, "indicator": float(vals.max()), "grid_size": len(grid)}
            else:
                raise ValueError(method)
        V, added = k_append(V, model.truth(mu), model.K)
        selected.append(mu); tr["added"] = bool(added); trace.append(tr)
        if not added:
            break
    return V, np.array(selected), {"calls": calls, "iterations": nit,
                                    "seconds": seconds, "trace": trace}


def gradient_check(model):
    V = np.empty((model.n, 0))
    for mu in (-0.8, 0.0, 0.7):
        V, _ = k_append(V, model.truth(mu), model.K)
    data = model.reduced_data(V); rows = []
    for mu in (-0.55, -0.07, 0.09, 0.61):
        val, grad = model.indicator_and_grad(data, mu)
        h = 1e-6
        fd = (model.indicator_and_grad(data, mu + h)[0] -
              model.indicator_and_grad(data, mu - h)[0]) / (2 * h)
        rows.append({"mu": mu, "value": val, "gradient": grad, "finite_difference": fd,
                     "relative_error": abs(grad - fd) / max(abs(fd), 1e-12)})
    return rows


def run(args):
    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    model = FlowReversal(args.nx, args.epsilon)
    checks = gradient_check(model)
    print("gradient check", max(x["relative_error"] for x in checks), flush=True)
    test = np.linspace(*model.domain, args.test_size)
    rows = []
    methods = [("cont_8", None), ("cont_100", None)]
    methods += [(f"contms_{k}_8", None) for k in args.multistarts]
    methods += [("pool", p) for p in args.pools]
    if args.oracle: methods.append(("oracle_grid", None))
    for seed in args.seeds:
        for method, psize in methods:
            label = f"pool_{psize}" if method == "pool" else method
            V, selected, info = train(model, method, seed, args.rank, psize)
            errors = model.deployment_errors(V, test)
            wi = int(np.argmax(errors))
            row = {"method": label, "seed": seed, "rank": V.shape[1],
                   "max_relative_K_error": float(errors[wi]),
                   "mean_relative_K_error": float(errors.mean()),
                   "worst_mu": float(test[wi]), "selected_mu": selected.tolist(),
                   "objective_calls": info["calls"], "optimizer_iterations": info["iterations"],
                   "selection_seconds": info["seconds"], "trace": info["trace"]}
            rows.append(row)
            print(label, seed, "max", row["max_relative_K_error"],
                  "worst", row["worst_mu"], "calls", row["objective_calls"], flush=True)
    summary = {}
    for name in sorted({r["method"] for r in rows}):
        rr = [r for r in rows if r["method"] == name]
        vals = np.array([r["max_relative_K_error"] for r in rr])
        summary[name] = {"values": vals.tolist(), "median": float(np.median(vals)),
                         "mean": float(vals.mean()),
                         "std": float(vals.std(ddof=1)) if len(vals)>1 else 0.0,
                         "objective_calls_mean": float(np.mean([r["objective_calls"] for r in rr]))}
    result = {"case": "steady one-dimensional flow reversal", "equation": "-eps*u_xx+mu*u_x=1",
              "domain": [0, 1], "parameter_domain": [-1, 1], "epsilon": args.epsilon,
              "nx": args.nx, "rank_budget": args.rank, "test_size": args.test_size,
              "common_initial_snapshot": 0.0, "single_start": True,
              "pool_protocol": "fixed; includes -1, 0, 1 plus iid uniform points",
              "search_metric": "K-dual residual squared divided by RB K-norm squared",
              "deployment_metric": "relative K-norm error on independent uniform test grid",
              "gradient_checks": checks, "seeds": args.seeds, "rows": rows, "summary": summary}
    path = outdir / f"flow_reversal_eps{args.epsilon:g}_rank{args.rank}.json"
    path.write_text(json.dumps(result, indent=2)); print("saved", path, flush=True)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=1000)
    p.add_argument("--epsilon", type=float, default=5e-3)
    p.add_argument("--rank", type=int, default=10)
    p.add_argument("--test-size", type=int, default=2001)
    p.add_argument("--seeds", type=int, nargs="+", default=[11,22,33,44,55])
    p.add_argument("--pools", type=int, nargs="+", default=[20,50,100,500])
    p.add_argument("--multistarts", type=int, nargs="+", default=[])
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--output", default="flow_reversal_results")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
