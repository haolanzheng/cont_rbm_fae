"""Thermal-fin experiment driver; numerical code lives in sibling modules."""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .hf import Fin
from .rom import RB, PoolCache
from .sampling import select
from .data import make_test
from .evaluate import evaluate, uncached_values, local_audit, validate


def run(p, method, seed, rank, out):
    rb = RB(p, rank)
    started = time.perf_counter()
    rb.add(p.fom(np.zeros(p.d)))
    init = time.perf_counter() - started
    anchors = np.random.default_rng(seed + 100000).uniform(-1, 1, (rank - 1, p.d))
    pool = (
        PoolCache(
            rb, np.random.default_rng(seed).uniform(-1, 1, (int(method[4:]), p.d))
        )
        if method.startswith("pool")
        else None
    )
    fresh_rng = np.random.default_rng(seed)
    records = []
    search = 0.0
    enrich = init
    audit = 0.0
    for n in range(1, rank):
        t = time.perf_counter()
        if method.startswith("resamp"):
            zz = fresh_rng.uniform(-1, 1, (int(method[6:]), p.d))
            vals = uncached_values(rb, zz)
            ii = int(vals.argmax())
            z = zz[ii]
            info = {"log_indicator_squared": float(vals[ii]), "value_calls": len(vals)}
        elif pool is not None:
            vals = pool.values()
            ii = int(vals.argmax())
            z = pool.z[ii]
            info = {"log_indicator_squared": float(vals[ii]), "value_calls": len(vals)}
        else:
            z, info = select(rb, method, anchors[n - 1])
        elapsed = time.perf_counter() - t
        search += elapsed
        info.update({"rank_before": n, "search_seconds": elapsed, "z": z.tolist()})
        t = time.perf_counter()
        if method == "local":
            j, g = rb.fg(z)
            info.update(local_audit(rb, z, j, g))
        if n in [1, 25, 50, 100, 149]:
            direct = rb.explicit(z)
            info["explicit_indicator_squared"] = direct
            info["gram_relative_difference"] = (
                abs(np.exp(info["log_indicator_squared"]) - direct) / direct
            )
        audit += time.perf_counter() - t
        t = time.perf_counter()
        rb.add(p.fom(z))
        enrich += time.perf_counter() - t
        records.append(info)
        if n % 10 == 0 or n == rank - 1:
            print(
                method,
                seed,
                "rank",
                n + 1,
                "search",
                round(search, 2),
                "enrich",
                round(enrich, 2),
                "pg",
                info.get("projected_gradient_inf"),
                flush=True,
            )
        if n % 25 == 0:
            np.savez(
                out / f"{method}_{seed}_checkpoint.npz",
                V=rb.V[:, : rb.n],
                selected=np.array([r["z"] for r in records]),
            )
    np.savez(
        out / f"{method}_{seed}_basis.npz",
        V=rb.V,
        selected=np.array([r["z"] for r in records]),
    )
    (out / f"{method}_{seed}_training.json").write_text(
        json.dumps(
            {
                "records": records,
                "search_seconds": search,
                "enrichment_seconds": enrich,
                "local_audit_seconds": audit,
            },
            indent=2,
        )
    )
    meta = np.load(out / "test_meta.npz")
    u = np.load(out / "test_u.npy", mmap_mode="r")
    audit_start = time.perf_counter()
    metrics = evaluate(
        p,
        rb.V,
        meta["z"],
        u,
        meta["norms"],
        [n for n in [1, 25, 50, 75, 100, 125, 150, 200, 250, 300] if n <= rank]
        + ([] if rank in [1, 25, 50, 75, 100, 125, 150, 200, 250, 300] else [rank]),
    )
    data = {
        "dimension": p.d,
        "dofs": p.n,
        "rank": rank,
        "seed": seed,
        "method": method,
        "raw_indicator": True,
        "q_compressed": p.q,
        "search_seconds": search,
        "enrichment_seconds": enrich,
        "offline_seconds": search + enrich,
        "local_audit_seconds": audit,
        "evaluation_seconds": time.perf_counter() - audit_start,
        "records": records,
        "metrics": metrics,
    }
    np.savez(
        out / f"{method}_{seed}_basis.npz",
        V=rb.V,
        selected=np.array([r["z"] for r in records]),
    )
    (out / f"{method}_{seed}.json").write_text(json.dumps(data, indent=2))
    print("RESULT", method, seed, json.dumps(metrics[str(rank)]), flush=True)
    (out / f"{method}_{seed}_checkpoint.npz").unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=11)
    ap.add_argument("--rank", type=int, default=150)
    ap.add_argument("--method", default="local")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="fin_d11")
    ap.add_argument("--task", choices=["validate", "test", "run"], default="validate")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    with threadpool_limits(limits=args.threads):
        t = time.perf_counter()
        p = Fin(args.d)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        print("MODEL", p.n, p.d, p.q, "seconds", time.perf_counter() - t, flush=True)
        if args.task == "validate":
            v = validate(p)
            (out / "validation.json").write_text(json.dumps(v, indent=2))
            print(json.dumps(v, indent=2))
        elif args.task == "test":
            make_test(p, out)
        else:
            run(p, args.method, args.seed, args.rank, out)


if __name__ == "__main__":
    main()
