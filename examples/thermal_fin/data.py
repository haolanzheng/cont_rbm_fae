"""Boundary-enriched held-out truth set for the historical fin protocol."""

from __future__ import annotations
import time
import numpy as np
from scipy.stats import qmc


def make_test(p, out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1000)
    z = np.vstack(
        [
            2 * qmc.Sobol(p.d, scramble=True, seed=999).random_base2(12) - 1,
            rng.choice([-1.0, 1.0], size=(256, p.d)),
        ]
    )
    tilt = rng.uniform(-1, 1, (256, p.d))
    mask = rng.random(tilt.shape) < 0.5
    tilt[mask] = np.sign(tilt[mask])
    z = np.vstack([z, tilt])
    u = np.lib.format.open_memmap(
        out / "test_u.npy", mode="w+", dtype="float64", shape=(len(z), p.n)
    )
    norms = []
    start = time.perf_counter()
    for i, zz in enumerate(z):
        uu = p.fom(zz)
        u[i] = uu
        norms.append(float(uu @ (p.X @ uu)))
        if i % 512 == 0:
            print(
                "FOM TEST",
                i,
                "seconds",
                round(time.perf_counter() - start, 1),
                flush=True,
            )
    u.flush()
    np.savez(out / "test_meta.npz", z=z, norms=np.array(norms))
    print("FOM TEST DONE", time.perf_counter() - start, flush=True)
