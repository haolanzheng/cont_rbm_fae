"""Generate native-resolution GRF input fields used by Qiu--Bridges--Chen.

We discretize C=(delta I-gamma Laplacian)^-2 with P1 FEM.  Robin boundary
coefficient sqrt(delta*gamma)/1.42 follows hIPPYlib.  Element-local Gaussian
factors give an exact factor of the assembled consistent mass matrix.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[2]


from contgreedy.data.grf import matrices


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=80)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--validation-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--output-dir", default="data_80_formal1024")
    return parser.parse_args()


def main():
    args = arguments()
    cfg = json.loads((Path(__file__).parent / "configs/input_field.json").read_text())
    defaults = cfg["data"]
    if args.resolution < 3:
        raise ValueError("resolution must be at least 3")
    counts = {
        "train": args.train_count
        if args.train_count is not None
        else defaults["train"],
        "validation": args.validation_count
        if args.validation_count is not None
        else defaults["validation"],
        "test": args.test_count if args.test_count is not None else defaults["test"],
    }
    out = ROOT / "data" / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "fields.npy").exists():
        print("data already present")
        return
    g = cfg["grf"]
    N = sum(counts.values())
    rng = np.random.default_rng(g["seed"])
    elements_per_axis = args.resolution - 1
    xy, tri, M, K, R, L = matrices(elements_per_axis)
    A = g["delta"] * M + g["gamma"] * K + np.sqrt(g["delta"] * g["gamma"]) / 1.42 * R
    lu = spla.splu(A.tocsc())
    fields = np.lib.format.open_memmap(
        out / "fields.npy", mode="w+", dtype="float32", shape=(N, len(xy))
    )
    batch = 32
    for start in range(0, N, batch):
        b = min(batch, N - start)
        xi = rng.standard_normal((len(tri), 3, b))
        local = np.einsum("ij,ejb->eib", L, xi)
        rhs = np.zeros((len(xy), b))
        for a in range(3):
            np.add.at(rhs, tri[:, a], local[:, a])
        fields[start : start + b] = lu.solve(rhs).T.astype("float32")
        if start % 512 == 0:
            print("generated", start + b, "/", N, flush=True)
    fields.flush()
    arr = np.load(out / "fields.npy", mmap_mode="r")
    ntr = counts["train"]
    nv = counts["validation"]
    mean = float(arr[:ntr].mean())
    std = float(arr[:ntr].std())
    meta = {
        "shape": list(arr.shape),
        "grid_resolution": args.resolution,
        "splits": {
            "train": [0, ntr],
            "validation": [ntr, ntr + nv],
            "test": [ntr + nv, N],
        },
        "mean": mean,
        "std": std,
        "field_min": float(arr.min()),
        "field_max": float(arr.max()),
        "fem_mass_trace": float(M.diagonal().sum()),
        "sampler": "A^{-1} times exact element-local factor of consistent P1 mass",
        "coordinates": f"row-major x-fastest {args.resolution}x{args.resolution} unit square",
    }
    np.save(out / "coordinates.npy", xy.astype("float32"))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
