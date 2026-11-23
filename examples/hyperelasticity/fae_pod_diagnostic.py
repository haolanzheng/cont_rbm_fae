"""Training-only weighted randomized POD comparator for FAE reconstruction.

This is an attained linear benchmark, not a certified nonlinear lower bound.
The held-out test split is deliberately never evaluated here.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def benchmark(data, seed=91271):
    data = Path(data)
    metadata = json.loads((data / "metadata.json").read_text())
    fields = np.load(data / "fields.npy", mmap_mode="r")
    n = metadata["grid_resolution"]
    one_d = np.ones(n) / (n - 1)
    one_d[[0, -1]] *= 0.5
    root_weights = np.sqrt(np.outer(one_d, one_d).ravel())
    train = np.array(fields[slice(*metadata["splits"]["train"])], dtype=np.float64)
    mean = train.mean(axis=0)
    a = (train - mean) * root_weights
    sketch_size = min(160, *a.shape)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(a @ rng.standard_normal((a.shape[1], sketch_size)), mode="reduced")
    for _ in range(2):
        z, _ = np.linalg.qr(a.T @ q, mode="reduced")
        q, _ = np.linalg.qr(a @ z, mode="reduced")
    _, _, basis = np.linalg.svd(q.T @ a, full_matrices=False)
    results = []
    for rank in (8, 32, 64, 128):
        if rank > len(basis):
            continue
        row = {"rank": rank}
        for split in ("train", "validation"):
            raw = np.array(fields[slice(*metadata["splits"][split])], dtype=np.float64)
            weighted = (raw - mean) * root_weights
            residual = weighted - (weighted @ basis[:rank].T) @ basis[:rank]
            errors = np.linalg.norm(residual, axis=1) / np.linalg.norm(raw * root_weights, axis=1)
            row[split] = {"mean": float(errors.mean()), "p95": float(np.quantile(errors, 0.95)),
                          "max": float(errors.max())}
        results.append(row)
    return {"data_directory": str(data.resolve()),
            "fields_sha256": hashlib.sha256((data / "fields.npy").read_bytes()).hexdigest(),
            "method": "training-centered randomized POD with trapezoidal L2 quadrature",
            "seed": seed, "sketch_size": sketch_size, "power_iterations": 2,
            "test_evaluated": False, "results": results}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/data_80_formal1024")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.data)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
