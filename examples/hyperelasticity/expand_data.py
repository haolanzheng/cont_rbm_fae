"""Append independent training GRFs without changing held-out fields or scaling."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import scipy.sparse.linalg as spla

from contgreedy.data.grf import matrices


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def expand(parent, output, train_count, seed, grf):
    parent, output = Path(parent), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    meta = json.loads((parent / "metadata.json").read_text())
    old = np.load(parent / "fields.npy", mmap_mode="r")
    if list(old.shape) != meta["shape"] or not np.isfinite(old).all():
        raise ValueError("Invalid parent dataset")
    subsets = {k: old[slice(*v)] for k, v in meta["splits"].items()}
    if train_count <= len(subsets["train"]) or train_count > 8192:
        raise ValueError("Require parent train size < train_count <= 8192")
    if seed == grf["seed"]:
        raise ValueError("Use an independent seed, not the original stream")
    xy, tri, mass, stiffness, robin, factor = matrices(meta["grid_resolution"] - 1)
    coordinates = np.load(parent / "coordinates.npy", allow_pickle=False)
    np.testing.assert_array_equal(coordinates, xy.astype("float32"))
    operator = grf["delta"] * mass + grf["gamma"] * stiffness
    if grf["robin"]:
        operator += np.sqrt(grf["delta"] * grf["gamma"]) / 1.42 * robin
    solver = spla.splu(operator.tocsc())
    rng = np.random.default_rng(seed)
    nv, nt = len(subsets["validation"]), len(subsets["test"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".incomplete-", dir=output.parent))
    fields = np.lib.format.open_memmap(
        temporary / "fields.npy", mode="w+", dtype="float32",
        shape=(train_count + nv + nt, len(xy)),
    )
    fields[:len(subsets["train"])] = subsets["train"]
    for start in range(len(subsets["train"]), train_count, 32):
        size = min(32, train_count - start)
        noise = rng.standard_normal((len(tri), 3, size))
        local = np.einsum("ij,ejb->eib", factor, noise)
        rhs = np.zeros((len(xy), size))
        for vertex in range(3):
            np.add.at(rhs, tri[:, vertex], local[:, vertex])
        fields[start:start + size] = solver.solve(rhs).T.astype("float32")
        if (start - len(subsets["train"])) % 512 == 0:
            print(f"Training fields: {start + size}/{train_count}", flush=True)
    splits = {"train": [0, train_count], "validation": [train_count, train_count + nv],
              "test": [train_count + nv, train_count + nv + nt]}
    for name in ("validation", "test"):
        fields[slice(*splits[name])] = subsets[name]
        np.testing.assert_array_equal(fields[slice(*splits[name])], subsets[name])
    np.testing.assert_array_equal(fields[:len(subsets["train"])], subsets["train"])
    if not np.isfinite(fields).all():
        raise ValueError("Nonfinite generated fields")
    row_hashes = [digest(row) for row in fields]
    if len(set(row_hashes)) != len(row_hashes):
        raise ValueError("Duplicate field within or across splits")
    fields.flush()
    np.save(temporary / "coordinates.npy", coordinates)
    meta.update({
        "shape": list(fields.shape), "splits": splits,
        "field_min": float(fields.min()), "field_max": float(fields.max()),
        "extension": {
            "parent_directory": str(parent.resolve()),
            "parent_fields_sha256": hashlib.sha256((parent / "fields.npy").read_bytes()).hexdigest(),
            "parent_train_count": len(subsets["train"]), "added_train_count": train_count - len(subsets["train"]),
            "seed": seed, "grf": grf, "generation_batch_size": 32,
            "normalization": "Frozen mean/std from parent training split; held-out fields unchanged",
            "parent_split_array_sha256": {k: digest(v) for k, v in subsets.items()},
            "split_array_sha256": {k: digest(fields[slice(*v)]) for k, v in splits.items()},
            "checks": {"parent_train_prefix_exact": True, "held_out_exact": True,
                       "all_finite": True, "all_rows_unique": True},
        },
    })
    (temporary / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    del fields
    temporary.rename(output)
    return meta


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=root / "data/data_80_formal1024")
    parser.add_argument("--output", type=Path, default=root / "data/data_80_formal4096")
    parser.add_argument("--train-count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    grf = json.loads((Path(__file__).parent / "configs/input_field.json").read_text())["grf"]
    print(json.dumps(expand(args.parent, args.output, args.train_count, args.seed, grf), indent=2))


if __name__ == "__main__":
    main()
