import json

import numpy as np
import pytest

from contgreedy.data.grf import matrices
from examples.hyperelasticity.expand_data import expand


def test_extension_preserves_parent_and_held_out(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    fields = np.random.default_rng(51).normal(size=(8, 16)).astype("float32")
    np.save(parent / "fields.npy", fields)
    np.save(parent / "coordinates.npy", matrices(3)[0].astype("float32"))
    metadata = {
        "shape": [8, 16],
        "grid_resolution": 4,
        "splits": {"train": [0, 4], "validation": [4, 6], "test": [6, 8]},
        "mean": 0.17,
        "std": 2.3,
    }
    (parent / "metadata.json").write_text(json.dumps(metadata))
    grf = {"delta": 0.4, "gamma": 0.04, "robin": True, "seed": 51}
    first, second = tmp_path / "first", tmp_path / "second"
    result = expand(parent, first, 10, 52, grf)
    expand(parent, second, 10, 52, grf)
    extended = np.load(first / "fields.npy")
    np.testing.assert_array_equal(extended[:4], fields[:4])
    np.testing.assert_array_equal(extended[10:], fields[4:])
    np.testing.assert_array_equal(extended, np.load(second / "fields.npy"))
    np.testing.assert_array_equal(fields, np.load(parent / "fields.npy"))
    assert result["shape"] == [14, 16]
    assert result["mean"] == metadata["mean"] and result["std"] == metadata["std"]
    assert all(result["extension"]["checks"].values())
    with pytest.raises(FileExistsError):
        expand(parent, first, 10, 52, grf)
    with pytest.raises(ValueError, match="independent seed"):
        expand(parent, tmp_path / "bad", 10, 51, grf)
