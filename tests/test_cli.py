import json
from pathlib import Path
import numpy as np
import pytest
from examples.baseline.cli import run_config as spectral_run
from examples.thermal_fin.cli import run_config as fin_run

ROOT = Path(__file__).resolve().parents[1]


def test_spectral_smoke_and_no_implicit_cache(tmp_path):
    config = json.loads((ROOT / "examples/baseline/configs/smoke.json").read_text())
    output = tmp_path / "spectral"
    result = spectral_run(config, output)
    assert len(result["runs"]) == 4
    assert (output / "results.json").exists()
    with pytest.raises(FileExistsError):
        spectral_run(config, output)


def test_fin_smoke_and_physical_error(tmp_path):
    from examples.thermal_fin.hf import Fin

    config = json.loads((ROOT / "examples/thermal_fin/configs/smoke.json").read_text())
    output = tmp_path / "fin"
    fin_run(config, output)
    result = json.loads((output / "adam100_11.json").read_text())
    assert len(result["records"]) == 3
    assert result["metrics"]["4"]["vertices_max"] is None
    assert json.loads((output / "completion.json").read_text())["status"] == "completed"
    p = Fin(3, 4)
    V = np.load(output / "adam100_11_basis.npz")["V"]
    meta = np.load(output / "test_meta.npz")
    truth = np.load(output / "test_u.npy")
    errors = []
    for z, u in zip(meta["z"], truth):
        A = p.matrix_from_coeff(p.coeff(z))
        y = np.linalg.solve(V.T @ (A @ V), V.T @ p.f)
        e = u - V @ y
        errors.append(np.sqrt(e @ (p.X @ e) / (u @ (p.X @ u))))
    np.testing.assert_allclose(
        max(errors), result["metrics"]["4"]["max"], rtol=1e-6, atol=1e-8
    )
