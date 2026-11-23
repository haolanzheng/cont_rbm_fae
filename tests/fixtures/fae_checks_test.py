import json
from pathlib import Path
import pytest

pytest.importorskip("torch")
from contgreedy.fae.checks import run_checks


def test_checks_are_independent_of_working_directory(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "examples/hyperelasticity/configs/fae128.json").read_text()
    )
    monkeypatch.chdir(tmp_path)
    result = run_checks(config["model"])
    assert result["parameters"]["real_scalars"] == 496463
    assert result["pool_gradient_check"]
    assert result["all_module_gradients_finite"]
    assert [s["input"] for s in result["shapes"]] == [40, 80, 100]
    assert not list(tmp_path.iterdir())
