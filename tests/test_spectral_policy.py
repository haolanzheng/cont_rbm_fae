import json
from pathlib import Path

import pytest

from examples.baseline.cli import run_config


def test_paper_cli_rejects_legacy_final_iterate_before_creating_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "examples/baseline/configs/smoke.json").read_text())
    config["adam_policy"] = {
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "return_best": False,
    }
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="best-iterate"):
        run_config(config, output)
    assert not output.exists()
