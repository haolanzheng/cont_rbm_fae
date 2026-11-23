import json
import sys
from pathlib import Path

import pytest

from examples.hyperelasticity import fae_scaling


@pytest.mark.parametrize("full_score, expected_stages", [(0.2, 3), (0.08, 2)])
def test_scaling_sequence_uses_validation_only(
    tmp_path, monkeypatch, full_score, expected_stages
):
    calls = []

    def fake_train(command, **kwargs):
        directory = Path(command[command.index("--run-dir") + 1])
        directory.mkdir()
        score = [0.4, full_score, 0.15][len(calls)]
        calls.append(command)
        (directory / "metrics.json").write_text(
            json.dumps(
                {
                    "best_validation_relative_L2": score,
                    "test": {"relative_L2": {"mean": 999}},
                }
            )
        )
        (directory / "history.json").write_text(
            json.dumps([{"validation": {"relative_L2": {"mean": score + 0.01}}}])
        )

    monkeypatch.setattr(fae_scaling.subprocess, "run", fake_train)
    run = tmp_path / "experiment"
    monkeypatch.setattr(sys, "argv", ["fae_scaling", "--run-dir", str(run)])
    fae_scaling.main()
    result = json.loads((run / "experiment.json").read_text())
    assert result["status"] == "completed"
    assert len(calls) == len(result["stages"]) == expected_stages
    assert result["capacity_decision"]["train_larger_model"] == (expected_stages == 3)
    assert "--resume" not in str(calls)
    assert all(Path(c[c.index("--config") + 1]).parent == run for c in calls)
